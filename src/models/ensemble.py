from __future__ import annotations

"""
GNN Ensemble — weighted average of GIN + GCN + GINE predictions.

Weighting strategy: validation ROC-AUC softmax weights.
  Each model's weight = softmax(val_auc / temperature).
  This automatically down-weights poorly converged members (e.g. GCN at 0.65)
  without discarding them entirely — they still contribute diversity.

Temperature T=0.05: at T→0 this approaches argmax (winner-takes-all);
at T→∞ this approaches equal weights. T=0.05 gives strong preference to
the best model while keeping diversity benefit.

Reference: OGB leaderboard #1 (Multi-RF Fusion) and #2 (HyperFusion) both
use weighted ensemble strategies.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.models.gcn import GCNClassifier
from src.models.gin import GINClassifier
from src.models.gine import GINEClassifier


ENSEMBLE_MEMBERS = ["gin", "gcn", "gine"]
WEIGHT_TEMPERATURE = 0.05  # lower = more weight to best model


class GNNEnsemble(nn.Module):
    """
    Wraps trained GNN models and returns a weighted average of sigmoid outputs.
    Weights are set at construction time (from val AUC or equal).
    """

    def __init__(self, models: list[nn.Module], weights: Optional[list[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        assert abs(sum(weights) - 1.0) < 1e-4, f"Weights must sum to 1, got {sum(weights)}"
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, x, edge_index, batch, edge_attr=None):
        probs = []
        for model in self.models:
            if isinstance(model, GINEClassifier):
                logit = model(x, edge_index, batch, edge_attr=edge_attr)
            else:
                logit = model(x, edge_index, batch)
            probs.append(torch.sigmoid(logit))
        stacked = torch.stack(probs, dim=0)              # [n_models, B]
        avg = (stacked * self.weights.view(-1, 1)).sum(dim=0)  # [B]
        return avg


def _val_auc_weights(val_aucs: list[float], temperature: float = WEIGHT_TEMPERATURE) -> list[float]:
    """
    Convert validation AUC scores to softmax weights.
    Higher AUC → exponentially higher weight.
    """
    import math
    scaled = [v / temperature for v in val_aucs]
    max_s  = max(scaled)
    exps   = [math.exp(s - max_s) for s in scaled]  # numerically stable
    total  = sum(exps)
    return [e / total for e in exps]


def load_ensemble(
    checkpoints: dict[str, str | Path],
    device: torch.device,
    val_aucs: Optional[dict[str, float]] = None,
) -> GNNEnsemble:
    """
    Load ensemble from {model_name: checkpoint_path}.

    Args:
        checkpoints: dict of model name → path to .pt checkpoint
        device: torch device
        val_aucs: optional dict of model name → validation ROC-AUC.
                  If provided, uses softmax-weighted averaging.
                  If None, reads val_auc from checkpoint metadata, falling back
                  to equal weights if unavailable.

    Returns:
        GNNEnsemble ready for inference
    """
    models = []
    raw_weights = []
    names_loaded = []

    for name in ENSEMBLE_MEMBERS:
        ckpt_path = checkpoints.get(name)
        if ckpt_path is None or not Path(ckpt_path).exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        num_node_feat = ckpt.get("num_node_features", 9)
        num_edge_feat = ckpt.get("num_edge_features", 3)
        hidden_dim    = ckpt.get("args", {}).get("hidden_dim", 128)
        num_layers    = ckpt.get("args", {}).get("num_layers", 4)
        dropout       = ckpt.get("args", {}).get("dropout", 0.2)

        if name == "gin":
            m = GINClassifier(num_node_feat, hidden_dim, num_layers, dropout)
        elif name == "gcn":
            m = GCNClassifier(num_node_feat, hidden_dim, num_layers, dropout)
        elif name == "gine":
            m = GINEClassifier(num_node_feat, num_edge_feat, hidden_dim, num_layers, dropout)
        else:
            continue

        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models.append(m)
        names_loaded.append(name)

        # Determine weight for this model
        if val_aucs and name in val_aucs:
            raw_weights.append(val_aucs[name])
        else:
            # Try reading from checkpoint metadata
            saved_val = ckpt.get("best_val_roc_auc") or ckpt.get("val_roc_auc")
            raw_weights.append(float(saved_val) if saved_val else 0.75)

    if not models:
        raise FileNotFoundError(f"No ensemble checkpoints found in: {checkpoints}")

    # Compute softmax weights from val AUCs
    weights = _val_auc_weights(raw_weights, WEIGHT_TEMPERATURE)
    print(f"Ensemble weights: " +
          ", ".join(f"{n}={w:.3f}(val={v:.3f})"
                    for n, w, v in zip(names_loaded, weights, raw_weights)))

    ensemble = GNNEnsemble(models, weights)
    return ensemble.to(device)


def load_ensemble_from_dirs(
    run_dirs: dict[str, str | Path],
    device: torch.device,
) -> GNNEnsemble:
    """
    Convenience loader: reads checkpoints AND val_aucs from run_summary.json
    inside each model's output directory.

    run_dirs: {"gin": "outputs/run_gin", "gcn": "outputs/run_gcn", ...}
    """
    import json
    checkpoints = {}
    val_aucs    = {}

    for name, d in run_dirs.items():
        d = Path(d)
        ckpt = d / "best_model.pt"
        summary = d / "run_summary.json"
        if ckpt.exists():
            checkpoints[name] = ckpt
        if summary.exists():
            data = json.loads(summary.read_text())
            auc  = data.get("best_val_roc_auc")
            if auc:
                val_aucs[name] = float(auc)

    return load_ensemble(checkpoints, device, val_aucs=val_aucs)
