from __future__ import annotations

"""
GNN + Morgan RF Ensemble.

Four members:
  1. GIN  — sum aggregation, global add pool
  2. GCN  — mean aggregation, global mean pool
  3. GINE — sum aggregation + edge features
  4. Morgan FP + Random Forest — completely different feature space (ECFP4 2048-bit)

GNN members make complementary errors due to different aggregation schemes.
The RF member adds an orthogonal signal: it sees global substructure counts
that GNNs don't see (GNNs are bounded by 1-WL expressiveness, RF is not).

Weighting: val-AUC softmax with temperature T=0.05.
Higher val AUC → exponentially higher weight. Automatically down-weights
any member that underconverges in a given training run.

OGB leaderboard context:
  #9  MorganFP + RF alone:     0.8208 (CPU, no deep learning)
  #1  Multi-RF + Multi-GNN:    0.8476 (ensemble of both)
"""

from pathlib import Path
from typing import Optional
import math
import numpy as np
import torch
from torch import nn

from src.models.gcn  import GCNClassifier
from src.models.gin  import GINClassifier
from src.models.gine import GINEClassifier

GNN_MEMBERS = ["gin", "gcn", "gine"]
WEIGHT_TEMPERATURE = 0.05


# ── Weighting ─────────────────────────────────────────────────────────────────

def _val_auc_weights(val_aucs: list[float], temperature: float = WEIGHT_TEMPERATURE) -> list[float]:
    scaled = [v / temperature for v in val_aucs]
    max_s  = max(scaled)
    exps   = [math.exp(s - max_s) for s in scaled]
    total  = sum(exps)
    return [e / total for e in exps]


# ── GNN-only ensemble (nn.Module for .to(device) support) ────────────────────

class GNNEnsemble(nn.Module):
    def __init__(self, models: list[nn.Module], weights: Optional[list[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        assert abs(sum(weights) - 1.0) < 1e-4
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, x, edge_index, batch, edge_attr=None):
        probs = []
        for model in self.models:
            logit = (model(x, edge_index, batch, edge_attr=edge_attr)
                     if isinstance(model, GINEClassifier)
                     else model(x, edge_index, batch))
            probs.append(torch.sigmoid(logit))
        stacked = torch.stack(probs, dim=0)
        return (stacked * self.weights.view(-1, 1)).sum(dim=0)


# ── Full ensemble: GNN members + optional RF member ───────────────────────────

class FullEnsemble:
    """
    Combines GNNEnsemble (PyTorch) with MorganRFClassifier (sklearn).
    Accepts both PyG batch inputs (for GNNs) and SMILES strings (for RF).
    """

    def __init__(
        self,
        gnn_ensemble: GNNEnsemble,
        gnn_weight: float,
        rf_model,              # MorganRFClassifier or None
        rf_weight: float = 0.0,
    ):
        self.gnn = gnn_ensemble
        self.rf  = rf_model
        self.gnn_weight = gnn_weight
        self.rf_weight  = rf_weight

    @torch.no_grad()
    def predict_batch(self, batch, smiles_list: Optional[list[str]] = None) -> np.ndarray:
        """
        Returns numpy array of HIV inhibition probabilities.
        smiles_list required when RF model is present.
        """
        gnn_probs = self.gnn(
            batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
        ).cpu().numpy()

        if self.rf is not None and smiles_list is not None:
            rf_probs = self.rf.predict_proba_smiles(smiles_list)
            return self.gnn_weight * gnn_probs + self.rf_weight * rf_probs
        return gnn_probs

    def predict_smiles(self, smiles: str, batch) -> float:
        """Predict for a single molecule. batch = pre-built PyG Batch."""
        return float(self.predict_batch(batch, [smiles])[0])


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ensemble(
    checkpoints: dict[str, str | Path],
    device: torch.device,
    val_aucs: Optional[dict[str, float]] = None,
) -> GNNEnsemble:
    """Load GNN-only ensemble with optional val-AUC weighting."""
    models, raw_weights, names = [], [], []

    for name in GNN_MEMBERS:
        ckpt_path = checkpoints.get(name)
        if ckpt_path is None or not Path(ckpt_path).exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        nf = ckpt.get("num_node_features", 9)
        ef = ckpt.get("num_edge_features", 3)
        hd = ckpt.get("args", {}).get("hidden_dim", 128)
        nl = ckpt.get("args", {}).get("num_layers", 4)
        dp = ckpt.get("args", {}).get("dropout", 0.2)

        m = (GINClassifier(nf, hd, nl, dp)  if name == "gin"  else
             GCNClassifier(nf, hd, nl, dp)  if name == "gcn"  else
             GINEClassifier(nf, ef, hd, nl, dp))
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models.append(m)
        names.append(name)

        auc = (val_aucs or {}).get(name) or ckpt.get("best_val_roc_auc") or 0.75
        raw_weights.append(float(auc))

    if not models:
        raise FileNotFoundError(f"No GNN checkpoints found: {checkpoints}")

    weights = _val_auc_weights(raw_weights)
    print("GNN weights: " + ", ".join(
        f"{n}={w:.3f}(val={v:.3f})" for n, w, v in zip(names, weights, raw_weights)))
    return GNNEnsemble(models, weights).to(device)


def load_ensemble_from_dirs(
    run_dirs: dict[str, str | Path],
    device: torch.device,
) -> GNNEnsemble:
    """Load GNN ensemble reading val_auc from each dir's run_summary.json."""
    import json
    checkpoints, val_aucs = {}, {}
    for name, d in run_dirs.items():
        d = Path(d)
        if (d / "best_model.pt").exists():
            checkpoints[name] = d / "best_model.pt"
        if (d / "run_summary.json").exists():
            data = json.loads((d / "run_summary.json").read_text())
            if auc := data.get("best_val_roc_auc"):
                val_aucs[name] = float(auc)
    return load_ensemble(checkpoints, device, val_aucs=val_aucs)


def load_full_ensemble(
    gnn_run_dirs: dict[str, str | Path],
    rf_model_path: Optional[str | Path],
    device: torch.device,
) -> FullEnsemble:
    """
    Load the complete 4-member ensemble: GIN + GCN + GINE + Morgan RF.

    Weights between GNN bloc and RF bloc are determined by their
    respective val ROC-AUC scores (softmax T=0.05).
    """
    import json
    from src.models.morgan_rf import MorganRFClassifier

    # Load GNN ensemble + get its aggregate val AUC
    gnn_val_aucs = {}
    for name, d in gnn_run_dirs.items():
        sf = Path(d) / "run_summary.json"
        if sf.exists():
            data = json.loads(sf.read_text())
            if auc := data.get("best_val_roc_auc"):
                gnn_val_aucs[name] = float(auc)

    gnn_ens = load_ensemble_from_dirs(gnn_run_dirs, device)
    # Aggregate GNN val AUC = weighted average of members
    gnn_weights = gnn_ens.weights.cpu().numpy()
    gnn_names   = list(gnn_val_aucs.keys())
    gnn_aucs    = [gnn_val_aucs.get(n, 0.75) for n in gnn_names]
    # Use simple average of top GNN val AUC as the GNN bloc score
    gnn_bloc_auc = float(np.dot(gnn_weights, gnn_aucs)) if len(gnn_weights) == len(gnn_aucs) else max(gnn_aucs)

    # Load RF model
    rf, rf_val_auc = None, 0.0
    if rf_model_path and Path(rf_model_path).exists():
        rf = MorganRFClassifier.load(rf_model_path)
        rf_summary = Path(rf_model_path).parent / "run_summary.json"
        if rf_summary.exists():
            data = json.loads(rf_summary.read_text())
            rf_val_auc = float(data.get("best_val_roc_auc", 0.0))
        print(f"RF loaded — val AUC: {rf_val_auc:.4f}")

    # Softmax weights between GNN bloc and RF bloc
    blocs = [gnn_bloc_auc, rf_val_auc] if rf else [1.0]
    if rf:
        w = _val_auc_weights(blocs)
        gnn_w, rf_w = w[0], w[1]
    else:
        gnn_w, rf_w = 1.0, 0.0

    print(f"Bloc weights: GNN={gnn_w:.3f}(val={gnn_bloc_auc:.3f}), RF={rf_w:.3f}(val={rf_val_auc:.3f})")
    return FullEnsemble(gnn_ens, gnn_w, rf, rf_w)
