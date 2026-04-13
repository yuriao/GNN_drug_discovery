from __future__ import annotations

"""
GNN Ensemble — averages predictions from GIN + GCN + GINE.

Each model is trained independently with the same hyperparameters and
pos_weight. At inference, the three sigmoid probabilities are averaged
(equal weights). This exploits the fact that:
  - GIN uses sum aggregation (preserves neighbourhood size)
  - GCN uses mean aggregation + symmetric normalisation
  - GINE uses sum aggregation + edge features (bond type, aromaticity)

These three architectures make different errors → ensembling consistently
improves ROC-AUC by 1-3% on ogbg-molhiv.

Reference: OGB leaderboard top entries (Multi-RF Fusion #1, HyperFusion #2)
both use multi-model ensembles.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Batch

from src.models.gcn import GCNClassifier
from src.models.gin import GINClassifier
from src.models.gine import GINEClassifier


ENSEMBLE_MEMBERS = ["gin", "gcn", "gine"]


class GNNEnsemble(nn.Module):
    """
    Wraps three trained GNN models and returns the mean of their sigmoid outputs.
    Models are loaded from separate checkpoints; inference only (no joint training).
    """

    def __init__(self, models: list[nn.Module], weights: Optional[list[float]] = None):
        super().__init__()
        # Store as ModuleList so .to(device) propagates
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        assert abs(sum(weights) - 1.0) < 1e-5, "Weights must sum to 1"
        self.register_buffer("weights", torch.tensor(weights))

    @torch.no_grad()
    def forward(self, x, edge_index, batch, edge_attr=None):
        probs = []
        for model in self.models:
            if isinstance(model, GINEClassifier):
                logit = model(x, edge_index, batch, edge_attr=edge_attr)
            else:
                logit = model(x, edge_index, batch)
            probs.append(torch.sigmoid(logit))  # [B]
        # Weighted average of probabilities
        stacked = torch.stack(probs, dim=0)   # [n_models, B]
        avg = (stacked * self.weights.view(-1, 1)).sum(dim=0)  # [B]
        return avg


def load_ensemble(
    checkpoints: dict[str, str | Path],
    device: torch.device,
) -> GNNEnsemble:
    """
    Load ensemble from a dict of {model_name: checkpoint_path}.
    checkpoints must contain at least one of: "gin", "gcn", "gine"
    """
    models = []
    weights = []

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
        weights.append(1.0)

    if not models:
        raise FileNotFoundError(f"No ensemble checkpoints found in: {checkpoints}")

    # Normalise weights
    total = sum(weights)
    weights = [w / total for w in weights]

    ensemble = GNNEnsemble(models, weights)
    ensemble = ensemble.to(device)
    return ensemble
