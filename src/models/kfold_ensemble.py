from __future__ import annotations

"""
K-Fold ensemble inference.

Loads K checkpoints (one per fold) and averages their predictions.
Combines with the main GNN+RF ensemble via bloc-level softmax weighting.
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from src.models.factory import build_model
from src.models.gine import GINEClassifier


class KFoldEnsemble:
    """Averages predictions from K independently-trained fold models."""

    def __init__(self, models: list[nn.Module], model_names: list[str],
                 weights: Optional[list[float]] = None):
        self.models = models
        self.model_names = model_names
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        assert abs(sum(weights) - 1.0) < 1e-4
        self.weights = weights

    def to(self, device):
        self.models = [m.to(device) for m in self.models]
        self.device = device
        return self

    @torch.no_grad()
    def predict_batch(self, batch) -> np.ndarray:
        probs = []
        for model, name, w in zip(self.models, self.model_names, self.weights):
            kw = {"edge_attr": batch.edge_attr} if name == "gine" else {}
            logit = model(batch.x, batch.edge_index, batch.batch, **kw)
            probs.append(torch.sigmoid(logit).cpu().numpy() * w)
        return sum(probs)

    @torch.no_grad()
    def predict_smiles(self, smiles: str) -> float:
        import sys
        if str(Path(__file__).parent.parent.parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from src.data.smiles_to_graph import smiles_to_pyg
        from torch_geometric.data import Batch
        graph = smiles_to_pyg(smiles)
        batch = Batch.from_data_list([graph])
        if hasattr(self, "device"):
            batch = batch.to(self.device)
        return float(self.predict_batch(batch)[0])


def load_kfold_ensemble(
    kfold_dir: str | Path,
    device: torch.device,
    val_auc_weighted: bool = True,
) -> Optional[KFoldEnsemble]:
    """
    Load all fold checkpoints from kfold_dir/fold_*/best_model.pt.
    Uses val_auc softmax weights if val_auc_weighted=True.
    Returns None if no folds found.
    """
    kfold_dir = Path(kfold_dir)
    fold_dirs = sorted(kfold_dir.glob("fold_*"))
    if not fold_dirs:
        return None

    models, names, val_aucs = [], [], []

    for fd in fold_dirs:
        ckpt_path = fd / "best_model.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu")
        mtype = ckpt.get("model_type", "gin")
        nf = ckpt.get("num_node_features", 9)
        ef = ckpt.get("num_edge_features", 3)
        hd = ckpt.get("args", {}).get("hidden_dim", 128)
        nl = ckpt.get("args", {}).get("num_layers", 4)
        dp = ckpt.get("args", {}).get("dropout", 0.2)

        m = build_model(mtype, nf, hd, nl, dp, num_edge_features=ef)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models.append(m)
        names.append(mtype)
        val_aucs.append(float(ckpt.get("best_val_roc_auc", 0.75)))

    if not models:
        return None

    # Compute weights
    if val_auc_weighted and len(set(val_aucs)) > 1:
        T = 0.05
        scaled = [v / T for v in val_aucs]
        max_s  = max(scaled)
        exps   = [math.exp(s - max_s) for s in scaled]
        total  = sum(exps)
        weights = [e / total for e in exps]
    else:
        weights = [1.0 / len(models)] * len(models)

    print(f"K-Fold ensemble: {len(models)} folds loaded")
    for fd, w, v in zip(fold_dirs, weights, val_aucs):
        print(f"  {fd.name}: weight={w:.3f} val_auc={v:.4f}")

    ens = KFoldEnsemble(models, names, weights)
    ens.to(device)
    return ens
