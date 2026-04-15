from __future__ import annotations
"""
K-Fold ensemble inference — supports both vanilla GIN and EnhancedGINClassifier.
Handles global RDKit feature injection automatically based on checkpoint metadata.
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from src.models.factory import build_model


class KFoldEnsemble:
    def __init__(self, models: list[nn.Module], model_names: list[str],
                 use_global_features: list[bool], weights: Optional[list[float]] = None):
        self.models              = models
        self.model_names         = model_names
        self.use_global_features = use_global_features
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        assert abs(sum(weights) - 1.0) < 1e-4
        self.weights = weights
        self.device  = torch.device("cpu")

    def to(self, device):
        self.models = [m.to(device) for m in self.models]
        self.device = device
        return self

    @torch.no_grad()
    def predict_batch(self, batch, smiles_list: Optional[list[str]] = None) -> np.ndarray:
        """
        Run all fold models on a batch and return weighted average.
        smiles_list: SMILES for the batch molecules (needed for global features).
        """
        import torch
        probs = []
        for model, name, use_gf, w in zip(
            self.models, self.model_names, self.use_global_features, self.weights
        ):
            gf = None
            if use_gf and smiles_list is not None:
                from src.data.mol_features import batch_smiles_to_features
                gf_np = batch_smiles_to_features(smiles_list)
                gf = torch.from_numpy(gf_np).to(self.device)

            logit = model(batch.x, batch.edge_index, batch.batch, global_feat=gf)
            probs.append(torch.sigmoid(logit).cpu().numpy() * w)
        return sum(probs)

    @torch.no_grad()
    def predict_smiles(self, smiles: str) -> float:
        import sys
        _root = str(Path(__file__).parent.parent.parent)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from src.data.smiles_to_graph import smiles_to_pyg
        from torch_geometric.data import Batch
        graph = smiles_to_pyg(smiles)
        batch = Batch.from_data_list([graph]).to(self.device)
        return float(self.predict_batch(batch, [smiles])[0])


def load_kfold_ensemble(
    kfold_dir: str | Path,
    device: torch.device,
    val_auc_weighted: bool = True,
) -> Optional[KFoldEnsemble]:
    kfold_dir = Path(kfold_dir)
    fold_dirs = sorted(kfold_dir.glob("fold_*"))
    if not fold_dirs:
        return None

    models, names, use_gf_flags, val_aucs = [], [], [], []

    for fd in fold_dirs:
        ckpt_path = fd / "best_model.pt"
        if not ckpt_path.exists():
            continue
        ckpt  = torch.load(ckpt_path, map_location="cpu")
        mtype = ckpt.get("model_type", "gin_enhanced")
        nf    = ckpt.get("num_node_features", 9)
        ef    = ckpt.get("num_edge_features", 3)
        hd    = ckpt.get("args", {}).get("hidden_dim", 128)
        nl    = ckpt.get("args", {}).get("num_layers", 4)
        dp    = ckpt.get("args", {}).get("dropout", 0.2)
        uvn   = ckpt.get("use_virtual_node", True)
        ujk   = ckpt.get("use_jk", True)
        ugf   = ckpt.get("use_global_features", True)

        m = build_model(mtype, nf, hd, nl, dp,
                        num_edge_features=ef,
                        use_virtual_node=uvn,
                        use_jk=ujk,
                        use_global_features=ugf)
        m.load_state_dict(ckpt["model_state"])
        m.eval()
        models.append(m)
        names.append(mtype)
        use_gf_flags.append(ugf)
        val_aucs.append(float(ckpt.get("best_val_roc_auc", 0.75)))

    if not models:
        return None

    if val_auc_weighted and len(set(val_aucs)) > 1:
        T      = 0.05
        scaled = [v / T for v in val_aucs]
        max_s  = max(scaled)
        exps   = [math.exp(s - max_s) for s in scaled]
        total  = sum(exps)
        weights = [e / total for e in exps]
    else:
        weights = [1.0 / len(models)] * len(models)

    print(f"K-Fold ensemble: {len(models)} folds")
    for fd, w, v in zip(fold_dirs, weights, val_aucs):
        print(f"  {fd.name}: weight={w:.3f}  val_auc={v:.4f}")

    ens = KFoldEnsemble(models, names, use_gf_flags, weights)
    ens.to(device)
    return ens
