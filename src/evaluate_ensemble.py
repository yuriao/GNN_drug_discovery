from __future__ import annotations
"""
Evaluate ensemble: 5-Fold GIN + Morgan RF on ogbg-molhiv test set.

Architecture:
  - 5-Fold GIN: K GIN models trained on complementary splits, predictions averaged
  - Morgan RF:  Random Forest on ECFP4 fingerprints — orthogonal feature space
  - Final:      val-AUC softmax bloc weighting between the two
"""

import argparse, json
from pathlib import Path

import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from src.models.ensemble import _val_auc_weights
from src.models.kfold_ensemble import load_kfold_ensemble


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gnn-dirs",   nargs="*", default=[], help="Kept for backward compat (ignored)")
    p.add_argument("--kfold-dir",  required=True, help="Dir containing fold_1/ fold_2/ ...")
    p.add_argument("--rf-path",    default=None)
    p.add_argument("--outfile",    default="outputs/ensemble_summary.json")
    args = p.parse_args()

    print(f"K-Fold dir: {args.kfold_dir}")
    print(f"RF path:    {args.rf_path}")

    dataset     = PygGraphPropPredDataset(name="ogbg-molhiv")
    split_idx   = dataset.get_idx_split()
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=256, shuffle=False)
    device      = torch.device("cpu")

    # SMILES for RF
    import pandas as pd
    raw_path = Path(dataset.root) / "mapping" / "mol.csv.gz"
    df_smi   = pd.read_csv(raw_path)
    scol     = next(c for c in df_smi.columns if "smiles" in c.lower())
    all_smi  = df_smi[scol].tolist()
    test_idx = split_idx["test"].tolist()
    test_smi = [all_smi[i] for i in test_idx]

    # True test labels
    y_true = []
    for batch in test_loader:
        y    = batch.y.view(-1).float()
        mask = ~torch.isnan(y)
        y_true.extend(y[mask].numpy().tolist())
    y_true = np.array(y_true)

    # ── Component 1: K-Fold GIN ensemble ─────────────────────────────────────
    kfold_ens = load_kfold_ensemble(args.kfold_dir, device)
    if kfold_ens is None:
        raise RuntimeError(f"No fold checkpoints found in {args.kfold_dir}")

    kfold_probs = []
    for batch in test_loader:
        y    = batch.y.view(-1).float()
        mask = ~torch.isnan(y)
        if mask.sum() == 0: continue
        batch = batch.to(device)
        p = kfold_ens.predict_batch(batch)
        kfold_probs.extend(p[mask.numpy().astype(bool)].tolist())
    kfold_probs = np.array(kfold_probs)
    kfold_auc   = float(roc_auc_score(y_true, kfold_probs))
    print(f"5-Fold GIN ensemble test AUC: {kfold_auc:.4f}")

    # Val AUC for weighting
    kfold_sf = Path(args.kfold_dir) / "kfold_summary.json"
    kfold_val_auc = 0.80  # default
    if kfold_sf.exists():
        kdata = json.loads(kfold_sf.read_text())
        kfold_val_auc = float(kdata.get("best_val_roc_auc", 0.80))

    # ── Component 2: Morgan RF ────────────────────────────────────────────────
    rf_probs    = None
    rf_val_auc  = 0.0
    if args.rf_path and Path(args.rf_path).exists():
        from src.models.morgan_rf import MorganRFClassifier
        rf      = MorganRFClassifier.load(args.rf_path)
        rf_raw  = rf.predict_proba_smiles(test_smi)

        # Align with test mask
        rf_masked = []
        offset = 0
        for batch in test_loader:
            y    = batch.y.view(-1).float()
            mask = ~torch.isnan(y)
            n    = batch.num_graphs
            for i in range(n):
                if mask[i]:
                    rf_masked.append(float(rf_raw[offset + i]))
            offset += n
        rf_probs = np.array(rf_masked)
        rf_auc   = float(roc_auc_score(y_true, rf_probs))

        rf_sf = Path(args.rf_path).parent / "run_summary.json"
        if rf_sf.exists():
            rf_val_auc = float(json.loads(rf_sf.read_text()).get("best_val_roc_auc", 0.0))
        print(f"Morgan RF test AUC:           {rf_auc:.4f}  (val={rf_val_auc:.4f})")

    # ── Blend with val-AUC softmax weights ────────────────────────────────────
    if rf_probs is not None:
        bloc_vals = [kfold_val_auc, rf_val_auc]
        weights   = _val_auc_weights(bloc_vals, temperature=0.05)
        print(f"Bloc weights: GIN_KFold={weights[0]:.3f}(val={bloc_vals[0]:.3f}), "
              f"RF={weights[1]:.3f}(val={bloc_vals[1]:.3f})")
        final_probs = weights[0] * kfold_probs + weights[1] * rf_probs
    else:
        final_probs = kfold_probs
        print("RF unavailable — using K-Fold GIN only")

    final_auc = float(roc_auc_score(y_true, final_probs))
    print(f"\n{'='*50}")
    print(f"FINAL ENSEMBLE Test ROC-AUC:  {final_auc:.4f}")
    print(f"{'='*50}")

    # Summaries
    members = {}
    if kfold_sf.exists():
        members["kfold_gin"] = json.loads(kfold_sf.read_text())
    if args.rf_path:
        rf_sf2 = Path(args.rf_path).parent / "run_summary.json"
        if rf_sf2.exists():
            members["morgan_rf"] = json.loads(rf_sf2.read_text())

    result = {
        "ensemble_test_roc_auc": final_auc,
        "members": list(members.keys()),
        "strategy": "5-Fold CV GIN + Morgan FP+RF (val-AUC softmax bloc weights)",
        "kfold_gin_test_auc": kfold_auc,
        "rf_test_auc": float(roc_auc_score(y_true, rf_probs)) if rf_probs is not None else None,
        "weighting": "val_auc_softmax",
        "temperature": 0.05,
        "member_summaries": members,
    }
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    Path(args.outfile).write_text(json.dumps(result, indent=2))
    print(f"Saved → {args.outfile}")


if __name__ == "__main__":
    main()
