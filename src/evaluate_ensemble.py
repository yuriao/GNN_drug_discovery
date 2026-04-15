from __future__ import annotations
"""Evaluate: 5-Fold Enhanced GIN + Morgan RF on ogbg-molhiv test set."""

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
    p.add_argument("--gnn-dirs",  nargs="*", default=[])
    p.add_argument("--kfold-dir", required=True)
    p.add_argument("--rf-path",   default=None)
    p.add_argument("--outfile",   default="outputs/ensemble_summary.json")
    args = p.parse_args()

    dataset     = PygGraphPropPredDataset(name="ogbg-molhiv")
    split_idx   = dataset.get_idx_split()
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=256, shuffle=False)
    device      = torch.device("cpu")

    import pandas as pd
    raw_path = Path(dataset.root) / "mapping" / "mol.csv.gz"
    df_smi   = pd.read_csv(raw_path)
    scol     = next(c for c in df_smi.columns if "smiles" in c.lower())
    all_smi  = df_smi[scol].tolist()
    test_idx = split_idx["test"].tolist()
    test_smi = [all_smi[i] for i in test_idx]

    # True labels
    y_true = []
    for batch in test_loader:
        y    = batch.y.view(-1).float()
        mask = ~torch.isnan(y)
        y_true.extend(y[mask].numpy().tolist())
    y_true = np.array(y_true)

    # ── K-Fold Enhanced GIN ───────────────────────────────────────────────────
    kfold_ens = load_kfold_ensemble(args.kfold_dir, device)
    if kfold_ens is None:
        raise RuntimeError(f"No fold checkpoints found in {args.kfold_dir}")

    kfold_probs, offset = [], 0
    for batch in test_loader:
        y    = batch.y.view(-1).float()
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            offset += batch.num_graphs; continue
        batch    = batch.to(device)
        bsmi     = test_smi[offset : offset + batch.num_graphs]
        offset  += batch.num_graphs
        p = kfold_ens.predict_batch(batch, bsmi)
        kfold_probs.extend(p[mask.numpy().astype(bool)].tolist())

    kfold_probs = np.array(kfold_probs)
    kfold_auc   = float(roc_auc_score(y_true, kfold_probs))
    print(f"5-Fold Enhanced GIN test AUC: {kfold_auc:.4f}")

    kfold_sf      = Path(args.kfold_dir) / "kfold_summary.json"
    kfold_val_auc = 0.80
    if kfold_sf.exists():
        kfold_val_auc = float(json.loads(kfold_sf.read_text()).get("best_val_roc_auc", 0.80))

    # ── Morgan RF ─────────────────────────────────────────────────────────────
    rf_probs, rf_val_auc = None, 0.0
    if args.rf_path and Path(args.rf_path).exists():
        from src.models.morgan_rf import MorganRFClassifier
        rf     = MorganRFClassifier.load(args.rf_path)
        rf_raw = rf.predict_proba_smiles(test_smi)

        rf_masked, offset = [], 0
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

    # ── Blend ─────────────────────────────────────────────────────────────────
    if rf_probs is not None:
        bloc_vals   = [kfold_val_auc, rf_val_auc]
        weights     = _val_auc_weights(bloc_vals, temperature=0.05)
        final_probs = weights[0] * kfold_probs + weights[1] * rf_probs
        print(f"Weights: GIN_KFold={weights[0]:.3f}, RF={weights[1]:.3f}")
    else:
        final_probs = kfold_probs

    final_auc = float(roc_auc_score(y_true, final_probs))
    print(f"\n{'='*50}")
    print(f"FINAL ENSEMBLE Test ROC-AUC: {final_auc:.4f}")
    print(f"{'='*50}")

    members = {}
    if kfold_sf.exists():
        members["kfold_enhanced_gin"] = json.loads(kfold_sf.read_text())
    if args.rf_path:
        rf_sf2 = Path(args.rf_path).parent / "run_summary.json"
        if rf_sf2.exists():
            members["morgan_rf"] = json.loads(rf_sf2.read_text())

    result = {
        "ensemble_test_roc_auc": final_auc,
        "members": list(members.keys()),
        "strategy": "5-Fold Enhanced GIN (VNode+JK+RDKit) + Morgan RF | val-AUC softmax",
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
