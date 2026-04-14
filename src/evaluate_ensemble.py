from __future__ import annotations
"""Evaluate the full 4-member ensemble (GIN+GCN+GINE+MorganRF) on ogbg-molhiv test set."""

import argparse, json
from pathlib import Path

import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from src.models.ensemble import load_full_ensemble


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gnn-dirs", nargs="+", required=True,
                   help="Output dirs for GIN, GCN, GINE e.g. outputs/run_gin outputs/run_gcn outputs/run_gine")
    p.add_argument("--rf-path",  default=None, help="Path to RF .joblib checkpoint")
    p.add_argument("--outfile",  default="outputs/ensemble_summary.json")
    args = p.parse_args()

    gnn_run_dirs = {}
    for d in args.gnn_dirs:
        name = Path(d).name.replace("run_", "")  # "run_gin" → "gin"
        gnn_run_dirs[name] = d

    print(f"GNN dirs: {gnn_run_dirs}")
    print(f"RF path:  {args.rf_path}")

    dataset   = PygGraphPropPredDataset(name="ogbg-molhiv")
    split_idx = dataset.get_idx_split()

    # Load SMILES for RF
    import pandas as pd
    raw_path = Path(dataset.root) / "mapping" / "mol.csv.gz"
    df_smi   = pd.read_csv(raw_path)
    scol     = next(c for c in df_smi.columns if "smiles" in c.lower())
    all_smi  = df_smi[scol].tolist()
    test_idx = split_idx["test"].tolist()
    test_smi = [all_smi[i] for i in test_idx]

    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=256, shuffle=False)
    device = torch.device("cpu")

    ensemble = load_full_ensemble(
        gnn_run_dirs=gnn_run_dirs,
        rf_model_path=args.rf_path,
        device=device,
    )

    y_true, y_score = [], []
    offset = 0
    for batch in test_loader:
        y    = batch.y.view(-1).float()
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            offset += batch.num_graphs
            continue
        bsmi = test_smi[offset : offset + batch.num_graphs]
        offset += batch.num_graphs
        prob = ensemble.predict_batch(batch, bsmi)
        y_true.append(y[mask].numpy())
        y_score.append(prob[mask.numpy().astype(bool)])

    yt  = np.concatenate(y_true)
    ys  = np.concatenate(y_score)
    auc = roc_auc_score(yt, ys)
    print(f"Full Ensemble Test ROC-AUC: {auc:.4f}")

    # Collect individual summaries
    members = {}
    for name, d in gnn_run_dirs.items():
        sf = Path(d) / "run_summary.json"
        if sf.exists():
            members[name] = json.loads(sf.read_text())
    if args.rf_path:
        rf_sf = Path(args.rf_path).parent / "run_summary.json"
        if rf_sf.exists():
            members["morgan_rf"] = json.loads(rf_sf.read_text())

    result = {
        "ensemble_test_roc_auc": auc,
        "members": list(members.keys()),
        "strategy": "GIN+GCN+GINE (val-AUC softmax) + Morgan FP+RF (softmax bloc weight)",
        "weighting": "val_auc_softmax",
        "temperature": 0.05,
        "member_summaries": members,
    }
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    Path(args.outfile).write_text(json.dumps(result, indent=2))
    print(f"Saved → {args.outfile}")


if __name__ == "__main__":
    main()
