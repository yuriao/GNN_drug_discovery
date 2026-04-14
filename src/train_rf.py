from __future__ import annotations

"""
Train Morgan Fingerprint + Random Forest on ogbg-molhiv.

Reads SMILES from the OGB dataset's mol.csv.gz mapping file,
trains RF on the scaffold-split training set, evaluates on val/test,
and saves the model + metrics for ensemble use.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from src.models.morgan_rf import MorganRFClassifier


def load_smiles_and_labels(dataset_name: str = "ogbg-molhiv"):
    """Load SMILES + labels + scaffold split from OGB."""
    from ogb.graphproppred import PygGraphPropPredDataset
    import torch

    dataset   = PygGraphPropPredDataset(name=dataset_name)
    split_idx = dataset.get_idx_split()

    # Load SMILES from mapping file
    raw_path = Path(dataset.root) / "mapping" / "mol.csv.gz"
    df = pd.read_csv(raw_path)
    smiles_col = next(c for c in df.columns if "smiles" in c.lower())
    all_smiles = df[smiles_col].tolist()

    # Extract labels
    all_labels = [dataset[i].y.view(-1)[0].item() for i in range(len(dataset))]

    def get_split(idx_tensor):
        idx = idx_tensor.tolist()
        smiles = [all_smiles[i] for i in idx]
        labels = [int(all_labels[i]) for i in idx]
        return smiles, labels

    train_smiles, train_labels = get_split(split_idx["train"])
    valid_smiles, valid_labels = get_split(split_idx["valid"])
    test_smiles,  test_labels  = get_split(split_idx["test"])

    return (train_smiles, train_labels,
            valid_smiles, valid_labels,
            test_smiles,  test_labels)


def main():
    parser = argparse.ArgumentParser(description="Train Morgan FP + RF on ogbg-molhiv")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--outdir",       type=str, default="outputs/run_rf")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading OGB molhiv SMILES + scaffold split...")
    (train_smiles, train_labels,
     valid_smiles, valid_labels,
     test_smiles,  test_labels) = load_smiles_and_labels()

    print(f"Train: {len(train_smiles)}, Val: {len(valid_smiles)}, Test: {len(test_smiles)}")

    clf = MorganRFClassifier(n_estimators=args.n_estimators, random_state=args.seed)
    clf.fit(train_smiles, train_labels)

    # Evaluate
    def evaluate(smiles, labels):
        probs = clf.predict_proba_smiles(smiles)
        mask  = [l != -1 for l in labels]
        yt = np.array([l for l, m in zip(labels, mask) if m])
        ys = np.array([p for p, m in zip(probs,  mask) if m])
        return {
            "roc_auc": float(roc_auc_score(yt, ys)),
            "pr_auc":  float(average_precision_score(yt, ys)),
        }

    val_metrics  = evaluate(valid_smiles, valid_labels)
    test_metrics = evaluate(test_smiles,  test_labels)

    print(f"Val  ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")

    # Save model
    clf.save(outdir / "best_model.joblib")

    summary = {
        "model": "morgan_rf",
        "n_estimators": args.n_estimators,
        "morgan_radius": 2,
        "morgan_nbits": 2048,
        "best_val_roc_auc": val_metrics["roc_auc"],
        "test_roc_auc_at_best_val": test_metrics["roc_auc"],
        "val_pr_auc":  val_metrics["pr_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
