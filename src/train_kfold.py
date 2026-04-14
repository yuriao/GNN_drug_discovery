from __future__ import annotations

"""
K-Fold Cross-Validation training for GNN on ogbg-molhiv.

Strategy:
  - Combine train + validation splits from the scaffold split (~36,993 molecules)
  - Divide into K folds using StratifiedKFold (preserves ~3.5% positive rate per fold)
  - Train one GIN model per fold; each sees K-1 folds (~29,594 molecules), validates on 1 (~7,399)
  - Save K checkpoints — at inference, average all K models' predictions

Why this improves over single-model training:
  - Each fold model trains on more data than the original train split alone
  - K models make different errors → ensemble reduces variance
  - No information leakage: test set is never touched

Reference: This is the standard competition approach — OGB leaderboard top entries
all use ensemble strategies derived from multiple training runs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ogb.graphproppred import PygGraphPropPredDataset
from src.models.factory import build_model
from src.utils.metrics import safe_roc_auc, safe_pr_auc
from src.utils.seed import set_seed


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, device, model_name="gin"):
    model.train()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(-1).float()
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        optimizer.zero_grad()
        kw = {"edge_attr": batch.edge_attr} if model_name == "gine" else {}
        logits = model(batch.x, batch.edge_index, batch.batch, **kw)
        loss = criterion(logits[mask], y[mask])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(mask.sum())
        total_n    += int(mask.sum())
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device, model_name="gin"):
    model.eval()
    y_true, y_score = [], []
    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(-1).float()
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        kw = {"edge_attr": batch.edge_attr} if model_name == "gine" else {}
        logits = model(batch.x, batch.edge_index, batch.batch, **kw)
        probs  = torch.sigmoid(logits)
        y_true.append(y[mask].cpu().numpy())
        y_score.append(probs[mask].cpu().numpy())
    if not y_true:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    yt = np.concatenate(y_true)
    ys = np.concatenate(y_score)
    return {"roc_auc": safe_roc_auc(yt, ys), "pr_auc": safe_pr_auc(yt, ys)}


# ── K-Fold main ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="K-Fold CV GNN training on ogbg-molhiv")
    p.add_argument("--model",        choices=["gcn","gin","gine"], default="gin")
    p.add_argument("--k-folds",      type=int,   default=5)
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--hidden-dim",   type=int,   default=128)
    p.add_argument("--num-layers",   type=int,   default=4)
    p.add_argument("--dropout",      type=float, default=0.2)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--pos-weight",   type=float, default=0.0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--outdir",       type=str,   default="outputs/run_kfold")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ogbg-molhiv ...")
    dataset   = PygGraphPropPredDataset(name="ogbg-molhiv")
    split_idx = dataset.get_idx_split()

    # Combine train + val (keep test held out — never touched)
    train_idx = split_idx["train"].numpy()
    valid_idx = split_idx["valid"].numpy()
    test_idx  = split_idx["test"].numpy()
    tv_idx    = np.concatenate([train_idx, valid_idx])  # train+val pool

    # Get labels for stratification
    all_labels = np.array([
        dataset[int(i)].y.view(-1)[0].item() for i in range(len(dataset))
    ])
    tv_labels  = all_labels[tv_idx]
    test_data  = dataset[split_idx["test"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_node_features = int(dataset[0].x.shape[1])
    num_edge_features = int(dataset[0].edge_attr.shape[1]) if dataset[0].edge_attr is not None else 3

    print(f"Train+Val pool: {len(tv_idx):,} molecules | Test: {len(test_idx):,}")
    print(f"Positive rate:  {tv_labels.mean()*100:.2f}%")
    print(f"K-Fold:         K={args.k_folds}, model={args.model}, epochs={args.epochs}")

    # Auto pos_weight
    if args.pos_weight > 0:
        pw = torch.tensor([args.pos_weight])
    else:
        n_pos = float(tv_labels.sum())
        n_neg = float(len(tv_labels)) - n_pos
        pw = torch.tensor([n_neg / max(n_pos, 1)])
        print(f"Auto pos_weight: {pw.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw.to(device))
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # ── K-Fold loop ───────────────────────────────────────────────────────────
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_summaries = []
    all_test_preds = []  # [K, n_test] for OOF ensemble eval

    for fold, (train_rel, val_rel) in enumerate(skf.split(tv_idx, tv_labels)):
        fold_train_idx = tv_idx[train_rel]
        fold_val_idx   = tv_idx[val_rel]

        fold_train = dataset[torch.from_numpy(fold_train_idx)]
        fold_val   = dataset[torch.from_numpy(fold_val_idx)]

        train_loader = DataLoader(fold_train, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(fold_val,   batch_size=args.batch_size, shuffle=False)

        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.k_folds}  |  train={len(fold_train):,}  val={len(fold_val):,}")
        print(f"{'='*60}")

        model = build_model(
            args.model, num_node_features, args.hidden_dim,
            args.num_layers, args.dropout, num_edge_features
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )

        best_val_auc = 0.0
        best_state   = None
        wait         = 0
        fold_history = []

        for epoch in range(1, args.epochs + 1):
            loss = run_epoch(model, train_loader, optimizer, criterion, device, args.model)
            val_m = evaluate(model, val_loader, device, args.model)
            scheduler.step()

            fold_history.append({
                "epoch": epoch, "loss": loss, "val_roc_auc": val_m["roc_auc"]
            })

            if val_m["roc_auc"] > best_val_auc:
                best_val_auc = val_m["roc_auc"]
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
                marker = " ✓"
            else:
                wait += 1
                marker = ""

            if epoch % 5 == 0 or marker:
                print(f"  Epoch {epoch:3d} | loss={loss:.4f} | val_auc={val_m['roc_auc']:.4f}{marker}")

            if wait >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Restore best weights
        model.load_state_dict(best_state)
        model.eval()

        # Evaluate on test set
        test_m = evaluate(model, test_loader, device, args.model)

        # Collect test predictions for ensemble
        fold_probs = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                y = batch.y.view(-1).float()
                mask = ~torch.isnan(y)
                kw = {"edge_attr": batch.edge_attr} if args.model == "gine" else {}
                logits = model(batch.x, batch.edge_index, batch.batch, **kw)
                probs = torch.sigmoid(logits)[mask].cpu().numpy()
                fold_probs.extend(probs.tolist())
        all_test_preds.append(np.array(fold_probs))

        print(f"  Fold {fold+1} best val AUC: {best_val_auc:.4f}  test AUC: {test_m['roc_auc']:.4f}")

        # Save checkpoint
        fold_dir = outdir / f"fold_{fold+1}"
        fold_dir.mkdir(exist_ok=True)
        torch.save({
            "model_state": best_state,
            "model_type":  args.model,
            "fold":        fold + 1,
            "best_val_roc_auc": best_val_auc,
            "test_roc_auc": test_m["roc_auc"],
            "args": vars(args),
            "num_node_features": num_node_features,
            "num_edge_features": num_edge_features,
        }, fold_dir / "best_model.pt")

        pd.DataFrame(fold_history).to_csv(fold_dir / "metrics.csv", index=False)

        fold_summaries.append({
            "fold":            fold + 1,
            "val_roc_auc":     best_val_auc,
            "test_roc_auc":    test_m["roc_auc"],
            "epochs_ran":      len(fold_history),
        })

    # ── Ensemble evaluation across folds ────────────────────────────────────
    from sklearn.metrics import roc_auc_score

    # True test labels
    test_true = []
    for batch in test_loader:
        y = batch.y.view(-1).float()
        mask = ~torch.isnan(y)
        test_true.extend(y[mask].numpy().tolist())
    test_true = np.array(test_true)

    # Average predictions across K folds
    ensemble_preds = np.mean(all_test_preds, axis=0)
    ensemble_auc   = float(roc_auc_score(test_true, ensemble_preds))

    print(f"\n{'='*60}")
    print(f"K-FOLD ENSEMBLE RESULTS ({args.k_folds} folds)")
    print(f"{'='*60}")
    for s in fold_summaries:
        print(f"  Fold {s['fold']}: val={s['val_roc_auc']:.4f}  test={s['test_roc_auc']:.4f}")
    print(f"  Individual mean test AUC: {np.mean([s['test_roc_auc'] for s in fold_summaries]):.4f}")
    print(f"  K-Fold ENSEMBLE test AUC: {ensemble_auc:.4f}")

    summary = {
        "model":              args.model,
        "k_folds":            args.k_folds,
        "ensemble_test_roc_auc": ensemble_auc,
        "individual_mean_test_roc_auc": float(np.mean([s["test_roc_auc"] for s in fold_summaries])),
        "best_val_roc_auc":   float(max(s["val_roc_auc"] for s in fold_summaries)),
        "fold_summaries":     fold_summaries,
        "args":               vars(args),
    }
    (outdir / "kfold_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nCheckpoints saved to: {outdir}/fold_*/best_model.pt")
    print(json.dumps({k: v for k, v in summary.items() if k != "fold_summaries"}, indent=2))


if __name__ == "__main__":
    main()
