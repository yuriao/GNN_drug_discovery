from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.data.load_ogb import build_dataloaders, save_dataset_stats
from src.models.factory import build_model
from src.utils.metrics import safe_pr_auc, safe_roc_auc
from src.utils.seed import set_seed


def run_epoch(model, loader, optimizer, criterion, device, model_name: str = "gin"):
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(-1).float()
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue

        optimizer.zero_grad()

        # GINE requires edge_attr; GIN/GCN ignore it
        if model_name == "gine":
            logits = model(batch.x, batch.edge_index, batch.batch,
                           edge_attr=batch.edge_attr)
        else:
            logits = model(batch.x, batch.edge_index, batch.batch)

        loss = criterion(logits[mask], y[mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * int(mask.sum().item())
        total_count += int(mask.sum().item())

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, device, model_name: str = "gin"):
    model.eval()
    y_true, y_score = [], []

    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(-1).float()
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue

        if model_name == "gine":
            logits = model(batch.x, batch.edge_index, batch.batch,
                           edge_attr=batch.edge_attr)
        else:
            logits = model(batch.x, batch.edge_index, batch.batch)

        probs = torch.sigmoid(logits)
        y_true.append(y[mask].detach().cpu().numpy())
        y_score.append(probs[mask].detach().cpu().numpy())

    if not y_true:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}

    yt = np.concatenate(y_true)
    ys = np.concatenate(y_score)
    return {
        "roc_auc": safe_roc_auc(yt, ys),
        "pr_auc":  safe_pr_auc(yt, ys),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN on ogbg-molhiv")
    parser.add_argument("--model", choices=["gcn", "gin", "gine"], default="gin")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--pos-weight", type=float, default=0.0,
                        help="BCE pos_weight (0 = auto-compute from class balance)")
    parser.add_argument("--augment-positives", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="outputs/run")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset, split_idx, train_loader, valid_loader, test_loader = build_dataloaders(
        batch_size=args.batch_size,
        augment_positives=args.augment_positives,
    )
    save_dataset_stats(dataset, split_idx, args.outdir)

    num_edge_feat = dataset[0].edge_attr.shape[1] if dataset[0].edge_attr is not None else 3
    model = build_model(
        args.model,
        dataset[0].x.shape[1],
        args.hidden_dim,
        args.num_layers,
        args.dropout,
        num_edge_features=num_edge_feat,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.pos_weight > 0:
        pw = torch.tensor([args.pos_weight])
    else:
        all_labels = torch.cat([data.y.view(-1) for data in train_loader.dataset])
        mask = ~torch.isnan(all_labels)
        n_pos = float(all_labels[mask].sum())
        n_neg = float(mask.sum()) - n_pos
        pw = torch.tensor([n_neg / max(n_pos, 1)])
        print(f"Auto pos_weight: {pw.item():.1f} ({int(n_pos)} pos / {int(n_neg)} neg)")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pw.to(device))

    best_val = -1.0
    best_epoch = -1
    best_test_at_val = float("nan")
    wait = 0
    history = []

    for epoch in tqdm(range(1, args.epochs + 1), desc=f"Training {args.model.upper()}"):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, args.model)
        val_metrics  = evaluate(model, valid_loader, device, args.model)
        test_metrics = evaluate(model, test_loader,  device, args.model)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_roc_auc":  val_metrics["roc_auc"],
            "val_pr_auc":   val_metrics["pr_auc"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_pr_auc":  test_metrics["pr_auc"],
        }
        history.append(row)

        if val_metrics["roc_auc"] > best_val:
            best_val = val_metrics["roc_auc"]
            best_epoch = epoch
            best_test_at_val = test_metrics["roc_auc"]
            wait = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_type": args.model,
                    "args": vars(args),
                    "num_node_features": int(dataset[0].x.shape[1]),
                    "num_edge_features": int(num_edge_feat),
                },
                outdir / "best_model.pt",
            )
        else:
            wait += 1

        if wait >= args.patience:
            break

    pd.DataFrame(history).to_csv(outdir / "metrics.csv", index=False)

    summary = {
        "model": args.model,
        "best_epoch": best_epoch,
        "best_val_roc_auc": best_val,
        "test_roc_auc_at_best_val": best_test_at_val,
        "device": str(device),
        "epochs_ran": int(len(history)),
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
