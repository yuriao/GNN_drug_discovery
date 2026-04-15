from __future__ import annotations
"""
K-Fold CV training for Enhanced GIN on ogbg-molhiv.

Improvements over vanilla GIN:
  1. Virtual Node        — global super-node for long-range atom communication
  2. Jumping Knowledge   — concat all layer outputs before pooling
  3. Global RDKit feats  — MW, LogP, TPSA, HBA/HBD, QED etc. injected into MLP head

Each fold:
  - Trains on K-1 parts of (train+val combined)
  - Validates on the remaining 1 part
  - Saves best checkpoint (by val ROC-AUC)
  - Uses cosine annealing LR schedule

At inference: average all K folds' sigmoid probabilities.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ogb.graphproppred import PygGraphPropPredDataset
from src.models.factory import build_model
from src.data.mol_features import add_global_features_to_dataset, NUM_GLOBAL_FEATURES
from src.utils.metrics import safe_roc_auc, safe_pr_auc
from src.utils.seed import set_seed


# ── helpers ───────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, device, use_global_feats=True):
    model.train()
    total_loss, total_n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        y    = batch.y.view(-1).float()
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        optimizer.zero_grad()
        gf = batch.global_feat if (use_global_feats and hasattr(batch, "global_feat")) else None
        logits = model(batch.x, batch.edge_index, batch.batch, global_feat=gf)
        loss   = criterion(logits[mask], y[mask])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(mask.sum())
        total_n    += int(mask.sum())
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device, use_global_feats=True):
    model.eval()
    y_true, y_score = [], []
    for batch in loader:
        batch = batch.to(device)
        y    = batch.y.view(-1).float()
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        gf = batch.global_feat if (use_global_feats and hasattr(batch, "global_feat")) else None
        logits = model(batch.x, batch.edge_index, batch.batch, global_feat=gf)
        probs  = torch.sigmoid(logits)
        y_true.append(y[mask].cpu().numpy())
        y_score.append(probs[mask].cpu().numpy())
    if not y_true:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    yt = np.concatenate(y_true)
    ys = np.concatenate(y_score)
    return {"roc_auc": safe_roc_auc(yt, ys), "pr_auc": safe_pr_auc(yt, ys)}


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           choices=["gin", "gin_enhanced"], default="gin_enhanced")
    p.add_argument("--k-folds",         type=int,   default=5)
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch-size",      type=int,   default=64)
    p.add_argument("--hidden-dim",      type=int,   default=128)
    p.add_argument("--num-layers",      type=int,   default=4)
    p.add_argument("--dropout",         type=float, default=0.2)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--weight-decay",    type=float, default=1e-5)
    p.add_argument("--patience",        type=int,   default=10)
    p.add_argument("--pos-weight",      type=float, default=0.0)
    p.add_argument("--no-virtual-node", action="store_true")
    p.add_argument("--no-jk",          action="store_true")
    p.add_argument("--no-global-feats", action="store_true")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--outdir",          type=str,   default="outputs/run_gin_kfold")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    use_vnode  = not args.no_virtual_node
    use_jk     = not args.no_jk
    use_gfeat  = not args.no_global_feats and (args.model == "gin_enhanced")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading ogbg-molhiv...")
    dataset   = PygGraphPropPredDataset(name="ogbg-molhiv")
    split_idx = dataset.get_idx_split()

    # Load SMILES for global feature computation
    smiles_list = None
    if use_gfeat:
        raw_path = Path(dataset.root) / "mapping" / "mol.csv.gz"
        import pandas as _pd
        df = _pd.read_csv(raw_path)
        scol = next(c for c in df.columns if "smiles" in c.lower())
        smiles_list = df[scol].tolist()
        print(f"Computing RDKit global features for {len(smiles_list):,} molecules...")
        from src.data.mol_features import smiles_to_global_features
        import numpy as _np
        all_feats = []
        from tqdm import tqdm as _tqdm
        for smi in _tqdm(smiles_list, desc="RDKit features"):
            f = smiles_to_global_features(smi)
            all_feats.append(f if f is not None else _np.zeros(NUM_GLOBAL_FEATURES, dtype=_np.float32))
        all_feats = _np.stack(all_feats)
        all_feats = _np.nan_to_num(all_feats, nan=0.0, posinf=5.0, neginf=-5.0)
        n_bad = int(_np.isnan(all_feats).sum())
        print(f"Global features computed. Shape: {all_feats.shape}  NaN remaining: {n_bad}")

    all_labels = np.array([
        dataset[int(i)].y.view(-1)[0].item() for i in range(len(dataset))
    ])
    train_idx = split_idx["train"].numpy()
    valid_idx = split_idx["valid"].numpy()
    test_idx  = split_idx["test"].numpy()
    tv_idx    = np.concatenate([train_idx, valid_idx])
    tv_labels = all_labels[tv_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_node_features = int(dataset[0].x.shape[1])
    num_edge_features = int(dataset[0].edge_attr.shape[1]) if dataset[0].edge_attr is not None else 3

    print(f"Train+Val: {len(tv_idx):,} | Test: {len(test_idx):,} | Device: {device}")
    print(f"Model: {args.model} | VNode: {use_vnode} | JK: {use_jk} | GlobalFeat: {use_gfeat}")

    # pos_weight
    if args.pos_weight > 0:
        pw = torch.tensor([args.pos_weight])
    else:
        n_pos = float(tv_labels.sum())
        n_neg = float(len(tv_labels)) - n_pos
        pw = torch.tensor([n_neg / max(n_pos, 1)])
        print(f"Auto pos_weight: {pw.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw.to(device))

    # Build test loader with global features
    test_data = list(dataset[split_idx["test"]])
    if use_gfeat:
        for i, idx in enumerate(test_idx):
            test_data[i].global_feat = torch.from_numpy(all_feats[idx]).unsqueeze(0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # K-Fold
    from sklearn.metrics import roc_auc_score
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_summaries = []
    all_test_preds = []

    for fold, (train_rel, val_rel) in enumerate(skf.split(tv_idx, tv_labels)):
        fold_train_idx = tv_idx[train_rel]
        fold_val_idx   = tv_idx[val_rel]

        # Build fold datasets with global features
        fold_train = [dataset[int(i)] for i in fold_train_idx]
        fold_val   = [dataset[int(i)] for i in fold_val_idx]
        if use_gfeat:
            for j, idx in enumerate(fold_train_idx):
                fold_train[j].global_feat = torch.from_numpy(all_feats[idx]).unsqueeze(0)
            for j, idx in enumerate(fold_val_idx):
                fold_val[j].global_feat = torch.from_numpy(all_feats[idx]).unsqueeze(0)

        train_loader = DataLoader(fold_train, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(fold_val,   batch_size=args.batch_size, shuffle=False)

        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{args.k_folds}  train={len(fold_train):,}  val={len(fold_val):,}")
        print(f"{'='*60}")

        model = build_model(
            args.model, num_node_features, args.hidden_dim,
            args.num_layers, args.dropout,
            num_edge_features=num_edge_features,
            use_virtual_node=use_vnode,
            use_jk=use_jk,
            use_global_features=use_gfeat,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if fold == 0:
            print(f"Model parameters: {n_params:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_auc, best_state, wait = 0.0, None, 0
        fold_history = []

        for epoch in range(1, args.epochs + 1):
            loss  = run_epoch(model, train_loader, optimizer, criterion, device, use_gfeat)
            val_m = evaluate(model, val_loader, device, use_gfeat)
            scheduler.step()

            fold_history.append({"epoch": epoch, "loss": loss, "val_roc_auc": val_m["roc_auc"]})

            if val_m["roc_auc"] > best_val_auc:
                best_val_auc = val_m["roc_auc"]
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
                marker = " ✓"
            else:
                wait += 1
                marker = ""

            if epoch % 5 == 0 or marker:
                print(f"  Ep {epoch:3d} | loss={loss:.4f} | val_auc={val_m['roc_auc']:.4f}{marker}")

            if wait >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        model.load_state_dict(best_state)
        model.eval()
        test_m = evaluate(model, test_loader, device, use_gfeat)

        # Collect fold test predictions
        fold_probs = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                y    = batch.y.view(-1).float()
                mask = ~torch.isnan(y)
                gf   = batch.global_feat if (use_gfeat and hasattr(batch, "global_feat")) else None
                logits = model(batch.x, batch.edge_index, batch.batch, global_feat=gf)
                probs  = torch.sigmoid(logits)[mask].cpu().numpy()
                fold_probs.extend(probs.tolist())
        all_test_preds.append(np.array(fold_probs))

        print(f"  Fold {fold+1} → val={best_val_auc:.4f}  test={test_m['roc_auc']:.4f}")

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
            "use_virtual_node":  use_vnode,
            "use_jk":            use_jk,
            "use_global_features": use_gfeat,
        }, fold_dir / "best_model.pt")
        pd.DataFrame(fold_history).to_csv(fold_dir / "metrics.csv", index=False)
        fold_summaries.append({"fold": fold+1, "val_roc_auc": best_val_auc,
                                "test_roc_auc": test_m["roc_auc"], "epochs_ran": len(fold_history)})

    # Ensemble test evaluation
    test_true = []
    for batch in test_loader:
        y = batch.y.view(-1).float()
        mask = ~torch.isnan(y)
        test_true.extend(y[mask].numpy().tolist())
    test_true = np.array(test_true)

    ensemble_preds = np.mean(all_test_preds, axis=0)
    ensemble_auc   = float(roc_auc_score(test_true, ensemble_preds))

    print(f"\n{'='*60}")
    print(f"{args.k_folds}-FOLD ENSEMBLE — model: {args.model}")
    for s in fold_summaries:
        print(f"  Fold {s['fold']}: val={s['val_roc_auc']:.4f}  test={s['test_roc_auc']:.4f}")
    print(f"  Individual mean: {np.mean([s['test_roc_auc'] for s in fold_summaries]):.4f}")
    print(f"  K-Fold ENSEMBLE: {ensemble_auc:.4f}")
    print(f"{'='*60}")

    summary = {
        "model": args.model,
        "k_folds": args.k_folds,
        "use_virtual_node":    use_vnode,
        "use_jk":              use_jk,
        "use_global_features": use_gfeat,
        "ensemble_test_roc_auc": ensemble_auc,
        "individual_mean_test_roc_auc": float(np.mean([s["test_roc_auc"] for s in fold_summaries])),
        "best_val_roc_auc": float(max(s["val_roc_auc"] for s in fold_summaries)),
        "fold_summaries": fold_summaries,
        "args": vars(args),
    }
    (outdir / "kfold_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "fold_summaries"}, indent=2))


if __name__ == "__main__":
    main()
