from __future__ import annotations
"""Evaluate full ensemble: GIN 5-Fold + GCN + GINE + Morgan RF on ogbg-molhiv test set."""

import argparse, json
from pathlib import Path

import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from src.models.ensemble import load_full_ensemble, load_ensemble_from_dirs, _val_auc_weights
from src.models.kfold_ensemble import load_kfold_ensemble


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gnn-dirs", nargs="+", required=True)
    p.add_argument("--kfold-dir", default=None, help="Dir with fold_1/ fold_2/ ... subdirs")
    p.add_argument("--rf-path",   default=None)
    p.add_argument("--outfile",   default="outputs/ensemble_summary.json")
    args = p.parse_args()

    gnn_run_dirs = {Path(d).name.replace("run_", ""): d for d in args.gnn_dirs}
    print(f"GNN dirs:    {gnn_run_dirs}")
    print(f"K-Fold dir:  {args.kfold_dir}")
    print(f"RF path:     {args.rf_path}")

    dataset   = PygGraphPropPredDataset(name="ogbg-molhiv")
    split_idx = dataset.get_idx_split()

    import pandas as pd
    raw_path  = Path(dataset.root) / "mapping" / "mol.csv.gz"
    df_smi    = pd.read_csv(raw_path)
    scol      = next(c for c in df_smi.columns if "smiles" in c.lower())
    all_smi   = df_smi[scol].tolist()
    test_idx  = split_idx["test"].tolist()
    test_smi  = [all_smi[i] for i in test_idx]

    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=256, shuffle=False)
    device = torch.device("cpu")

    # ── True labels ──────────────────────────────────────────────────────────
    y_true = []
    for batch in test_loader:
        y = batch.y.view(-1).float()
        mask = ~torch.isnan(y)
        y_true.extend(y[mask].numpy().tolist())
    y_true = np.array(y_true)

    # ── Component 1: K-Fold GIN ensemble ─────────────────────────────────────
    kfold_ens = None
    kfold_val_auc = 0.0
    if args.kfold_dir and Path(args.kfold_dir).exists():
        kfold_ens = load_kfold_ensemble(args.kfold_dir, device)
        if kfold_ens:
            sf = Path(args.kfold_dir) / "kfold_summary.json"
            if sf.exists():
                kdata = json.loads(sf.read_text())
                kfold_val_auc = float(kdata.get("best_val_roc_auc", 0.75))
            kfold_probs = []
            offset = 0
            for batch in test_loader:
                y = batch.y.view(-1).float()
                mask = ~torch.isnan(y)
                if mask.sum() == 0:
                    offset += batch.num_graphs; continue
                batch = batch.to(device)
                p = kfold_ens.predict_batch(batch)
                kfold_probs.extend(p[mask.numpy().astype(bool)].tolist())
                offset += batch.num_graphs
            kfold_probs = np.array(kfold_probs)
            kfold_auc = float(roc_auc_score(y_true, kfold_probs))
            print(f"K-Fold GIN ensemble test AUC: {kfold_auc:.4f}")

    # ── Component 2: GCN + GINE (additional diversity) ───────────────────────
    extra_dirs = {k: v for k, v in gnn_run_dirs.items() if k in ("gcn", "gine")}
    extra_probs = {}
    extra_val_aucs = {}
    for mname, mdir in extra_dirs.items():
        sf = Path(mdir) / "run_summary.json"
        if sf.exists():
            extra_val_aucs[mname] = float(json.loads(sf.read_text()).get("best_val_roc_auc", 0.75))

    if extra_dirs:
        try:
            extra_ens = load_ensemble_from_dirs(extra_dirs, device)
            ep = []
            for batch in test_loader:
                y = batch.y.view(-1).float()
                mask = ~torch.isnan(y)
                if mask.sum() == 0: continue
                batch = batch.to(device)
                p = extra_ens(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
                ep.extend(p[mask].cpu().numpy().tolist())
            extra_probs_arr = np.array(ep)
            extra_auc = float(roc_auc_score(y_true, extra_probs_arr))
            extra_val_auc = np.mean(list(extra_val_aucs.values()))
            print(f"GCN+GINE ensemble test AUC: {extra_auc:.4f}")
        except Exception as e:
            print(f"Extra GNN ensemble failed: {e}")
            extra_probs_arr = None
            extra_val_auc = 0.0
    else:
        extra_probs_arr = None
        extra_val_auc = 0.0

    # ── Component 3: Morgan RF ────────────────────────────────────────────────
    rf_probs_arr = None
    rf_val_auc   = 0.0
    if args.rf_path and Path(args.rf_path).exists():
        from src.models.morgan_rf import MorganRFClassifier
        rf = MorganRFClassifier.load(args.rf_path)
        rf_raw = rf.predict_proba_smiles(test_smi)
        # Align with mask
        rf_masked = []
        offset = 0
        for batch in test_loader:
            y = batch.y.view(-1).float()
            mask = ~torch.isnan(y)
            n = batch.num_graphs
            rf_masked.extend([rf_raw[offset + i] for i in range(n) if mask[i]])
            offset += n
        rf_probs_arr = np.array(rf_masked)
        rf_auc = float(roc_auc_score(y_true, rf_probs_arr))
        rf_sf = Path(args.rf_path).parent / "run_summary.json"
        if rf_sf.exists():
            rf_val_auc = float(json.loads(rf_sf.read_text()).get("best_val_roc_auc", 0.0))
        print(f"Morgan RF test AUC: {rf_auc:.4f}")

    # ── Combine all components with val-AUC softmax weighting ────────────────
    components = {}
    comp_val_aucs = []
    if kfold_ens is not None:
        components["kfold_gin"] = kfold_probs
        comp_val_aucs.append(kfold_val_auc or 0.80)
    if extra_probs_arr is not None:
        components["gcn_gine"] = extra_probs_arr
        comp_val_aucs.append(extra_val_auc or 0.75)
    if rf_probs_arr is not None:
        components["morgan_rf"] = rf_probs_arr
        comp_val_aucs.append(rf_val_auc or 0.80)

    if not components:
        print("No ensemble components available — using single GIN")
        gin_dir = gnn_run_dirs.get("gin", list(gnn_run_dirs.values())[0])
        ens = load_ensemble_from_dirs({"gin": gin_dir}, device)
        ep2 = []
        for batch in test_loader:
            y = batch.y.view(-1).float()
            mask = ~torch.isnan(y)
            if mask.sum() == 0: continue
            batch = batch.to(device)
            p = ens(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
            ep2.extend(p[mask].cpu().numpy().tolist())
        final_probs = np.array(ep2)
    else:
        weights = _val_auc_weights(comp_val_aucs)
        print("Component weights: " + ", ".join(
            f"{n}={w:.3f}(val={v:.3f})"
            for n, w, v in zip(components.keys(), weights, comp_val_aucs)))
        final_probs = sum(w * p for w, p in zip(weights, components.values()))

    final_auc = float(roc_auc_score(y_true, final_probs))
    print(f"\n{'='*50}")
    print(f"FINAL ENSEMBLE Test ROC-AUC: {final_auc:.4f}")
    print(f"{'='*50}")

    # Individual summaries
    members = {}
    for k, d in gnn_run_dirs.items():
        sf = Path(d) / "run_summary.json"
        if sf.exists():
            members[k] = json.loads(sf.read_text())
    if args.kfold_dir and (Path(args.kfold_dir) / "kfold_summary.json").exists():
        members["kfold_gin"] = json.loads((Path(args.kfold_dir) / "kfold_summary.json").read_text())
    if args.rf_path:
        rf_sf = Path(args.rf_path).parent / "run_summary.json"
        if rf_sf.exists():
            members["morgan_rf"] = json.loads(rf_sf.read_text())

    result = {
        "ensemble_test_roc_auc": final_auc,
        "members": list(members.keys()),
        "strategy": "5-Fold GIN + GCN+GINE ensemble + Morgan RF — val-AUC softmax bloc weights",
        "weighting": "val_auc_softmax",
        "temperature": 0.05,
        "member_summaries": members,
    }
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    Path(args.outfile).write_text(json.dumps(result, indent=2))
    print(f"Saved → {args.outfile}")


if __name__ == "__main__":
    main()
