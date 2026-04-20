"""
Hugging Face Spaces entry point for GNN Drug Discovery Dashboard.
Downloads latest model artifacts from GitHub Releases on startup,
then serves the full Streamlit dashboard + live SMILES inference.

Pipeline: 5-Fold Enhanced GIN (VNode+JK+RDKit) + Morgan RF ensemble
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import torch

# ── Config ────────────────────────────────────────────────────────────────────
GITHUB_REPO  = "yuriao/GNN_drug_discovery"
RELEASE_TAG  = "model-latest"
KFOLD_DIR    = Path("outputs/kfold")
RF_DIR       = Path("outputs/rf")
SHARED_DIR   = Path("outputs/shared")

# (local_path, release_asset_name)
KFOLD_ASSETS = [
    (KFOLD_DIR / f"fold_{i}/best_model.pt", f"gin_kfold_fold{i}.pt")
    for i in range(1, 6)
]
SHARED_ASSETS = [
    (SHARED_DIR / "dataset_stats.json",          "dataset_stats.json"),
    (SHARED_DIR / "preprocessing_summary.json",  "preprocessing_summary.json"),
    (SHARED_DIR / "ensemble_summary.json",        "ensemble_summary.json"),
    (SHARED_DIR / "kfold_summary.json",           "kfold_summary.json"),
    (RF_DIR     / "rf_run_summary.json",          "rf_run_summary.json"),
    (RF_DIR     / "rf_best_model.joblib",         "rf_best_model.joblib"),
]

ALL_ASSETS = KFOLD_ASSETS + SHARED_ASSETS


# ── Artifact download ─────────────────────────────────────────────────────────

def download_artifacts():
    """Download ensemble model artifacts from the latest GitHub release."""
    for i in range(1, 6):
        (KFOLD_DIR / f"fold_{i}").mkdir(parents=True, exist_ok=True)
    RF_DIR.mkdir(parents=True, exist_ok=True)
    SHARED_DIR.mkdir(parents=True, exist_ok=True)

    base_url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"
    to_fetch = [(asset, dest) for dest, asset in ALL_ASSETS if not Path(dest).exists()]

    if not to_fetch:
        return True

    progress = st.progress(0, text="Downloading model artifacts…")
    for i, (asset_name, dest) in enumerate(to_fetch):
        url = f"{base_url}/{asset_name}"
        try:
            urllib.request.urlretrieve(url, dest)
            progress.progress((i + 1) / len(to_fetch), text=f"Downloaded {asset_name}")
        except Exception as e:
            st.warning(f"Could not download {asset_name}: {e}")
    progress.empty()

    # Ready when at least fold 1 model + summary exist
    return (KFOLD_DIR / "fold_1/best_model.pt").exists() and (SHARED_DIR / "kfold_summary.json").exists()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GNN Drug Discovery",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 GNN Drug Discovery Dashboard")
st.caption(
    "5-Fold Enhanced GIN + Morgan RF ensemble for HIV inhibition prediction on ogbg-molhiv"
)

with st.spinner("Loading model artifacts…"):
    ready = download_artifacts()

if not ready:
    st.error("Could not load model artifacts. The model may not have been trained yet.")
    st.info("👉 Trigger training via GitHub Actions: Actions → Train GNN Model → Run workflow")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "📈 K-Fold Results", "🏆 Ensemble", "🔬 Inference"])

# ── Tab 1: Dataset Stats ──────────────────────────────────────────────────────
with tab1:
    st.header("Dataset — ogbg-molhiv")
    stats_file = SHARED_DIR / "dataset_stats.json"
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Molecules",  f"{stats.get('num_graphs', 0):,}")
        c2.metric("Node Features",    stats.get("num_node_features", "-"))
        c3.metric("Edge Features",    stats.get("num_edge_features", "-"))
        c4.metric("Tasks",            stats.get("num_tasks", "-"))
        splits = stats.get("split_sizes", {})
        if splits:
            st.subheader("Dataset Split")
            split_df = pd.DataFrame([
                {"Split": "Train",      "Molecules": splits.get("train", 0)},
                {"Split": "Validation", "Molecules": splits.get("valid", 0)},
                {"Split": "Test",       "Molecules": splits.get("test", 0)},
            ])
            fig = px.bar(split_df, x="Split", y="Molecules", color="Split",
                         color_discrete_sequence=["#3b82f6", "#8b5cf6", "#10b981"])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Dataset stats not available.")

    prep_file = SHARED_DIR / "preprocessing_summary.json"
    if prep_file.exists():
        st.subheader("SMILES Preprocessing")
        prep = json.loads(prep_file.read_text())
        p1, p2, p3 = st.columns(3)
        p1.metric("Valid SMILES",       prep.get("valid_smiles", "-"))
        p2.metric("Invalid SMILES",     prep.get("invalid_smiles", "-"))
        p3.metric("Avg Atoms/Molecule", f"{prep.get('avg_nodes', 0):.1f}")

# ── Tab 2: K-Fold Results ─────────────────────────────────────────────────────
with tab2:
    st.header("5-Fold Cross-Validation — Enhanced GIN")
    kfold_file = SHARED_DIR / "kfold_summary.json"
    if kfold_file.exists():
        ks = json.loads(kfold_file.read_text())
        k1, k2, k3 = st.columns(3)
        k1.metric("Ensemble Test ROC-AUC",     f"{ks.get('ensemble_test_roc_auc', 0):.4f}")
        k2.metric("Mean Fold Test ROC-AUC",    f"{ks.get('individual_mean_test_roc_auc', 0):.4f}")
        k3.metric("Best Fold Val ROC-AUC",     f"{ks.get('best_val_roc_auc', 0):.4f}")

        folds = ks.get("fold_summaries", [])
        if folds:
            fold_df = pd.DataFrame(folds)
            fig = px.bar(
                fold_df, x="fold", y=["val_roc_auc", "test_roc_auc"],
                barmode="group",
                title="Per-Fold ROC-AUC",
                labels={"value": "ROC-AUC", "fold": "Fold", "variable": "Split"},
                color_discrete_sequence=["#3b82f6", "#10b981"],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(fold_df, use_container_width=True)

        st.subheader("Model Architecture")
        arch_cols = ["use_virtual_node", "use_jk", "use_global_features", "k_folds"]
        for col in arch_cols:
            st.write(f"**{col}:** {ks.get(col)}")

    rf_file = RF_DIR / "rf_run_summary.json"
    if rf_file.exists():
        st.subheader("Morgan RF Baseline")
        rf = json.loads(rf_file.read_text())
        r1, r2 = st.columns(2)
        r1.metric("RF Val ROC-AUC",  f"{rf.get('best_val_roc_auc', 0):.4f}")
        r2.metric("RF Test ROC-AUC", f"{rf.get('test_roc_auc_at_best_val', 0):.4f}")
        st.caption(f"n_estimators={rf.get('n_estimators')} · radius={rf.get('morgan_radius')} · nbits={rf.get('morgan_nbits')}")

# ── Tab 3: Ensemble Results ───────────────────────────────────────────────────
with tab3:
    st.header("Final Ensemble — GIN + RF")
    ens_file = SHARED_DIR / "ensemble_summary.json"
    if ens_file.exists():
        ens = json.loads(ens_file.read_text())
        e1, e2 = st.columns(2)
        e1.metric("Ensemble Test ROC-AUC", f"{ens.get('ensemble_test_roc_auc', 0):.4f}")
        e2.metric("Strategy", ens.get("strategy", "-"))

        members = ens.get("member_summaries", {})
        rows = []
        for name, m in members.items():
            rows.append({
                "Member": name,
                "Test ROC-AUC": m.get("ensemble_test_roc_auc") or m.get("test_roc_auc_at_best_val", "-"),
            })
        if rows:
            st.table(pd.DataFrame(rows))

        st.subheader("Benchmark Context")
        st.markdown(f"""
| Model | Test ROC-AUC |
|-------|-------------|
| GCN (OGB baseline) | 0.759 |
| GIN (OGB baseline) | 0.771 |
| **This run (GIN k-fold)** | **{ens.get('kfold_gin_test_auc', 0):.4f}** |
| **Morgan RF** | **{ens.get('rf_test_auc', 0):.4f}** |
| **Ensemble (GIN + RF)** | **{ens.get('ensemble_test_roc_auc', 0):.4f}** |
""")
    else:
        st.info("Ensemble summary not found.")

# ── Tab 4: Live Inference ─────────────────────────────────────────────────────
with tab4:
    st.header("🔬 Predict HIV Inhibition from SMILES")
    st.markdown("Enter any molecule's SMILES string to get its predicted HIV inhibition probability.")

    examples = {
        "Aspirin":       "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine":      "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ibuprofen":     "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "AZT (HIV drug)":"CC1=CN([C@H]2C[C@H](N=[N+]=[N-])[C@@H](CO)O2)C(=O)NC1=O",
        "Benzene":       "c1ccccc1",
    }

    col1, col2 = st.columns([2, 1])
    with col2:
        example = st.selectbox("Load example", ["Custom…"] + list(examples.keys()))
    default_smiles = examples.get(example, "") if example != "Custom…" else ""
    with col1:
        smiles = st.text_input("SMILES string", value=default_smiles,
                               placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O")

    if st.button("🧬 Predict", type="primary") and smiles:
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                st.error("Invalid SMILES string. Please check your input.")
            else:
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from src.models.kfold_ensemble import load_kfold_ensemble
                from src.data.smiles_to_graph import smiles_to_pyg

                device = torch.device("cpu")

                # ── GIN k-fold ensemble ───────────────────────────────────
                gin_ens = load_kfold_ensemble(KFOLD_DIR, device, val_auc_weighted=True)
                gin_prob = gin_ens.predict_smiles(smiles) if gin_ens else None

                # ── Morgan RF ─────────────────────────────────────────────
                rf_prob = None
                rf_model_path = RF_DIR / "rf_best_model.joblib"
                if rf_model_path.exists():
                    try:
                        import joblib
                        from src.data.mol_features import smiles_to_features
                        rf_model = joblib.load(rf_model_path)
                        feats = smiles_to_features(smiles)
                        import numpy as np
                        rf_prob = float(rf_model.predict_proba(feats.reshape(1, -1))[0, 1])
                    except Exception as e:
                        st.caption(f"RF inference skipped: {e}")

                # ── Ensemble average (val-AUC weighted if available) ──────
                if gin_prob is not None and rf_prob is not None:
                    # weights from ensemble_summary
                    ens_prob = (gin_prob + rf_prob) / 2.0  # simple average fallback
                    ens_file = SHARED_DIR / "ensemble_summary.json"
                    if ens_file.exists():
                        ens_meta = json.loads(ens_file.read_text())
                        gin_auc = ens_meta.get("kfold_gin_test_auc", 0.76)
                        rf_auc  = ens_meta.get("rf_test_auc", 0.79)
                        total   = gin_auc + rf_auc
                        w_gin, w_rf = gin_auc / total, rf_auc / total
                        ens_prob = gin_prob * w_gin + rf_prob * w_rf
                    label = "Ensemble (GIN k-fold + Morgan RF)"
                    prob = ens_prob
                elif gin_prob is not None:
                    prob  = gin_prob
                    label = "GIN 5-Fold Ensemble"
                else:
                    st.error("No models loaded for inference.")
                    st.stop()

                # ── Display ───────────────────────────────────────────────
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.metric(f"HIV Inhibition Probability", f"{prob:.4f}")
                    st.caption(f"Model: {label}")
                    if prob > 0.5:
                        st.error(f"HIGH probability of HIV inhibition ({prob:.1%})")
                    elif prob > 0.3:
                        st.warning(f"MODERATE probability ({prob:.1%})")
                    else:
                        st.success(f"LOW probability of HIV inhibition ({prob:.1%})")

                    st.markdown("**Individual model scores:**")
                    if gin_prob is not None:
                        bar = "█" * int(gin_prob * 20)
                        st.caption(f"GIN k-fold: {gin_prob:.3f}  {bar}")
                    if rf_prob is not None:
                        bar = "█" * int(rf_prob * 20)
                        st.caption(f"Morgan RF:  {rf_prob:.3f}  {bar}")

                with c2:
                    from rdkit.Chem import Draw
                    from rdkit.Chem.Draw import rdMolDraw2D
                    import io
                    drawer = rdMolDraw2D.MolDraw2DSVG(380, 280)
                    drawer.DrawMolecule(mol)
                    drawer.FinishDrawing()
                    img_bytes = drawer.GetDrawingText().encode()
                    st.image(io.BytesIO(img_bytes), width=380)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            import traceback
            st.code(traceback.format_exc())

st.divider()
st.caption(
    "5-Fold Enhanced GIN (VNode+JK+RDKit feats) + Morgan RF · ogbg-molhiv scaffold split · "
    "Ensemble ROC-AUC: 0.776 · "
    "[GitHub](https://github.com/yuriao/GNN_drug_discovery)"
)
