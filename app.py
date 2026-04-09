"""
Hugging Face Spaces entry point for GNN Drug Discovery Dashboard.
Downloads latest model artifacts from GitHub Releases on startup,
then serves the full Streamlit dashboard + live SMILES inference.
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
GITHUB_REPO = "yuriao/GNN_drug_discovery"
RELEASE_TAG = "model-latest"
ARTIFACTS_DIR = Path("outputs/run_gin")
ARTIFACTS = [
    "best_model.pt",
    "metrics.csv",
    "run_summary.json",
    "dataset_stats.json",
    "preprocessing_summary.json",
]

# ── Artifact download ─────────────────────────────────────────────────────────

def download_artifacts():
    """Download model artifacts from the latest GitHub release if not present."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    base_url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}"
    missing = [f for f in ARTIFACTS if not (ARTIFACTS_DIR / f).exists()]
    if not missing:
        return True
    progress = st.progress(0, text="Downloading model artifacts...")
    for i, fname in enumerate(missing):
        url = f"{base_url}/{fname}"
        dest = ARTIFACTS_DIR / fname
        try:
            urllib.request.urlretrieve(url, dest)
            progress.progress((i + 1) / len(missing), text=f"Downloaded {fname}")
        except Exception as e:
            st.warning(f"Could not download {fname}: {e}")
    progress.empty()
    return all((ARTIFACTS_DIR / f).exists() for f in ["run_summary.json", "metrics.csv"])


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GNN Drug Discovery",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 GNN Drug Discovery Dashboard")
st.caption("Graph Neural Network pipeline for HIV inhibition prediction on ogbg-molhiv")

# Download artifacts on first load
with st.spinner("Loading model artifacts..."):
    ready = download_artifacts()

if not ready:
    st.error("Could not load model artifacts. The model may not have been trained yet.")
    st.info("👉 Trigger training via GitHub Actions: Actions → Train GNN Model → Run workflow")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "📈 Training", "🏆 Results", "🔬 Inference"])

# ── Tab 1: Dataset Stats ──────────────────────────────────────────────────────
with tab1:
    st.header("Dataset — ogbg-molhiv")
    stats_file = ARTIFACTS_DIR / "dataset_stats.json"
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Molecules", f"{stats.get('num_graphs', 0):,}")
        c2.metric("Node Features", stats.get("num_node_features", "-"))
        c3.metric("Edge Features", stats.get("num_edge_features", "-"))
        c4.metric("Tasks", stats.get("num_tasks", "-"))
        splits = stats.get("split_sizes", {})
        if splits:
            st.subheader("Dataset Split")
            split_df = pd.DataFrame([
                {"Split": "Train", "Molecules": splits.get("train", 0)},
                {"Split": "Validation", "Molecules": splits.get("valid", 0)},
                {"Split": "Test", "Molecules": splits.get("test", 0)},
            ])
            fig = px.bar(split_df, x="Split", y="Molecules", color="Split",
                         color_discrete_sequence=["#3b82f6", "#8b5cf6", "#10b981"])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Dataset stats not available.")

    prep_file = ARTIFACTS_DIR / "preprocessing_summary.json"
    if prep_file.exists():
        st.subheader("SMILES Preprocessing")
        prep = json.loads(prep_file.read_text())
        p1, p2, p3 = st.columns(3)
        p1.metric("Valid SMILES", prep.get("valid_smiles", "-"))
        p2.metric("Invalid SMILES", prep.get("invalid_smiles", "-"))
        p3.metric("Avg Atoms/Molecule", f"{prep.get('avg_nodes', 0):.1f}")

# ── Tab 2: Training Curves ────────────────────────────────────────────────────
with tab2:
    st.header("Training Curves")
    metrics_file = ARTIFACTS_DIR / "metrics.csv"
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        if not df.empty:
            fig_loss = px.line(df, x="epoch", y="train_loss", title="Training Loss",
                               labels={"train_loss": "Loss", "epoch": "Epoch"},
                               color_discrete_sequence=["#ef4444"])
            st.plotly_chart(fig_loss, use_container_width=True)

            fig_roc = px.line(df, x="epoch",
                              y=[c for c in ["val_roc_auc", "test_roc_auc"] if c in df.columns],
                              title="ROC-AUC by Epoch",
                              labels={"value": "ROC-AUC", "epoch": "Epoch", "variable": "Split"},
                              color_discrete_sequence=["#3b82f6", "#10b981"])
            st.plotly_chart(fig_roc, use_container_width=True)

            fig_pr = px.line(df, x="epoch",
                             y=[c for c in ["val_pr_auc", "test_pr_auc"] if c in df.columns],
                             title="PR-AUC by Epoch",
                             labels={"value": "PR-AUC", "epoch": "Epoch", "variable": "Split"},
                             color_discrete_sequence=["#8b5cf6", "#f59e0b"])
            st.plotly_chart(fig_pr, use_container_width=True)
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No training metrics found.")

# ── Tab 3: Final Results ──────────────────────────────────────────────────────
with tab3:
    st.header("Final Results")
    summary_file = ARTIFACTS_DIR / "run_summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        m1, m2, m3 = st.columns(3)
        m1.metric("Model", summary.get("model", "-").upper())
        m2.metric("Best Val ROC-AUC", f"{summary.get('best_val_roc_auc', 0):.4f}")
        m3.metric("Test ROC-AUC @ Best Val", f"{summary.get('test_roc_auc_at_best_val', 0):.4f}")

        col1, col2 = st.columns(2)
        col1.metric("Best Epoch", summary.get("best_epoch", "-"))
        col2.metric("Total Epochs Run", summary.get("epochs_ran", "-"))

        st.subheader("Full Summary")
        st.json(summary)

        st.subheader("Benchmark Context")
        st.markdown("""
        | Model | Reported Test ROC-AUC |
        |-------|----------------------|
        | GCN (OGB baseline) | ~0.759 |
        | GIN (OGB baseline) | ~0.771 |
        | **This run** | **{:.3f}** |
        """.format(summary.get("test_roc_auc_at_best_val", 0)))
    else:
        st.info("No run summary found.")

# ── Tab 4: Live Inference ─────────────────────────────────────────────────────
with tab4:
    st.header("🔬 Predict HIV Inhibition from SMILES")
    st.markdown("Enter any molecule's SMILES string to get its predicted HIV inhibition probability.")

    examples = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "AZT (HIV drug)": "CC1=CN([C@H]2C[C@H](N=[N+]=[N-])[C@@H](CO)O2)C(=O)NC1=O",
        "Benzene": "c1ccccc1",
    }

    col1, col2 = st.columns([2, 1])
    with col2:
        example = st.selectbox("Load example", ["Custom..."] + list(examples.keys()))

    default_smiles = examples.get(example, "") if example != "Custom..." else ""
    with col1:
        smiles = st.text_input("SMILES string", value=default_smiles,
                               placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O")

    if st.button("🧬 Predict", type="primary") and smiles:
        model_path = ARTIFACTS_DIR / "best_model.pt"
        if not model_path.exists():
            st.error("Model checkpoint not found.")
        else:
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    st.error("Invalid SMILES string. Please check your input.")
                else:
                    import sys
                    sys.path.insert(0, str(Path(__file__).parent))
                    from src.data.smiles_to_graph import smiles_to_pyg
                    from src.models.factory import build_model
                    from torch_geometric.data import Batch

                    ckpt = torch.load(model_path, map_location="cpu")
                    model = build_model(
                        ckpt.get("model_type", "gin"),
                        ckpt.get("num_node_features", 9),
                        ckpt.get("args", {}).get("hidden_dim", 128),
                        ckpt.get("args", {}).get("num_layers", 4),
                        ckpt.get("args", {}).get("dropout", 0.2),
                    )
                    model.load_state_dict(ckpt["model_state"])
                    model.eval()

                    graph = smiles_to_pyg(smiles)
                    batch = Batch.from_data_list([graph])
                    with torch.no_grad():
                        logit = model(batch.x, batch.edge_index, batch.batch)
                        prob = torch.sigmoid(logit).item()

                    st.metric("HIV Inhibition Probability", f"{prob:.4f}")
                    if prob > 0.5:
                        st.success(f"⚠️ HIGH probability of HIV inhibition ({prob:.1%})")
                    else:
                        st.info(f"✅ LOW probability of HIV inhibition ({prob:.1%})")

                    # Show molecule structure
                    from rdkit.Chem import Draw
                    from rdkit.Chem.Draw import rdMolDraw2D
                    import io
                    drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
                    drawer.DrawMolecule(mol)
                    drawer.FinishDrawing()
                    st.image(io.BytesIO(drawer.GetDrawingText().encode()), width=400)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Model trained on ogbg-molhiv (scaffold split) · Metric: ROC-AUC · "
           "[GitHub](https://github.com/yuriao/GNN_drug_discovery)")
