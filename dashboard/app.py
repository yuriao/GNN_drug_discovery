from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="GNN Drug Discovery Dashboard", layout="wide")
st.title("GNN Molecular Pipeline Dashboard")
st.caption("Showcasing data collection, preprocessing, training, and testing results.")

outputs_root = Path("outputs")
runs = sorted([p for p in outputs_root.glob("*") if p.is_dir()])

if not runs:
    st.warning("No run directories found under outputs/. Train a model first.")
    st.stop()

run_names = [r.name for r in runs]
selected_run = st.sidebar.selectbox("Select run", run_names, index=len(run_names) - 1)
run_dir = outputs_root / selected_run

st.header("1) Data collection")
dataset_stats_file = run_dir / "dataset_stats.json"
if dataset_stats_file.exists():
    stats = json.loads(dataset_stats_file.read_text())
    c1, c2, c3 = st.columns(3)
    c1.metric("Total graphs", stats.get("num_graphs", "-"))
    c2.metric("Tasks", stats.get("num_tasks", "-"))
    c3.metric("Node features", stats.get("num_node_features", "-"))
    st.json(stats)
else:
    st.info("dataset_stats.json not found for selected run.")

st.header("2) Preprocessing summary")
prep_file = run_dir / "preprocessing_summary.json"
if prep_file.exists():
    prep = json.loads(prep_file.read_text())
    p1, p2, p3 = st.columns(3)
    p1.metric("Valid SMILES", prep.get("valid_smiles", "-"))
    p2.metric("Invalid SMILES", prep.get("invalid_smiles", "-"))
    p3.metric("Avg atoms(nodes)", f"{prep.get('avg_nodes', 0):.2f}")
    st.json(prep)
else:
    st.info("preprocessing_summary.json not found. Run preprocessing summary generation.")

st.header("3) Training curves")
metrics_file = run_dir / "metrics.csv"
if metrics_file.exists():
    df = pd.read_csv(metrics_file)
    if not df.empty:
        fig1 = px.line(df, x="epoch", y=["train_loss"], title="Train Loss")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(df, x="epoch", y=["val_roc_auc", "test_roc_auc"], title="ROC-AUC by Epoch")
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.line(df, x="epoch", y=["val_pr_auc", "test_pr_auc"], title="PR-AUC by Epoch")
        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.info("metrics.csv is empty.")
else:
    st.info("metrics.csv not found for selected run.")

st.header("4) Testing summary")
summary_file = run_dir / "run_summary.json"
if summary_file.exists():
    summary = json.loads(summary_file.read_text())
    s1, s2, s3 = st.columns(3)
    s1.metric("Best validation ROC-AUC", f"{summary.get('best_val_roc_auc', float('nan')):.4f}")
    s2.metric("Test ROC-AUC @ best val", f"{summary.get('test_roc_auc_at_best_val', float('nan')):.4f}")
    s3.metric("Epochs ran", summary.get("epochs_ran", "-"))
    st.json(summary)
else:
    st.info("run_summary.json not found for selected run.")
