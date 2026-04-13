# GNN Drug Discovery — HIV Inhibition Prediction

[![Train GNN Model](https://github.com/yuribo/GNN_drug_discovery/actions/workflows/train.yml/badge.svg)](https://github.com/yuribo/GNN_drug_discovery/actions/workflows/train.yml)
[![Live Dashboard](https://img.shields.io/badge/🧬_Dashboard-HuggingFace_Spaces-yellow)](https://huggingface.co/spaces/yuriao/gnn-drug-discovery)

An end-to-end Graph Neural Network (GNN) pipeline for molecular biology. Predicts whether a molecule inhibits HIV replication using the **ogbg-molhiv** benchmark (41,127 molecules, scaffold split, ROC-AUC metric).

---

## Live Demo

🧬 **[Open Dashboard on Hugging Face Spaces](https://huggingface.co/spaces/yuriao/gnn-drug-discovery)**

The dashboard provides:
- **Dataset stats** — molecule counts, feature dimensions, split sizes
- **Training curves** — loss, ROC-AUC, PR-AUC per epoch (interactive Plotly charts)
- **Final results** — best val/test ROC-AUC vs OGB baseline benchmarks
- **Live inference** — paste any SMILES string → get HIV inhibition probability + molecule visualization

---

## Architecture

```
SMILES string
     ↓
OGB / RDKit → PyG graph (atoms=nodes, bonds=edges)
     ↓
GCN or GIN (4 message-passing layers, hidden_dim=128)
     ↓
global pooling → graph embedding
     ↓
MLP head → sigmoid → HIV inhibition probability [0, 1]
```

### Models

| Model | Pooling | Benchmark ROC-AUC |
|-------|---------|-------------------|
| **GCN** | mean | ~0.759 |
| **GIN** | sum | ~0.771 |

GIN (Graph Isomorphism Network) is generally stronger for molecular tasks due to its more expressive per-layer MLP aggregation.

---

## Repository Structure

```
GNN_drug_discovery/
├── src/
│   ├── data/
│   │   ├── load_ogb.py          # OGB dataset loading + scaffold split
│   │   └── smiles_to_graph.py   # SMILES → PyG Data object
│   ├── models/
│   │   ├── gcn.py               # GCNClassifier (GCNConv + global_mean_pool)
│   │   ├── gin.py               # GINClassifier (GINConv + global_add_pool)
│   │   └── factory.py           # build_model() helper
│   ├── utils/
│   │   ├── metrics.py           # safe_roc_auc, safe_pr_auc
│   │   └── seed.py              # deterministic seed utility
│   ├── train.py                 # Training loop + early stopping + checkpoint
│   ├── evaluate.py              # Load checkpoint → val/test metrics
│   ├── infer.py                 # CLI inference from SMILES
│   └── run_pipeline.py          # Preprocessing summary helper
├── dashboard/
│   └── app.py                   # Local Streamlit dashboard
├── app.py                       # Hugging Face Spaces entry point
├── configs/
│   └── default.yaml             # Default hyperparameters
├── outputs/                     # Training artifacts (gitignored except examples)
├── .github/
│   └── workflows/
│       └── train.yml            # CI: auto-train + publish release artifacts
└── requirements.txt
```

---

## Deployment

### Automatic Training (GitHub Actions)

Training runs automatically on every push that touches `src/`, `configs/`, or `requirements.txt`.

**Workflow:** `.github/workflows/train.yml`
- Installs PyTorch (CPU), torch-geometric, OGB, RDKit
- Trains GIN for 15 epochs with early stopping (patience=5)
- Saves `best_model.pt`, `metrics.csv`, `run_summary.json`, `dataset_stats.json`
- Publishes all artifacts to **GitHub Release** tag `model-latest`

**Manual trigger** (with custom params):
```
GitHub → Actions → Train GNN Model → Run workflow
  model: gin  (or gcn)
  epochs: 30
```

Artifacts are available at:
```
https://github.com/yuriao/GNN_drug_discovery/releases/tag/model-latest
```

### Live Dashboard (Hugging Face Spaces)

**URL:** `https://huggingface.co/spaces/yuriao/gnn-drug-discovery`

The Space runs `app.py` (root level). On startup it:
1. Downloads model artifacts from the `model-latest` GitHub Release
2. Serves the full Streamlit dashboard with 4 tabs
3. Enables live SMILES → HIV inhibition probability inference

The dashboard auto-updates each time a new model is trained and published.

**Re-deploy after new training run:**
Training CI automatically publishes a new `model-latest` release → the HF Space picks up new artifacts on next cold start (or force a restart from the HF Space settings).

---

## Local Setup

```bash
# 1. Clone
git clone https://github.com/yuriao/GNN_drug_discovery
cd GNN_drug_discovery

# 2. Create environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies (CPU)
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.5.3
pip install "numpy<2" ogb rdkit scikit-learn pandas pyyaml tqdm streamlit plotly
```

## Train a Model

```bash
# GIN baseline (recommended)
python -m src.train \
  --model gin \
  --epochs 30 \
  --batch-size 64 \
  --hidden-dim 128 \
  --num-layers 4 \
  --dropout 0.2 \
  --patience 10 \
  --outdir outputs/run_gin

# GCN baseline
python -m src.train --model gcn --epochs 30 --outdir outputs/run_gcn
```

## Evaluate a Checkpoint

```bash
python -m src.evaluate \
  --checkpoint outputs/run_gin/best_model.pt \
  --model gin \
  --batch-size 64
```

## Infer from SMILES

```bash
# Predict HIV inhibition probability for aspirin
python -m src.infer \
  --checkpoint outputs/run_gin/best_model.pt \
  --model gin \
  --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"

# Output:
# SMILES: CC(=O)OC1=CC=CC=C1C(=O)O
# Predicted HIV inhibition probability: 0.0231
```

## Run Local Dashboard

```bash
# First train a model, then:
streamlit run dashboard/app.py
# → Opens at http://localhost:8501
```

---

## Key Design Decisions

- **Scaffold split** (not random) — ensures molecules with similar scaffolds don't leak between train/test; gives realistic generalization estimates
- **ROC-AUC primary metric** — standard for imbalanced binary classification (~1.5% positive rate in molhiv)
- **NaN masking** — some molecules have missing labels; loss/eval skip them cleanly
- **Early stopping** — patience on val ROC-AUC prevents overfitting on small positive class
- **CPU-first training** — ogbg-molhiv is tractable on CPU in 15–30 min; no GPU required for baseline results

---

## Extending This Project

1. **Multi-task** — swap to `ogbg-molpcba` (128 tasks)
2. **Interpretability** — add GradCAM / atom importance saliency maps
3. **Modern architectures** — try GraphGPS or MPNN variants
4. **Custom inference** — upload your own molecule dataset as CSV of SMILES
5. **Tox21 extension** — toxicity prediction on public bioassay data

---

## References

- [Open Graph Benchmark (OGB)](https://ogb.stanford.edu/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [RDKit](https://www.rdkit.org/)
- [How Powerful are Graph Neural Networks? (Xu et al., 2019)](https://arxiv.org/abs/1810.00826) — GIN paper
