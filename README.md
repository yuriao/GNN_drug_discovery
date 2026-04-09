# GNN Drug Discovery Starter (Molecular Biology)

This repository implements an end-to-end Graph Neural Network (GNN) project for molecular biology problems using the `ogbg-molhiv` benchmark.

## What is included

- Data collection and split loading from OGB (`ogbg-molhiv`)
- Molecular graph preprocessing from SMILES
- GCN and GIN baseline models
- Training + evaluation pipeline with ROC-AUC reporting
- CLI inference from SMILES
- Streamlit dashboard to visualize data, preprocessing, and model results

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train a model

```bash
python -m src.train --model gin --epochs 30 --batch-size 64 --hidden-dim 128 --outdir outputs/run_gin
```

## Evaluate a saved model

```bash
python -m src.evaluate --checkpoint outputs/run_gin/best_model.pt --model gin --batch-size 64
```

## Infer from SMILES

```bash
python -m src.infer --checkpoint outputs/run_gin/best_model.pt --model gin --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

## Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard reads output artifacts from `outputs/*`, including:

- `dataset_stats.json`
- `preprocessing_summary.json`
- `metrics.csv`
- `run_summary.json`

## Notes

- The training setup assumes `torch`, `torch-geometric`, and OGB are installed and compatible with your CUDA/CPU environment.
- For first review iterations, start with a small epoch count (e.g., 5–10) to validate the pipeline before longer runs.
