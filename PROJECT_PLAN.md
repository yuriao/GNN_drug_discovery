# Project Plan: A Simple GNN Project for Molecular Biology

## 1) Proposed starter project
**Title:** Predict HIV inhibition from molecular graphs (`ogbg-molhiv`) using a Graph Neural Network.

**Why this project is a good first choice**
- It is a **real biological task**: classify whether a molecule inhibits HIV replication.
- The dataset (`ogbg-molhiv`) is **well-curated and standard**, so results are comparable.
- The scope is simple enough for a first repo while still scientifically meaningful.
- It has an established train/validation/test split (scaffold split), which is useful for realistic generalization checks.

## 2) Online references used (for plan grounding)
- OGB graph property docs: `ogbg-molhiv` is a scaffold-split binary classification benchmark with ROC-AUC evaluation.
- PyTorch Geometric docs: standard graph representation (`Data` objects with node/edge features).
- RDKit + OGB notes: molecules are represented as atoms (nodes) and bonds (edges); RDKit is used in molecular preprocessing pipelines.
- Tox21 official data/tools page: confirms availability of public bioassay/toxicity resources for future extension tasks.

## 3) Project objectives (MVP)
1. Build a reproducible pipeline for molecular graph classification.
2. Train a baseline GNN (GCN or GIN) on `ogbg-molhiv`.
3. Report ROC-AUC on validation/test with clear experiment tracking.
4. Provide a minimal inference script for user-provided SMILES.

## 4) Technical stack (simple and stable)
- **Language:** Python 3.11+
- **Core ML:** PyTorch
- **GNN library:** PyTorch Geometric
- **Molecular toolkit:** RDKit
- **Benchmark/data API:** OGB
- **Experiment tracking (lightweight):** CSV + TensorBoard (optional: Weights & Biases)

## 5) Suggested repository structure
```text
GNN_drug_discovery/
  README.md
  pyproject.toml (or requirements.txt)
  data/
  src/
    data/
      load_ogb.py
      smiles_to_graph.py
    models/
      gcn.py
      gin.py
    train.py
    evaluate.py
    infer.py
    utils/
      seed.py
      metrics.py
  configs/
    default.yaml
  outputs/
  tests/
    test_data_pipeline.py
    test_model_forward.py
```

## 6) Step-by-step implementation plan

### Phase A — Setup (Day 1)
- Create Python environment and install dependencies.
- Add formatting/linting (`ruff`, `black`) and basic CI (optional but recommended).
- Add deterministic seed utilities and config file.

**Deliverable:** clean environment + runnable template scripts.

### Phase B — Data pipeline (Day 1–2)
- Load `ogbg-molhiv` via OGB/PyG.
- Implement dataloaders for train/valid/test using the official scaffold split.
- Add quick EDA script:
  - class balance,
  - molecule size distribution,
  - missing labels checks.

**Deliverable:** `python -m src.data.load_ogb` prints dataset stats and split sizes.

### Phase C — Baseline model training (Day 2–3)
- Implement **GCN baseline** with:
  - graph conv layers,
  - global pooling,
  - MLP head.
- Train with BCE loss; log ROC-AUC.
- Add early stopping on validation ROC-AUC.

**Deliverable:** baseline checkpoint + metrics report.

### Phase D — Stronger baseline (Day 3–4)
- Add **GIN model** (often strong for molecular tasks).
- Compare GCN vs GIN under matched training settings.
- Save experiment summary table (model, params, val/test ROC-AUC, runtime).

**Deliverable:** model comparison markdown/csv report.

### Phase E — Inference utility (Day 4)
- Add script: input SMILES → graph → predicted HIV inhibition probability.
- Include input validation + clear error messages for invalid SMILES.

**Deliverable:** `infer.py` usable from CLI.

### Phase F — Documentation and reproducibility (Day 4–5)
- Add README with quickstart, training, evaluation, and inference commands.
- Pin versions and add reproducibility notes.
- Add 2–3 unit tests for data/model sanity.

**Deliverable:** first complete release of MVP repository.

## 7) Minimal experimental protocol
- **Primary metric:** ROC-AUC (aligned with `ogbg-molhiv`).
- **Secondary checks:** PR-AUC, confusion matrix at tuned threshold.
- **Reproducibility:** run each model with 3 seeds and report mean ± std.
- **Stopping rule:** early stop by validation ROC-AUC patience (e.g., 20 epochs).

## 8) Risks and mitigations
- **Class imbalance / unstable metrics** → track both ROC-AUC and PR-AUC; use seed averaging.
- **Overfitting** → dropout, weight decay, early stopping.
- **Data leakage** → always use official scaffold split.
- **Chemistry parsing errors in custom inference** → RDKit validation and user feedback.

## 9) Optional extensions after MVP
1. Multi-task prediction on `ogbg-molpcba`.
2. Add interpretability (e.g., atom importance saliency).
3. Domain adaptation to a custom assay dataset (e.g., Tox21-derived tasks).
4. Try modern molecular GNN variants (GraphGPS/MPNN-style architectures).

## 10) Acceptance criteria for MVP
- One-command train/eval scripts run end-to-end.
- Reproducible val/test ROC-AUC for GCN + GIN.
- CLI inference works on example SMILES in README.
- At least basic tests pass in CI/local.

---

## Recommendation for your approval
If this looks good, the next step is to implement **Phase A + B + C only** first (smallest useful milestone), then review metrics before expanding to GIN and inference.
