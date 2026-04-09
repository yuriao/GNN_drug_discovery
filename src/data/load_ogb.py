from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


def build_dataloaders(
    dataset_name: str = "ogbg-molhiv",
    batch_size: int = 64,
    augment_positives: int = 0,
):
    """
    Load ogbg-molhiv with optional SMILES augmentation of positive (active) samples.

    augment_positives: number of random SMILES variants to generate per positive
    training molecule. 0 = no augmentation.

    Based on: Li et al. (2022) "A Novel Molecular Representation Learning for
    Molecular Property Prediction with a Multiple SMILES-Based Augmentation"
    Computational Intelligence and Neuroscience.
    """
    dataset = PygGraphPropPredDataset(name=dataset_name)
    split_idx = dataset.get_idx_split()

    train_dataset = dataset[split_idx["train"]]
    valid_dataset = dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]]

    if augment_positives > 0:
        train_dataset = _augment_positive_smiles(
            dataset, split_idx["train"], train_dataset, augment_positives
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataset, split_idx, train_loader, valid_loader, test_loader


def _augment_positive_smiles(
    dataset: PygGraphPropPredDataset,
    train_idx,
    train_dataset,
    n_augments: int,
) -> list:
    """
    For each positive molecule in the training set, generate n_augments additional
    graph representations by randomising its SMILES string ordering via RDKit.

    This is the canonical SMILES augmentation technique for molecular GNNs:
    one molecule → many valid SMILES → many different but equivalent PyG graphs.
    Only applied to positives to balance the class distribution.

    [Bjerrum 2017; Li et al. 2022]
    """
    try:
        from rdkit import Chem
        from ogb.utils import smiles2graph
    except ImportError:
        logger.warning("RDKit not available — skipping SMILES augmentation")
        return list(train_dataset)

    # Get SMILES for all training molecules from OGB
    smiles_list = dataset.smiles_list if hasattr(dataset, "smiles_list") else None
    if smiles_list is None:
        # Load from raw data
        try:
            import pandas as pd
            raw_path = Path(dataset.root) / "mapping" / "mol.csv.gz"
            df = pd.read_csv(raw_path)
            smiles_col = [c for c in df.columns if "smiles" in c.lower()][0]
            all_smiles = df[smiles_col].tolist()
            train_smiles = [all_smiles[i] for i in train_idx.tolist()]
        except Exception as e:
            logger.warning("Could not load SMILES for augmentation: %s", e)
            return list(train_dataset)
    else:
        train_smiles = [smiles_list[i] for i in train_idx.tolist()]

    augmented = list(train_dataset)  # start with originals
    n_pos, n_aug_added = 0, 0

    for i, (data, smi) in enumerate(zip(train_dataset, train_smiles)):
        label = data.y.view(-1)[0].item()
        if label != 1:
            continue
        n_pos += 1

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        added = 0
        attempts = 0
        seen_smiles = {smi}

        while added < n_augments and attempts < n_augments * 3:
            attempts += 1
            try:
                # Generate a random SMILES by randomising atom ordering
                rand_smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                if rand_smi in seen_smiles:
                    continue
                seen_smiles.add(rand_smi)

                # Convert to graph
                graph = smiles2graph(rand_smi)
                node_feat = torch.from_numpy(graph["node_feat"]).to(torch.long)
                aug_data = Data(
                    x=node_feat,
                    edge_index=torch.from_numpy(graph["edge_index"]).to(torch.long),
                    edge_attr=torch.from_numpy(graph["edge_feat"]).to(torch.long),
                    y=data.y.clone(),
                    num_nodes=node_feat.size(0),
                )
                augmented.append(aug_data)
                added += 1
                n_aug_added += 1
            except Exception:
                continue

    n_total_pos = n_pos + n_aug_added
    n_neg = sum(1 for d in train_dataset if d.y.view(-1)[0].item() == 0)
    logger.info(
        "SMILES augmentation: %d original positives → %d total positives "
        "(%d added) | negatives: %d | new ratio: %.1f%%",
        n_pos, n_total_pos, n_aug_added, n_neg,
        100 * n_total_pos / (n_total_pos + n_neg),
    )
    print(
        f"Augmentation: {n_pos} positives × {n_augments} → {n_aug_added} added "
        f"({n_total_pos} total pos vs {n_neg} neg = "
        f"{100*n_total_pos/(n_total_pos+n_neg):.1f}% positive rate)"
    )
    return augmented


def save_dataset_stats(dataset, split_idx, outdir: str) -> Path:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    stats = {
        "num_graphs": len(dataset),
        "num_tasks": int(dataset.num_tasks),
        "num_node_features": int(dataset[0].x.shape[1]),
        "num_edge_features": int(dataset[0].edge_attr.shape[1]),
        "split_sizes": {
            "train": int(len(split_idx["train"])),
            "valid": int(len(split_idx["valid"])),
            "test": int(len(split_idx["test"])),
        },
    }

    file_path = out_path / "dataset_stats.json"
    file_path.write_text(json.dumps(stats, indent=2))
    return file_path


if __name__ == "__main__":
    dataset, split_idx, *_ = build_dataloaders()
    path = save_dataset_stats(dataset, split_idx, "outputs/dataset_info")
    print(f"Saved dataset stats to: {path}")
