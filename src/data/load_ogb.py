from __future__ import annotations

import json
from pathlib import Path

from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader


def build_dataloaders(dataset_name: str = "ogbg-molhiv", batch_size: int = 64):
    dataset = PygGraphPropPredDataset(name=dataset_name)
    split_idx = dataset.get_idx_split()

    train_dataset = dataset[split_idx["train"]]
    valid_dataset = dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataset, split_idx, train_loader, valid_loader, test_loader


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
