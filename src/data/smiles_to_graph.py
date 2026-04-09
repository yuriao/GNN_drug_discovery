from __future__ import annotations

import json
from pathlib import Path

import torch
from ogb.utils import smiles2graph
from torch_geometric.data import Data


def smiles_to_pyg(smiles: str) -> Data:
    graph = smiles2graph(smiles)
    data = Data(
        x=torch.from_numpy(graph["node_feat"]).to(torch.long),
        edge_index=torch.from_numpy(graph["edge_index"]).to(torch.long),
        edge_attr=torch.from_numpy(graph["edge_feat"]).to(torch.long),
    )
    return data


def summarize_smiles(smiles_list: list[str], outdir: str) -> Path:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    num_valid = 0
    total_nodes = 0
    total_edges = 0

    for s in smiles_list:
        try:
            g = smiles_to_pyg(s)
            num_valid += 1
            total_nodes += int(g.num_nodes)
            total_edges += int(g.edge_index.shape[1])
        except Exception:
            continue

    summary = {
        "total_smiles": len(smiles_list),
        "valid_smiles": num_valid,
        "invalid_smiles": len(smiles_list) - num_valid,
        "avg_nodes": (total_nodes / max(num_valid, 1)),
        "avg_edges": (total_edges / max(num_valid, 1)),
    }

    file_path = out_path / "preprocessing_summary.json"
    file_path.write_text(json.dumps(summary, indent=2))
    return file_path
