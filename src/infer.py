from __future__ import annotations

import argparse

import torch
from rdkit import Chem
from torch_geometric.data import Batch

from src.data.smiles_to_graph import smiles_to_pyg
from src.models.factory import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Inference from SMILES")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", choices=["gcn", "gin"], required=True)
    parser.add_argument("--smiles", type=str, required=True)
    return parser.parse_args()


def validate_smiles(smiles: str) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")


def main():
    args = parse_args()
    validate_smiles(args.smiles)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(
        args.model,
        ckpt["num_node_features"],
        ckpt["args"].get("hidden_dim", 128),
        ckpt["args"].get("num_layers", 4),
        ckpt["args"].get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    graph = smiles_to_pyg(args.smiles)
    batch = Batch.from_data_list([graph]).to(device)

    with torch.no_grad():
        logit = model(batch.x, batch.edge_index, batch.batch)
        prob = torch.sigmoid(logit).item()

    print(f"SMILES: {args.smiles}")
    print(f"Predicted HIV inhibition probability: {prob:.4f}")


if __name__ == "__main__":
    main()
