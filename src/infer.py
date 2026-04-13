from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rdkit import Chem
from torch_geometric.data import Batch

from src.data.smiles_to_graph import smiles_to_pyg
from src.models.ensemble import load_ensemble
from src.models.factory import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Inference from SMILES (single model or ensemble)")
    parser.add_argument("--smiles", type=str, required=True)
    # Single model mode
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model", choices=["gcn", "gin", "gine"], default=None)
    # Ensemble mode (provide all three checkpoints)
    parser.add_argument("--gin-checkpoint",  type=str, default=None)
    parser.add_argument("--gcn-checkpoint",  type=str, default=None)
    parser.add_argument("--gine-checkpoint", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {args.smiles}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph  = smiles_to_pyg(args.smiles)
    batch  = Batch.from_data_list([graph]).to(device)

    # ── Ensemble mode ─────────────────────────────────────────────────────────
    ensemble_ckpts = {
        "gin":  args.gin_checkpoint,
        "gcn":  args.gcn_checkpoint,
        "gine": args.gine_checkpoint,
    }
    if any(v for v in ensemble_ckpts.values()):
        ensemble = load_ensemble(
            {k: v for k, v in ensemble_ckpts.items() if v},
            device,
        )
        with torch.no_grad():
            prob = ensemble(batch.x, batch.edge_index, batch.batch,
                            edge_attr=batch.edge_attr).item()
        print(f"SMILES : {args.smiles}")
        print(f"Ensemble HIV inhibition probability : {prob:.4f}")
        return

    # ── Single model mode ─────────────────────────────────────────────────────
    if not args.checkpoint or not args.model:
        raise ValueError("Provide either --gin/gcn/gine-checkpoint flags (ensemble) "
                         "or both --checkpoint and --model (single model).")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(
        args.model,
        ckpt.get("num_node_features", 9),
        ckpt.get("args", {}).get("hidden_dim", 128),
        ckpt.get("args", {}).get("num_layers", 4),
        ckpt.get("args", {}).get("dropout", 0.2),
        num_edge_features=ckpt.get("num_edge_features", 3),
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()

    with torch.no_grad():
        if args.model == "gine":
            logit = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr)
        else:
            logit = model(batch.x, batch.edge_index, batch.batch)
        prob = torch.sigmoid(logit).item()

    print(f"SMILES : {args.smiles}")
    print(f"[{args.model.upper()}] HIV inhibition probability : {prob:.4f}")


if __name__ == "__main__":
    main()
