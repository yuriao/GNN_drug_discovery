from __future__ import annotations

import argparse
import json

import torch

from src.data.load_ogb import build_dataloaders
from src.models.factory import build_model
from src.train import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on ogbg-molhiv")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", choices=["gcn", "gin"], required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    dataset, _, _, valid_loader, test_loader = build_dataloaders(batch_size=args.batch_size)
    model = build_model(
        args.model,
        ckpt.get("num_node_features", dataset[0].x.shape[1]),
        ckpt["args"].get("hidden_dim", 128),
        ckpt["args"].get("num_layers", 4),
        ckpt["args"].get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    val_metrics = evaluate(model, valid_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    out = {"val": val_metrics, "test": test_metrics}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
