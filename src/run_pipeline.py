from __future__ import annotations

import argparse
from pathlib import Path

from src.data.smiles_to_graph import summarize_smiles


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline helper utilities")
    parser.add_argument("--outdir", type=str, default="outputs/run")
    parser.add_argument(
        "--sample-smiles",
        nargs="*",
        default=[
            "CC(=O)OC1=CC=CC=C1C(=O)O",
            "CCN(CC)CCOC(=O)C1=CC=CC=C1Cl",
            "c1ccccc1",
            "N[C@@H](CC1=CC=CC=C1)C(O)=O",
        ],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    summary_path = summarize_smiles(args.sample_smiles, args.outdir)
    print(f"Saved preprocessing summary: {summary_path}")


if __name__ == "__main__":
    main()
