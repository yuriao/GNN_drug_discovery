from __future__ import annotations

from src.models.gcn import GCNClassifier
from src.models.gin import GINClassifier
from src.models.gine import GINEClassifier


def build_model(
    model_name: str,
    num_node_features: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    num_edge_features: int = 3,
):
    name = model_name.lower()
    if name == "gcn":
        return GCNClassifier(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    if name == "gin":
        return GINClassifier(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    if name == "gine":
        return GINEClassifier(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model '{model_name}'. Expected one of: gcn, gin, gine")
