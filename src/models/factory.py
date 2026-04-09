from __future__ import annotations

from src.models.gcn import GCNClassifier
from src.models.gin import GINClassifier


def build_model(model_name: str, num_node_features: int, hidden_dim: int, num_layers: int, dropout: float):
    name = model_name.lower()
    if name == "gcn":
        return GCNClassifier(num_node_features=num_node_features, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    if name == "gin":
        return GINClassifier(num_node_features=num_node_features, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    raise ValueError(f"Unknown model '{model_name}'. Expected one of: gcn, gin")
