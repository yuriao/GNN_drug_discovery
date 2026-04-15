from __future__ import annotations

from src.models.gcn import GCNClassifier
from src.models.gin import GINClassifier
from src.models.gine import GINEClassifier
from src.models.gin_enhanced import EnhancedGINClassifier


def build_model(
    model_name: str,
    num_node_features: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    num_edge_features: int = 3,
    # Enhanced GIN options
    use_virtual_node: bool = True,
    use_jk: bool = True,
    use_global_features: bool = True,
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
        # Default GIN — unchanged for backward compat
        return GINClassifier(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    if name == "gin_enhanced":
        return EnhancedGINClassifier(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_virtual_node=use_virtual_node,
            use_jk=use_jk,
            use_global_features=use_global_features,
        )
    if name == "gine":
        return GINEClassifier(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model '{model_name}'. Expected: gcn, gin, gin_enhanced, gine")
