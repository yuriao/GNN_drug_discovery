from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, global_add_pool


class GINEClassifier(nn.Module):
    """
    Graph Isomorphism Network with Edge features (GINE).

    Extends GIN by incorporating edge_attr in the message-passing step via GINEConv.
    Bond type, stereochemistry, and ring membership are used — unlike plain GIN which
    ignores edge_attr entirely.

    Reference: Hu et al., "Strategies for Pre-training Graph Neural Networks", ICLR 2020
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # First layer: node_features -> hidden_dim
        mlp0 = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINEConv(mlp0, edge_dim=num_edge_features))

        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=num_edge_features))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, batch, edge_attr=None, return_node_embeddings: bool = False):
        x = x.float()
        ea = edge_attr.float() if edge_attr is not None else torch.zeros(
            edge_index.shape[1], self.convs[0].edge_dim, device=x.device
        )
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, ea))
            x = F.dropout(x, p=self.dropout, training=self.training)
        node_embeddings = x
        x = global_add_pool(x, batch)
        out = self.head(x).view(-1)
        if return_node_embeddings:
            return out, node_embeddings
        return out
