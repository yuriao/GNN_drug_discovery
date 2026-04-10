from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool


class GINClassifier(nn.Module):
    def __init__(self, num_node_features: int, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        mlp_in = nn.Sequential(nn.Linear(num_node_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(mlp_in))

        for _ in range(num_layers - 1):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, batch, return_node_embeddings: bool = False):
        x = x.float()
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        node_embeddings = x  # [N, hidden_dim] — per-atom representations
        x = global_add_pool(x, batch)
        out = self.head(x).view(-1)
        if return_node_embeddings:
            return out, node_embeddings
        return out
