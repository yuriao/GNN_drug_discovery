from __future__ import annotations
"""
Enhanced GIN with three improvements:
  1. Virtual Node — global "super-node" connected to all atoms for long-range communication
  2. Jumping Knowledge (JK) — concatenate embeddings from ALL layers, not just the last
  3. Global RDKit features — inject molecular descriptors into the MLP head

These are the top-3 literature-backed improvements for GNN on ogbg-molhiv.
Together they typically add +2–4% AUC over vanilla GIN.

References:
  Virtual node: Gilmer et al. (2017) MPNN; used in DeeperGCN+FLAG (OGB #18)
  Jumping Knowledge: Xu et al. (2018) "Representation Learning on Graphs with JK Networks"
  Global molecular features: FingerPrint+GMAN (#6, 0.824), Neural FingerPrints (#7, 0.823)
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool

from src.data.mol_features import NUM_GLOBAL_FEATURES


class EnhancedGINClassifier(nn.Module):
    """
    GIN with Virtual Node + Jumping Knowledge + Global RDKit Feature injection.

    Architecture:
        Input: node features [N, node_feat_dim] + optional global_feat [B, 12]

        1. Virtual node embedding: one learnable vector per graph, updated each layer
        2. GINConv layers: each layer also aggregates virtual node message
        3. JK-concat: concatenate all layer outputs → [N, hidden_dim * num_layers]
        4. Global sum pool → [B, hidden_dim * num_layers]
        5. Concatenate global RDKit features → [B, hidden_dim * num_layers + 12]
        6. MLP head → [B, 1]
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        use_virtual_node: bool = True,
        use_jk: bool = True,
        use_global_features: bool = True,
        num_global_features: int = NUM_GLOBAL_FEATURES,
    ):
        super().__init__()
        self.hidden_dim          = hidden_dim
        self.num_layers          = num_layers
        self.dropout             = dropout
        self.use_virtual_node    = use_virtual_node
        self.use_jk              = use_jk
        self.use_global_features = use_global_features

        # ── GIN convolution layers ────────────────────────────────────────────
        self.convs = nn.ModuleList()
        mlp0 = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(mlp0, train_eps=True))
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))

        # ── Virtual node components ───────────────────────────────────────────
        if use_virtual_node:
            # Embedding for the virtual node (one per graph)
            self.vnode_emb = nn.Embedding(1, hidden_dim)
            nn.init.constant_(self.vnode_emb.weight, 0)
            # MLP to update virtual node after each layer
            self.vnode_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers - 1)  # update after layers 1..K-1
            ])

        # ── Batch norms per layer ─────────────────────────────────────────────
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # ── JK aggregation ────────────────────────────────────────────────────
        # JK-concat: pool_dim = hidden_dim * num_layers
        # JK-none:   pool_dim = hidden_dim
        pool_dim = hidden_dim * num_layers if use_jk else hidden_dim

        # ── MLP head ──────────────────────────────────────────────────────────
        head_in = pool_dim + (num_global_features if use_global_features else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, batch, global_feat=None, return_node_embeddings=False):
        x = x.float()
        num_nodes = x.size(0)
        num_graphs = int(batch.max().item()) + 1

        # ── Initialise virtual node embeddings ────────────────────────────────
        if self.use_virtual_node:
            vnode = self.vnode_emb(
                torch.zeros(num_graphs, dtype=torch.long, device=x.device)
            )  # [num_graphs, hidden_dim]

        layer_outputs = []

        for i, conv in enumerate(self.convs):
            # Add virtual node signal to atom features before convolution
            if self.use_virtual_node and i > 0:
                # Broadcast virtual node to each atom in its graph
                x = x + vnode[batch]

            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outputs.append(x)

            # Update virtual node: aggregate over all atoms in each graph
            if self.use_virtual_node and i < self.num_layers - 1:
                # Pool all atoms → one vector per graph
                vnode_agg = global_add_pool(x, batch)   # [num_graphs, hidden_dim]
                vnode = F.dropout(
                    self.vnode_mlps[i](vnode + vnode_agg),
                    p=self.dropout, training=self.training
                )

        # ── Jumping Knowledge ─────────────────────────────────────────────────
        if self.use_jk:
            # Concatenate all layer outputs: [N, hidden_dim * num_layers]
            x_jk = torch.cat(layer_outputs, dim=1)
        else:
            x_jk = layer_outputs[-1]  # just last layer

        node_embeddings = x_jk

        # ── Global sum pooling ────────────────────────────────────────────────
        graph_emb = global_add_pool(x_jk, batch)  # [B, pool_dim]

        # ── Inject global RDKit features ──────────────────────────────────────
        if self.use_global_features and global_feat is not None:
            gf = global_feat.float()
            if gf.dim() == 3:
                gf = gf.squeeze(1)  # [B, 12]
            graph_emb = torch.cat([graph_emb, gf], dim=1)

        out = self.head(graph_emb).view(-1)

        if return_node_embeddings:
            return out, node_embeddings
        return out
