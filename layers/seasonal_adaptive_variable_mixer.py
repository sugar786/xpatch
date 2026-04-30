import torch
import torch.nn as nn


class SeasonalAdaptiveVariableMixer(nn.Module):
    """
    Seasonal / residual stream variable modeling for xPatch.

    This module is designed for seasonal patch representations.

    Input:
        x: [B, C, P, D]
           B: batch size
           C: number of variables
           P: number of patches
           D: patch hidden dimension

    Output:
        delta: [B, C, P, D]

    Main idea:
        1. Learn an adaptive sparse variable graph.
        2. Aggregate neighboring variable representations at each patch position.
        3. Use a gated fusion mechanism to control how much neighbor information is injected.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        node_dim: int = 16,
        topk: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.c_in = c_in
        self.topk = topk

        # Adaptive variable graph, inspired by adaptive adjacency learning.
        self.node_emb1 = nn.Parameter(torch.randn(c_in, node_dim))
        self.node_emb2 = nn.Parameter(torch.randn(node_dim, c_in))

        # Stabilize seasonal / residual patch features.
        self.norm = nn.LayerNorm(d_model)

        # Neighbor value projection.
        self.value_proj = nn.Linear(d_model, d_model)

        # Gated fusion. The gate is patch-wise, variable-wise, and feature-wise.
        self.gate_proj = nn.Linear(d_model * 2, d_model)

        # Output projection.
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def build_adj(self):
        """
        Build adaptive sparse variable adjacency.

        Returns:
            adj: [C, C]
                 Row-normalized adjacency matrix.
                 adj[i, j] means variable j contributes to variable i.
        """
        adj = torch.relu(torch.tanh(self.node_emb1 @ self.node_emb2))  # [C, C]

        # Remove self-loop. The residual path already preserves self information.
        eye = torch.eye(self.c_in, device=adj.device, dtype=torch.bool)
        adj = adj.masked_fill(eye, 0.0)

        # Sparse top-k neighbor selection.
        if self.topk is not None and self.topk > 0:
            k = min(self.topk, max(self.c_in - 1, 1))
            idx = torch.topk(adj, k=k, dim=-1).indices  # [C, k]

            mask = torch.zeros_like(adj)
            mask.scatter_(1, idx, 1.0)
            adj = adj * mask

        # Row normalization.
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)
        return adj

    def forward(self, x):
        """
        Args:
            x: [B, C, P, D]

        Returns:
            delta: [B, C, P, D]
        """
        z = self.norm(x)

        adj = self.build_adj()  # [C, C]

        # Aggregate neighbor variables for each patch position.
        # adj[i, j] means variable j contributes to variable i.
        neigh = torch.einsum("ij,bjpd->bipd", adj, z)  # [B, C, P, D]
        neigh = self.value_proj(neigh)

        # Gated fusion.
        gate = torch.sigmoid(self.gate_proj(torch.cat([z, neigh], dim=-1)))
        delta = gate * neigh

        delta = self.out_proj(delta)
        delta = self.dropout(delta)

        return delta
