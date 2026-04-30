import torch
import torch.nn as nn


class SeasonalAdaptiveVariableMixer(nn.Module):
    """
    Variable-aware seasonal / residual feature calibration for xPatch.

    This module is designed for xPatch's seasonal / residual patch stream.

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
        2. Use neighboring variables only to produce a gate.
        3. Calibrate the current variable's own seasonal / residual feature.
        4. Avoid directly injecting neighbor features into the seasonal stream.

    Difference from direct variable mixing:
        Direct mixing:
            delta_i = gate_i * neighbor_i

        This module:
            delta_i = gate_i * self_i

    This is more conservative and is usually safer for high-frequency
    seasonal / residual features, especially for long-horizon forecasting.
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

        # Adaptive variable graph.
        # adj[i, j] means variable j contributes to the gate of variable i.
        self.node_emb1 = nn.Parameter(torch.randn(c_in, node_dim))
        self.node_emb2 = nn.Parameter(torch.randn(node_dim, c_in))

        # Normalize patch features along hidden dimension.
        self.norm = nn.LayerNorm(d_model)

        # Neighbor feature projection used only for gate generation.
        self.neigh_proj = nn.Linear(d_model, d_model)

        # Self feature projection before calibration.
        self.self_proj = nn.Linear(d_model, d_model)

        # Gate generation from [self feature, neighbor summary].
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

        # Remove self-loop. Self information is preserved by the residual path.
        eye = torch.eye(self.c_in, device=adj.device, dtype=torch.bool)
        adj = adj.masked_fill(eye, 0.0)

        # Sparse top-k variable selection.
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
        # Stabilize seasonal / residual patch features.
        z = self.norm(x)  # [B, C, P, D]

        # Build adaptive variable graph.
        adj = self.build_adj()  # [C, C]

        # Neighbor summary for each variable and each patch.
        # Important: this neighbor summary is only used to generate gates.
        neigh = torch.einsum("ij,bjpd->bipd", adj, z)  # [B, C, P, D]
        neigh = self.neigh_proj(neigh)

        # Self feature to be calibrated.
        self_feat = self.self_proj(z)

        # Variable-aware gate.
        gate = torch.sigmoid(self.gate_proj(torch.cat([z, neigh], dim=-1)))

        # Conservative calibration:
        # use neighbor information only to decide the gate,
        # but do not directly inject neighbor features.
        delta = gate * self_feat

        delta = self.out_proj(delta)
        delta = self.dropout(delta)

        return delta
