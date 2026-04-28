import torch
import torch.nn as nn


class SparseTrendInteractor(nn.Module):
    """
    Sparse cross-variable interaction on filtered trend representations.

    Input:
        h:           [B, C, D]
        topk_idx:    [B, C, K]
        topk_scores: [B, C, K] signed finite scores

    Output:
        delta:       [B, C, D]
    """

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()

        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        h:           [B, C, D]
        topk_idx:    [B, C, K]
        topk_scores: [B, C, K], signed
        """
        B, C, D = h.shape
        K = topk_idx.shape[-1]

        if K == 0:
            return torch.zeros_like(h)

        # Safety: avoid NaN / Inf entering softmax.
        topk_scores = torch.nan_to_num(
            topk_scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Gather neighbors.
        h_expand = h.unsqueeze(1).expand(-1, C, -1, -1)            # [B, C, C, D]
        idx_expand = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, C, K, D]
        neigh = torch.gather(h_expand, dim=2, index=idx_expand)    # [B, C, K, D]

        # Project neighbor features.
        neigh = self.v_proj(neigh)                                 # [B, C, K, D]

        # Use absolute score for attention strength,
        # and sign(score) for message direction.
        attn = torch.softmax(topk_scores.abs(), dim=-1).unsqueeze(-1)  # [B, C, K, 1]
        sign = topk_scores.sign().unsqueeze(-1)                        # [B, C, K, 1]

        attn = self.dropout(attn)

        # Signed aggregation:
        # positively correlated variables send positive messages;
        # negatively correlated variables send reverse messages.
        agg = (attn * sign * neigh).sum(dim=2)                     # [B, C, D]

        # Feature-wise gated residual message.
        gate = torch.sigmoid(self.gate_proj(torch.cat([h, agg], dim=-1)))
        delta = gate * self.out_proj(agg)

        # Final safety.
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        return delta
