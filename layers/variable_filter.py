import math
import torch
import torch.nn as nn


class VariableFilter(nn.Module):
    """
    Learnable target-specific variable filtering on trend representations.

    Input:
        h: [B, C, D]
    Output:
        topk_idx:    [B, C, K]
        topk_scores: [B, C, K]
        scores:      [B, C, C]
    """
    def __init__(self, d_model: int, topk: int, dropout: float = 0.0):
        super().__init__()
        self.topk = topk
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor):
        """
        h: [B, C, D]
        """
        B, C, D = h.shape

        q = self.q_proj(h)                              # [B, C, D]
        k = self.k_proj(h)                              # [B, C, D]

        scores = torch.matmul(q, k.transpose(-1, -2))   # [B, C, C]
        scores = scores / math.sqrt(D)

        # mask self-loop during candidate selection
        eye = torch.eye(C, device=h.device, dtype=torch.bool).unsqueeze(0)  # [1, C, C]
        scores = scores.masked_fill(eye, float('-inf'))

        # clamp topk to valid range
        k_val = min(self.topk, max(C - 1, 1))
        topk_scores, topk_idx = torch.topk(scores, k=k_val, dim=-1)         # [B, C, K], [B, C, K]
        topk_scores = self.dropout(topk_scores)

        return topk_idx, topk_scores, scores
