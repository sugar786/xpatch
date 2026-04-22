import math
import torch
import torch.nn as nn


class VariableFilter(nn.Module):
    """
    Similarity-guided target-specific variable filtering on trend representations.

    Input:
        h: [B, C, D]
    Output:
        topk_idx:    [B, C, K]
        topk_scores: [B, C, K]
        scores:      [B, C, C]
    """
    def __init__(
        self,
        d_model: int,
        topk: int,
        dropout: float = 0.0,
        sim_sigma: float = 1.0,
        learnable_weight: float = 0.3,
    ):
        super().__init__()
        self.topk = topk
        self.sim_sigma = sim_sigma
        self.learnable_weight = learnable_weight

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor):
        """
        h: [B, C, D]
        """
        B, C, D = h.shape

        # -------------------------
        # 1) learnable score
        # -------------------------
        q = self.q_proj(h)                                 # [B, C, D]
        k = self.k_proj(h)                                 # [B, C, D]
        learnable_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)  # [B, C, C]

        # -------------------------
        # 2) similarity prior score
        # use RBF similarity on hidden trend representations
        # -------------------------
        dist = torch.cdist(h, h, p=2)                      # [B, C, C]
        sim_scores = torch.exp(-(dist ** 2) / (2 * (self.sim_sigma ** 2) + 1e-8))

        # convert similarity to logit-like space for easier fusion
        sim_scores = torch.log(sim_scores + 1e-8)

        # -------------------------
        # 3) fused score
        # -------------------------
        scores = self.learnable_weight * learnable_scores + (1.0 - self.learnable_weight) * sim_scores

        # mask self-loop
        eye = torch.eye(C, device=h.device, dtype=torch.bool).unsqueeze(0)  # [1, C, C]
        scores = scores.masked_fill(eye, float('-inf'))

        # clamp topk to valid range
        k_val = min(self.topk, max(C - 1, 1))
        topk_scores, topk_idx = torch.topk(scores, k=k_val, dim=-1)         # [B, C, K]
        topk_scores = self.dropout(topk_scores)

        return topk_idx, topk_scores, scores
