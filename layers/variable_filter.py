import math
import torch
import torch.nn as nn


class VariableFilter(nn.Module):
    """
    Statistical-prior-guided variable filtering.

    Filtering is performed mainly from raw trend signals,
    while hidden states are reserved for downstream interaction.

    Inputs:
        t_raw:    [B, C, L]   raw trend sequence
        t_hidden: [B, C, D]   encoded trend representation

    Outputs:
        topk_idx:    [B, C, K]
        topk_scores: [B, C, K]
        scores:      [B, C, C]
    """
    def __init__(
        self,
        d_model: int,
        topk: int,
        dropout: float = 0.0,
        learnable_weight: float = 0.1,
        use_lag_corr: bool = True,
        max_lag: int = 3,
    ):
        super().__init__()
        self.topk = topk
        self.learnable_weight = learnable_weight
        self.use_lag_corr = use_lag_corr
        self.max_lag = max_lag

        # keep a weak learnable scorer as auxiliary signal
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _normalize_ts(x: torch.Tensor):
        # x: [B, C, L]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, unbiased=False, keepdim=True) + 1e-6
        return (x - mean) / std

    def _pearson_corr(self, x: torch.Tensor):
        # x: [B, C, L], already normalized
        L = x.size(-1)
        corr = torch.matmul(x, x.transpose(-1, -2)) / L   # [B, C, C]
        return corr

    def _lag_corr(self, x: torch.Tensor):
        # x: [B, C, L], already normalized
        B, C, L = x.shape
        corr_list = []

        for lag in range(1, self.max_lag + 1):
            x1 = x[..., lag:]         # [B, C, L-lag]
            x2 = x[..., :-lag]        # [B, C, L-lag]
            cur = torch.matmul(x1, x2.transpose(-1, -2)) / (L - lag)
            corr_list.append(cur)

        if len(corr_list) == 0:
            return torch.zeros(B, C, C, device=x.device)

        lag_corr = torch.stack(corr_list, dim=0).max(dim=0).values
        return lag_corr

    def forward(self, t_raw: torch.Tensor, t_hidden: torch.Tensor):
        """
        t_raw: [B, C, L]
        t_hidden: [B, C, D]
        """
        B, C, L = t_raw.shape
        D = t_hidden.shape[-1]

        # -------------------------
        # 1) statistical prior from raw trend
        # -------------------------
        x = self._normalize_ts(t_raw)
        corr_scores = self._pearson_corr(x)   # [B, C, C]

        if self.use_lag_corr:
            lag_scores = self._lag_corr(x)
            stat_scores = 0.5 * corr_scores + 0.5 * lag_scores
        else:
            stat_scores = corr_scores

        # -------------------------
        # 2) weak learnable auxiliary score
        # -------------------------
        q = self.q_proj(t_hidden)
        k = self.k_proj(t_hidden)
        learnable_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)

        # -------------------------
        # 3) fused score
        # main filtering depends on statistical priors
        # -------------------------
        scores = (1.0 - self.learnable_weight) * stat_scores + self.learnable_weight * learnable_scores

        # mask self-loop
        eye = torch.eye(C, device=t_raw.device, dtype=torch.bool).unsqueeze(0)
        scores = scores.masked_fill(eye, float('-inf'))

        # clamp topk to valid range
        k_val = min(self.topk, max(C - 1, 1))
        topk_scores, topk_idx = torch.topk(scores, k=k_val, dim=-1)
        topk_scores = self.dropout(topk_scores)

        return topk_idx, topk_scores, scores
