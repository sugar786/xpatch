import math
import torch
import torch.nn as nn


class VariableFilter(nn.Module):
    """
    Statistical-prior-guided sparse variable filtering.

    Key design:
    1. Use absolute correlation magnitude for top-k neighbor selection,
       so strong negative correlations will not be ignored.
    2. Return signed finite scores for downstream modules.
       The interactor may choose to use score magnitude only or signed score.
    3. Keep all returned scores finite to avoid NaN in softmax.

    Inputs:
        t_raw:    [B, C, L]   raw sequence used for statistical filtering
        h:        [B, C, D]   representation used for learnable auxiliary score

    Outputs:
        topk_idx:           [B, C, K]
        topk_signed_scores: [B, C, K]
        scores:             [B, C, C] signed finite fused scores
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
        self.use_lag_corr = use_lag_corr
        self.max_lag = max_lag

        # Weak learnable scorer as auxiliary signal.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        # Learnable fusion weight.
        # Initialized so sigmoid(score_gate) ~= learnable_weight.
        learnable_weight = float(learnable_weight)
        learnable_weight = min(max(learnable_weight, 1e-4), 1.0 - 1e-4)
        init_logit = math.log(learnable_weight / (1.0 - learnable_weight))
        self.score_gate = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _normalize_ts(x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    @staticmethod
    def _make_eye_mask(C: int, device) -> torch.Tensor:
        """
        return: [1, C, C]
        """
        return torch.eye(C, device=device, dtype=torch.bool).unsqueeze(0)

    def _pearson_corr(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L], already normalized
        return: [B, C, C]
        """
        L = x.size(-1)
        corr = torch.matmul(x, x.transpose(-1, -2)) / max(L, 1)
        return corr

    def _lag_corr(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L], already normalized

        For multiple lags, select the lag with the largest absolute correlation,
        while preserving the original sign.
        """
        B, C, L = x.shape
        corr_list = []

        max_valid_lag = min(self.max_lag, max(L - 1, 0))
        for lag in range(1, max_valid_lag + 1):
            x1 = x[..., lag:]      # [B, C, L-lag]
            x2 = x[..., :-lag]     # [B, C, L-lag]
            cur = torch.matmul(x1, x2.transpose(-1, -2)) / max(L - lag, 1)
            corr_list.append(cur)

        if len(corr_list) == 0:
            return torch.zeros(B, C, C, device=x.device, dtype=x.dtype)

        lag_stack = torch.stack(corr_list, dim=0)  # [num_lag, B, C, C]
        best_lag_idx = lag_stack.abs().argmax(dim=0, keepdim=True)
        lag_corr = torch.gather(lag_stack, dim=0, index=best_lag_idx).squeeze(0)
        return lag_corr

    def forward(self, t_raw: torch.Tensor, h: torch.Tensor):
        """
        t_raw: [B, C, L]
        h:     [B, C, D]
        """
        B, C, _ = t_raw.shape
        D = h.shape[-1]

        # -------------------------
        # 1) Statistical prior from raw sequence
        # -------------------------
        x = self._normalize_ts(t_raw)
        corr_scores = self._pearson_corr(x)  # [B, C, C]

        if self.use_lag_corr:
            lag_scores = self._lag_corr(x)
            stat_scores = 0.5 * corr_scores + 0.5 * lag_scores
        else:
            stat_scores = corr_scores

        # -------------------------
        # 2) Weak learnable auxiliary score
        # -------------------------
        q = self.q_proj(h)
        k = self.k_proj(h)
        learnable_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)

        # -------------------------
        # 3) Fused signed score
        # -------------------------
        w = torch.sigmoid(self.score_gate)
        scores = (1.0 - w) * stat_scores + w * learnable_scores

        # Safety: remove possible numerical abnormal values before top-k.
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        # If C == 1, there is no cross-variable neighbor.
        if C <= 1:
            empty_idx = torch.empty(B, C, 0, device=t_raw.device, dtype=torch.long)
            empty_scores = torch.empty(B, C, 0, device=t_raw.device, dtype=t_raw.dtype)
            return empty_idx, empty_scores, scores

        # -------------------------
        # 4) Top-k selection
        # -------------------------
        eye = self._make_eye_mask(C, t_raw.device)

        # Important:
        # Do NOT set scores diagonal to -inf before abs(),
        # because abs(-inf) = inf and self-loop will be selected.
        select_scores = scores.abs()
        select_scores = select_scores.masked_fill(eye, float("-inf"))

        # Signed scores must stay finite. Diagonal is set to 0.
        signed_scores = scores.masked_fill(eye, 0.0)

        k_val = min(self.topk, C - 1)

        _, topk_idx = torch.topk(select_scores, k=k_val, dim=-1)

        # Return signed scores for downstream interaction.
        topk_signed_scores = torch.gather(signed_scores, dim=-1, index=topk_idx)
        topk_signed_scores = torch.nan_to_num(
            topk_signed_scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        topk_signed_scores = self.dropout(topk_signed_scores)

        # Return finite signed matrix for debugging / visualization.
        scores = signed_scores

        return topk_idx, topk_signed_scores, scores
