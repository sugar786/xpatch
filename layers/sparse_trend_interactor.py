import torch
import torch.nn as nn


class SparseTrendInteractor(nn.Module):
    """
    Sparse cross-variable interaction on filtered trend representations.

    Input:
        h:           [B, C, D]
        topk_idx:    [B, C, K]
        topk_scores: [B, C, K]

    Output:
        out:         [B, C, D]
    """
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, topk_idx: torch.Tensor, topk_scores: torch.Tensor):
        """
        h: [B, C, D]
        topk_idx: [B, C, K]
        topk_scores: [B, C, K]
        """
        B, C, D = h.shape
        K = topk_idx.shape[-1]

        # gather neighbors
        h_expand = h.unsqueeze(1).expand(-1, C, -1, -1)                       # [B, C, C, D]
        idx_expand = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)             # [B, C, K, D]
        neigh = torch.gather(h_expand, dim=2, index=idx_expand)               # [B, C, K, D]

        # attention within filtered candidate set
        attn = torch.softmax(topk_scores, dim=-1).unsqueeze(-1)               # [B, C, K, 1]
        attn = self.dropout(attn)

        neigh = self.v_proj(neigh)                                             # [B, C, K, D]
        agg = (attn * neigh).sum(dim=2)                                        # [B, C, D]

        gate = torch.sigmoid(self.gate_proj(torch.cat([h, agg], dim=-1)))     # [B, C, D]
        delta = gate * self.out_proj(agg)
        
        return delta
