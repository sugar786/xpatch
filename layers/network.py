import torch
from torch import nn

from layers.variable_filter import VariableFilter
from layers.sparse_trend_interactor import SparseTrendInteractor


class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 use_trend_interactor=False, topk=4, interactor_dropout=0.0):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.use_trend_interactor = use_trend_interactor
        self.topk = topk

        # =========================
        # Non-linear Stream (Seasonal)
        # =========================
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1

        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # CNN Depthwise
        self.conv1 = nn.Conv1d(
            self.patch_num,
            self.patch_num,
            kernel_size=patch_len,
            stride=patch_len,
            groups=self.patch_num
        )
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, kernel_size=1, stride=1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # =========================
        # Linear Trend Stream
        # keep original xPatch trend encoder intact
        # =========================
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        # original trend prediction head
        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # =========================
        # Prediction-level Trend Correction
        # =========================
        if self.use_trend_interactor:
            # now we filter with raw trend + predicted trend
            self.variable_filter = VariableFilter(
                d_model=self.pred_len,
                topk=self.topk,
                dropout=interactor_dropout,
                learnable_weight=0.1,
                use_lag_corr=True,
                max_lag=3
            )
            self.trend_interactor = SparseTrendInteractor(
                d_model=self.pred_len,
                dropout=interactor_dropout
            )

        # =========================
        # Streams Concatenation
        # =========================
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def _seasonal_stream(self, s):
        """
        s: [B, C, L]
        return: [B, C, pred_len]
        """
        B, C, I = s.shape

        # channel split for channel independence
        s = torch.reshape(s, (B * C, I))  # [B*C, L]

        # patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [B*C, patch_num, patch_len]

        # patch embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # depthwise conv
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # residual stream
        res = self.fc2(res)
        s = s + res

        # pointwise conv
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # flatten head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        s = torch.reshape(s, (B, C, self.pred_len))  # [B, C, pred_len]
        return s

    def _trend_stream(self, t):
        """
        t: [B, C, L]
        return: [B, C, pred_len]
        """
        t_raw = t  # keep raw trend for statistical filtering

        # -------------------------
        # original xPatch trend encoder
        # -------------------------
        t = self.fc5(t)           # [B, C, pred_len*4]
        t = self.avgpool1(t)      # [B, C, pred_len*2]
        t = self.ln1(t)

        t = self.fc6(t)           # [B, C, pred_len]
        t = self.avgpool2(t)      # [B, C, pred_len//2]
        t = self.ln2(t)

        # original trend prediction
        t = self.fc7(t)           # [B, C, pred_len]

        # -------------------------
        # prediction-level sparse correction
        # -------------------------
        if self.use_trend_interactor:
            topk_idx, topk_scores, _ = self.variable_filter(t_raw, t)
            delta_t = self.trend_interactor(t, topk_idx, topk_scores)
            t = t + 0.1 * delta_t

        return t

    def forward(self, s, t):
        # s - seasonality: [B, L, C]
        # t - trend:       [B, L, C]

        # to [B, C, L]
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        # seasonal stream
        s_out = self._seasonal_stream(s)   # [B, C, pred_len]

        # trend stream
        t_out = self._trend_stream(t)      # [B, C, pred_len]

        # streams concatenation
        x = torch.cat((s_out, t_out), dim=-1)  # [B, C, pred_len*2]
        x = self.fc8(x)                        # [B, C, pred_len]

        # back to [B, pred_len, C]
        x = x.permute(0, 2, 1)

        return x
