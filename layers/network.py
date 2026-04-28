import torch
from torch import nn

from layers.variable_filter import VariableFilter
from layers.sparse_trend_interactor import SparseTrendInteractor


class Network(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        patch_len,
        stride,
        padding_patch,
        use_trend_interactor=False,
        topk=4,
        interactor_dropout=0.0,
    ):
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

        if padding_patch == "end":
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
            groups=self.patch_num,
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
        # =========================
        # stage 1: trend encoder ([B, C, L] -> [B, C, D])
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        # stage 2: trend head ([B, C, D] -> [B, C, pred_len])
        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # =========================
        # Prediction-level variable correction
        # =========================
        # The correction is applied after seasonal and trend streams have
        # produced prediction-level features, instead of inside trend hidden states.
        if self.use_trend_interactor:
            self.pred_interactor_dim = pred_len * 2

            self.variable_filter = VariableFilter(
                d_model=self.pred_interactor_dim,
                topk=self.topk,
                dropout=interactor_dropout,
                learnable_weight=0.1,
                use_lag_corr=True,
                max_lag=3,
            )
            self.pred_interactor = SparseTrendInteractor(
                d_model=self.pred_interactor_dim,
                dropout=interactor_dropout,
            )

            # Initialize to roughly 0.1, matching the previous fixed residual strength,
            # but keep it learnable.
            # sigmoid(-2.2) ~= 0.10.
            self.interactor_alpha = nn.Parameter(torch.tensor(-2.2))

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

        # Channel split for channel independence on seasonal stream.
        s = torch.reshape(s, (B * C, I))  # [B*C, L]

        # Patching.
        if self.padding_patch == "end":
            s = self.padding_patch_layer(s)

        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [B*C, patch_num, patch_len]

        # Patch embedding.
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # Depthwise conv.
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual stream.
        res = self.fc2(res)
        s = s + res

        # Pointwise conv.
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten head.
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
        # stage 1: trend encoder
        t = self.fc5(t)       # [B, C, pred_len*4]
        t = self.avgpool1(t)  # [B, C, pred_len*2]
        t = self.ln1(t)

        t = self.fc6(t)       # [B, C, pred_len]
        t = self.avgpool2(t)  # [B, C, pred_len//2]
        t = self.ln2(t)       # [B, C, pred_len//2]

        # stage 2: trend head
        t = self.fc7(t)       # [B, C, pred_len]
        return t

    def forward(self, s, t):
        """
        s - seasonality: [B, L, C]
        t - trend:       [B, L, C]
        """

        # to [B, C, L]
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        # Save raw trend for statistical filtering.
        # Shape: [B, C, L]
        t_raw = t

        # seasonal stream
        s_out = self._seasonal_stream(s)  # [B, C, pred_len]

        # trend stream
        t_out = self._trend_stream(t)     # [B, C, pred_len]

        # Prediction-level feature before final projection.
        x_feat = torch.cat((s_out, t_out), dim=-1)  # [B, C, pred_len*2]

        # Prediction-level sparse variable correction.
        if self.use_trend_interactor:
            topk_idx, topk_scores, _ = self.variable_filter(t_raw, x_feat)
            delta = self.pred_interactor(x_feat, topk_idx, topk_scores)

            alpha = torch.sigmoid(self.interactor_alpha)
            x_feat = x_feat + alpha * delta

        # final prediction projection
        x = self.fc8(x_feat)  # [B, C, pred_len]

        # back to [B, pred_len, C]
        x = x.permute(0, 2, 1)

        return x
