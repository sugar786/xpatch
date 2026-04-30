import torch
from torch import nn

from layers.variable_filter import VariableFilter
from layers.sparse_trend_interactor import SparseTrendInteractor
from layers.seasonal_adaptive_variable_mixer import SeasonalAdaptiveVariableMixer


class Network(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        patch_len,
        stride,
        padding_patch,
        c_in,
        use_trend_interactor=False,
        use_seasonal_interactor=False,
        topk=4,
        interactor_dropout=0.0,
    ):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.c_in = c_in

        self.use_trend_interactor = use_trend_interactor
        self.use_seasonal_interactor = use_seasonal_interactor
        self.topk = topk

        # =========================
        # Non-linear Stream (Seasonal / Residual)
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

        # Seasonal / residual variable-aware calibration.
        # Inserted after patch embedding and before depthwise conv.
        if self.use_seasonal_interactor:
            self.seasonal_interactor = SeasonalAdaptiveVariableMixer(
                c_in=c_in,
                d_model=self.dim,
                node_dim=16,
                topk=self.topk,
                dropout=interactor_dropout,
            )

            # Zero-initialized residual gate.
            # The effective scale is 0.1 * tanh(seasonal_scale),
            # so the maximum absolute injection strength is limited to 0.1.
            self.seasonal_scale = nn.Parameter(torch.zeros(1))

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
        self.conv2 = nn.Conv1d(
            self.patch_num,
            self.patch_num,
            kernel_size=1,
            stride=1,
        )
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
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        # Original trend prediction head
        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # =========================
        # Trend Variable Interaction
        # Keep your previous trend branch module.
        # =========================
        if self.use_trend_interactor:
            self.variable_filter = VariableFilter(
                d_model=self.pred_len,
                topk=self.topk,
                dropout=interactor_dropout,
                learnable_weight=0.1,
                use_lag_corr=True,
                max_lag=3,
            )
            self.trend_interactor = SparseTrendInteractor(
                d_model=self.pred_len,
                dropout=interactor_dropout,
            )

            # Safer than fixed 0.1 correction.
            self.trend_scale = nn.Parameter(torch.zeros(1))

        # =========================
        # Streams Concatenation
        # =========================
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def _seasonal_stream(self, s):
        """
        Args:
            s: [B, C, L]

        Returns:
            s: [B, C, pred_len]
        """
        B, C, I = s.shape

        # Channel split for channel independence.
        s = torch.reshape(s, (B * C, I))  # [B*C, L]

        # Patching
        if self.padding_patch == "end":
            s = self.padding_patch_layer(s)

        s = s.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride,
        )
        # [B*C, patch_num, patch_len]

        # Patch embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)
        # [B*C, patch_num, dim]

        # =========================
        # Seasonal / residual variable-aware calibration
        # =========================
        if self.use_seasonal_interactor:
            s_4d = s.reshape(B, C, self.patch_num, self.dim)
            delta_s = self.seasonal_interactor(s_4d)

            # Conservative injection:
            # limit the maximum effective residual strength to 0.1.
            seasonal_scale = 0.1 * torch.tanh(self.seasonal_scale)
            s_4d = s_4d + seasonal_scale * delta_s

            s = s_4d.reshape(B * C, self.patch_num, self.dim)

        res = s

        # Depthwise conv
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual stream
        res = self.fc2(res)
        s = s + res

        # Pointwise conv
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        s = torch.reshape(s, (B, C, self.pred_len))  # [B, C, pred_len]

        return s

    def _trend_stream(self, t):
        """
        Args:
            t: [B, C, L]

        Returns:
            t: [B, C, pred_len]
        """
        t_raw = t

        # Original xPatch trend encoder
        t = self.fc5(t)           # [B, C, pred_len*4]
        t = self.avgpool1(t)      # [B, C, pred_len*2]
        t = self.ln1(t)

        t = self.fc6(t)           # [B, C, pred_len]
        t = self.avgpool2(t)      # [B, C, pred_len//2]
        t = self.ln2(t)

        # Original trend prediction
        t = self.fc7(t)           # [B, C, pred_len]

        # Trend variable correction
        if self.use_trend_interactor:
            topk_idx, topk_scores, _ = self.variable_filter(t_raw, t)
            delta_t = self.trend_interactor(t, topk_idx, topk_scores)

            trend_scale = torch.tanh(self.trend_scale)
            t = t + trend_scale * delta_t

        return t

    def forward(self, s, t):
        """
        Args:
            s: seasonal / residual input, [B, L, C]
            t: trend input, [B, L, C]

        Returns:
            x: [B, pred_len, C]
        """
        # To [B, C, L]
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        # Seasonal / residual stream
        s_out = self._seasonal_stream(s)  # [B, C, pred_len]

        # Trend stream
        t_out = self._trend_stream(t)     # [B, C, pred_len]

        # Streams concatenation
        x = torch.cat((s_out, t_out), dim=-1)  # [B, C, pred_len*2]
        x = self.fc8(x)                        # [B, C, pred_len]

        # Back to [B, pred_len, C]
        x = x.permute(0, 2, 1)

        return x
