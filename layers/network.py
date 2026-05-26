import torch
from torch import nn

from layers.ccm_xpatch import CCMClusterAssigner, ClusterAwareSeasonalHead, ClusterAwareLinear


class Network(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        patch_len,
        stride,
        padding_patch,
        use_ccm_head=False,
        ccm_head_type="seasonal",
        n_cluster=2,
        ccm_d_model=32,
        ccm_sigma=5.0,
        ccm_epsilon=0.2,
        ccm_gumbel_temp=0.5,
        ccm_use_gumbel=False,
        ccm_dropout=0.0,
        ccm_residual_weight=0.5,
    ):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len

        self.use_ccm_head = use_ccm_head
        self.ccm_head_type = ccm_head_type
        self.ccm_residual_weight = ccm_residual_weight

        assert self.ccm_head_type in ["seasonal", "trend", "both"], \
            "ccm_head_type must be one of ['seasonal', 'trend', 'both']"

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
        self.conv2 = nn.Conv1d(
            self.patch_num,
            self.patch_num,
            kernel_size=1,
            stride=1
        )
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head, original xPatch
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

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # =========================
        # CCM pre-temporal cluster assigner
        # =========================
        if self.use_ccm_head:
            self.cluster_assigner = CCMClusterAssigner(
                seq_len=seq_len,
                n_cluster=n_cluster,
                d_model=ccm_d_model,
                sigma=ccm_sigma,
                epsilon=ccm_epsilon,
                gumbel_temp=ccm_gumbel_temp,
                use_gumbel=ccm_use_gumbel,
                dropout=ccm_dropout,
            )

            if self.ccm_head_type in ["seasonal", "both"]:
                self.ccm_seasonal_head = ClusterAwareSeasonalHead(
                    in_dim=self.patch_num * patch_len,
                    pred_len=pred_len,
                    n_cluster=n_cluster,
                    dropout=ccm_dropout,
                )

            if self.ccm_head_type in ["trend", "both"]:
                self.ccm_trend_head = ClusterAwareLinear(
                    in_dim=pred_len // 2,
                    out_dim=pred_len,
                    n_cluster=n_cluster,
                    dropout=ccm_dropout,
                )

        # =========================
        # Streams Concatenation
        # =========================
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

        # Store CCM outputs for optional loss/debug.
        self.ccm_prob = None
        self.ccm_membership = None
        self.ccm_sim_matrix = None

    def _seasonal_backbone(self, s):
        """
        Args:
            s: [B, C, L]

        Returns:
            feat: [B, C, patch_num * patch_len]
        """
        B, C, I = s.shape

        # Channel-independent seasonal modeling
        s = torch.reshape(s, (B * C, I))  # [B*C, L]

        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)

        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # [B*C, patch_num, patch_len]

        # Patch embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # Depthwise convolution
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual stream
        res = self.fc2(res)
        s = s + res

        # Pointwise convolution
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten feature
        s = self.flatten1(s)  # [B*C, patch_num * patch_len]
        s = torch.reshape(s, (B, C, self.patch_num * self.patch_len))

        return s

    def _seasonal_stream(self, s, prob=None):
        """
        Args:
            s: [B, C, L]
            prob: [B, C, K] or None

        Returns:
            s_out: [B, C, pred_len]
        """
        feat = self._seasonal_backbone(s)

        s_base = self.fc3(feat)
        s_base = self.gelu4(s_base)
        s_base = self.fc4(s_base)

        if self.use_ccm_head and self.ccm_head_type in ["seasonal", "both"]:
            s_ccm = self.ccm_seasonal_head(feat, prob)
            s_out = s_base + self.ccm_residual_weight * (s_ccm - s_base)
        else:
            s_out = s_base

        return s_out

    def _trend_backbone(self, t):
        """
        Args:
            t: [B, C, L]

        Returns:
            hidden: [B, C, pred_len // 2]
        """
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        return t

    def _trend_stream(self, t, prob=None):
        """
        Args:
            t: [B, C, L]
            prob: [B, C, K] or None

        Returns:
            t_out: [B, C, pred_len]
        """
        hidden = self._trend_backbone(t)

        if self.use_ccm_head and self.ccm_head_type in ["trend", "both"]:
            t_out = self.ccm_trend_head(hidden, prob)
        else:
            t_out = self.fc7(hidden)

        return t_out

    def forward(self, s, t, x_raw=None):
        """
        Args:
            s: seasonality input, [B, L, C]
            t: trend input,       [B, L, C]
            x_raw: normalized raw input for CCM clustering, [B, L, C]

        Returns:
            x: prediction, [B, pred_len, C]
        """

        # reset cached CCM tensors
        self.ccm_prob = None
        self.ccm_membership = None
        self.ccm_sim_matrix = None

        if x_raw is None:
            x_raw = s + t

        prob = None

        # CCM cluster assignment is computed before temporal modules.
        if self.use_ccm_head:
            prob, membership, sim_matrix, _, _ = self.cluster_assigner(x_raw)

            # Cache original probability for diagnosis.
            self.ccm_prob = prob
            self.ccm_membership = membership
            self.ccm_sim_matrix = sim_matrix

            # Stabilize cluster identity for fixed-channel multivariate datasets.
            prob_for_head = prob.mean(dim=0, keepdim=True).expand_as(prob)
        else:
            prob_for_head = None

        # [B, L, C] -> [B, C, L]
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        s_out = self._seasonal_stream(s, prob=prob_for_head)
        t_out = self._trend_stream(t, prob=prob_for_head)
        
        # Original xPatch fusion
        x = torch.cat((s_out, t_out), dim=-1)
        x = self.fc8(x)

        # [B, C, pred_len] -> [B, pred_len, C]
        x = x.permute(0, 2, 1)

        return x
