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
        ccm_trend_residual_weight=0.3,
        ccm_use_prototype=True,
        ccm_prob_mode="learned",
        use_dual_ccm=False,
    ):
        super(Network, self).__init__()

        self.pred_len = pred_len
        self.seq_len = seq_len

        self.use_ccm_head = use_ccm_head
        self.ccm_head_type = ccm_head_type
        self.ccm_residual_weight = ccm_residual_weight
        self.ccm_trend_residual_weight = ccm_trend_residual_weight
        self.use_dual_ccm = use_dual_ccm

        assert self.ccm_head_type in ["seasonal", "trend", "both"], \
            "ccm_head_type must be one of ['seasonal', 'trend', 'both']"

        # =========================
        # Non-linear Stream, Seasonal
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
        # CCM assigners and heads
        # =========================
        if self.use_ccm_head:
            if self.use_dual_ccm:
                if self.ccm_head_type in ["seasonal", "both"]:
                    self.seasonal_cluster_assigner = CCMClusterAssigner(
                        seq_len=seq_len,
                        n_cluster=n_cluster,
                        d_model=ccm_d_model,
                        sigma=ccm_sigma,
                        epsilon=ccm_epsilon,
                        gumbel_temp=ccm_gumbel_temp,
                        use_gumbel=ccm_use_gumbel,
                        dropout=ccm_dropout,
                        use_prototype=ccm_use_prototype,
                        prob_mode=ccm_prob_mode,
                    )

                if self.ccm_head_type in ["trend", "both"]:
                    self.trend_cluster_assigner = CCMClusterAssigner(
                        seq_len=seq_len,
                        n_cluster=n_cluster,
                        d_model=ccm_d_model,
                        sigma=ccm_sigma,
                        epsilon=ccm_epsilon,
                        gumbel_temp=ccm_gumbel_temp,
                        use_gumbel=ccm_use_gumbel,
                        dropout=ccm_dropout,
                        use_prototype=ccm_use_prototype,
                        prob_mode=ccm_prob_mode,
                    )
            else:
                # backward-compatible single CCM
                self.cluster_assigner = CCMClusterAssigner(
                    seq_len=seq_len,
                    n_cluster=n_cluster,
                    d_model=ccm_d_model,
                    sigma=ccm_sigma,
                    epsilon=ccm_epsilon,
                    gumbel_temp=ccm_gumbel_temp,
                    use_gumbel=ccm_use_gumbel,
                    dropout=ccm_dropout,
                    use_prototype=ccm_use_prototype,
                    prob_mode=ccm_prob_mode,
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

        # Backward-compatible single CCM cache.
        self.ccm_prob = None
        self.ccm_membership = None
        self.ccm_sim_matrix = None
        self.ccm_channel_emb = None
        self.ccm_cluster_emb = None

        # Dual CCM cache.
        self.ccm_prob_s = None
        self.ccm_membership_s = None
        self.ccm_sim_matrix_s = None
        self.ccm_channel_emb_s = None
        self.ccm_cluster_emb_s = None

        self.ccm_prob_t = None
        self.ccm_membership_t = None
        self.ccm_sim_matrix_t = None
        self.ccm_channel_emb_t = None
        self.ccm_cluster_emb_t = None

    def _reset_ccm_cache(self):
        self.ccm_prob = None
        self.ccm_membership = None
        self.ccm_sim_matrix = None
        self.ccm_channel_emb = None
        self.ccm_cluster_emb = None

        self.ccm_prob_s = None
        self.ccm_membership_s = None
        self.ccm_sim_matrix_s = None
        self.ccm_channel_emb_s = None
        self.ccm_cluster_emb_s = None

        self.ccm_prob_t = None
        self.ccm_membership_t = None
        self.ccm_sim_matrix_t = None
        self.ccm_channel_emb_t = None
        self.ccm_cluster_emb_t = None

    def _seasonal_backbone(self, s):
        """
        Args:
            s: [B, C, L]

        Returns:
            feat: [B, C, patch_num * patch_len]
        """
        B, C, I = s.shape

        s = torch.reshape(s, (B * C, I))  # [B*C, L]

        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)

        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        res = self.fc2(res)
        s = s + res

        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        s = self.flatten1(s)
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

        t_base = self.fc7(hidden)

        if self.use_ccm_head and self.ccm_head_type in ["trend", "both"]:
            t_ccm = self.ccm_trend_head(hidden, prob)
            t_out = t_base + self.ccm_trend_residual_weight * (t_ccm - t_base)
        else:
            t_out = t_base

        return t_out

    def _compute_single_ccm_prob(self, x_raw):
        """
        Backward-compatible single CCM mode.

        Args:
            x_raw: [B, L, C]

        Returns:
            prob_for_head: [B, C, K]
        """
        prob, membership, sim_matrix, channel_emb, cluster_emb = self.cluster_assigner(x_raw)

        self.ccm_prob = prob
        self.ccm_membership = membership
        self.ccm_sim_matrix = sim_matrix
        self.ccm_channel_emb = channel_emb
        self.ccm_cluster_emb = cluster_emb

        # Stabilize cluster identity for fixed-channel datasets.
        prob_for_head = prob.mean(dim=0, keepdim=True).expand_as(prob)

        return prob_for_head

    def _compute_dual_ccm_prob(self, seasonal_raw, trend_raw):
        """
        Dual CCM mode.

        Args:
            seasonal_raw: seasonal_init, [B, L, C]
            trend_raw: trend_init, [B, L, C]

        Returns:
            prob_s_for_head: [B, C, K]
            prob_t_for_head: [B, C, K]
        """
        prob_s = None
        prob_t = None

        if self.ccm_head_type in ["seasonal", "both"]:
            prob_s, membership_s, sim_matrix_s, channel_emb_s, cluster_emb_s = \
                self.seasonal_cluster_assigner(seasonal_raw)

            self.ccm_prob_s = prob_s
            self.ccm_membership_s = membership_s
            self.ccm_sim_matrix_s = sim_matrix_s
            self.ccm_channel_emb_s = channel_emb_s
            self.ccm_cluster_emb_s = cluster_emb_s

            prob_s = prob_s.mean(dim=0, keepdim=True).expand_as(prob_s)

        if self.ccm_head_type in ["trend", "both"]:
            prob_t, membership_t, sim_matrix_t, channel_emb_t, cluster_emb_t = \
                self.trend_cluster_assigner(trend_raw)

            self.ccm_prob_t = prob_t
            self.ccm_membership_t = membership_t
            self.ccm_sim_matrix_t = sim_matrix_t
            self.ccm_channel_emb_t = channel_emb_t
            self.ccm_cluster_emb_t = cluster_emb_t

            prob_t = prob_t.mean(dim=0, keepdim=True).expand_as(prob_t)

        return prob_s, prob_t

    def forward(self, s, t, x_raw=None):
        """
        Args:
            s: seasonality input, [B, L, C]
            t: trend input,       [B, L, C]
            x_raw: input for single CCM mode, [B, L, C]

        Returns:
            x: prediction, [B, pred_len, C]
        """

        self._reset_ccm_cache()

        if x_raw is None:
            x_raw = s + t

        prob_s_for_head = None
        prob_t_for_head = None

        if self.use_ccm_head:
            if self.use_dual_ccm:
                # Dual-CCM:
                # seasonal_init -> P_s -> seasonal head
                # trend_init    -> P_t -> trend head
                prob_s_for_head, prob_t_for_head = self._compute_dual_ccm_prob(
                    seasonal_raw=s,
                    trend_raw=t,
                )
            else:
                # Single-CCM backward-compatible mode.
                prob_for_head = self._compute_single_ccm_prob(x_raw)
                prob_s_for_head = prob_for_head
                prob_t_for_head = prob_for_head

        # [B, L, C] -> [B, C, L]
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        s_out = self._seasonal_stream(s, prob=prob_s_for_head)
        t_out = self._trend_stream(t, prob=prob_t_for_head)

        # Original xPatch fusion
        x = torch.cat((s_out, t_out), dim=-1)
        x = self.fc8(x)

        # [B, C, pred_len] -> [B, pred_len, C]
        x = x.permute(0, 2, 1)

        return x
