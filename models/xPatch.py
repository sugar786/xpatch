import torch
import torch.nn as nn

from layers.decomp import DECOMP
from layers.network import Network
from layers.revin import RevIN
from layers.ccm_xpatch import CCMClusterLoss


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs

        self.aux_loss = None
        self.raw_cluster_loss = None
        self.raw_cluster_loss_s = None
        self.raw_cluster_loss_t = None

        # Parameters
        seq_len = configs.seq_len
        pred_len = configs.pred_len
        c_in = configs.enc_in

        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        # Moving Average / decomposition
        self.ma_type = configs.ma_type
        alpha = configs.alpha
        beta = configs.beta
        self.decomp = DECOMP(self.ma_type, alpha, beta)

        # CCM config
        self.use_ccm_head = getattr(configs, "use_ccm_head", False)
        self.use_dual_ccm = getattr(configs, "use_dual_ccm", False)

        self.ccm_loss_weight = getattr(configs, "ccm_loss_weight", 0.0)
        self.ccm_trend_loss_weight = getattr(configs, "ccm_trend_loss_weight", self.ccm_loss_weight)

        # Backward-compatible single-CCM input type.
        self.ccm_input_type = getattr(configs, "ccm_input_type", "raw")

        assert self.ccm_input_type in [
            "raw",
            "seasonal",
            "trend",
            "raw_plus_seasonal",
            "seasonal_plus_trend",
        ], (
            "ccm_input_type must be one of "
            "['raw', 'seasonal', 'trend', 'raw_plus_seasonal', 'seasonal_plus_trend']"
        )

        # xPatch backbone with optional CCM-aware prediction heads.
        self.net = Network(
            seq_len,
            pred_len,
            patch_len,
            stride,
            padding_patch,
            use_ccm_head=self.use_ccm_head,
            ccm_head_type=getattr(configs, "ccm_head_type", "seasonal"),
            n_cluster=getattr(configs, "n_cluster", 2),
            ccm_d_model=getattr(configs, "ccm_d_model", 32),
            ccm_sigma=getattr(configs, "ccm_sigma", 5.0),
            ccm_epsilon=getattr(configs, "ccm_epsilon", 0.2),
            ccm_gumbel_temp=getattr(configs, "ccm_gumbel_temp", 0.5),
            ccm_use_gumbel=getattr(configs, "ccm_use_gumbel", False),
            ccm_dropout=getattr(configs, "ccm_dropout", 0.0),
            ccm_residual_weight=getattr(configs, "ccm_residual_weight", 0.5),
            ccm_trend_residual_weight=getattr(configs, "ccm_trend_residual_weight", 0.3),
            ccm_use_prototype=getattr(configs, "ccm_use_prototype", True),
            ccm_prob_mode=getattr(configs, "ccm_prob_mode", "learned"),
            use_dual_ccm=self.use_dual_ccm,
        )

        if self.use_ccm_head and (self.ccm_loss_weight > 0 or self.ccm_trend_loss_weight > 0):
            self.ccm_loss_fn = CCMClusterLoss()
        else:
            self.ccm_loss_fn = None

    def get_aux_loss(self):
        """
        Called by exp_main.py after forward.
        """
        if self.aux_loss is None:
            return 0.0
        return self.aux_loss

    def get_raw_cluster_loss(self):
        """
        Backward-compatible total raw cluster loss.
        """
        if self.raw_cluster_loss is None:
            return 0.0
        return self.raw_cluster_loss

    def get_raw_cluster_loss_s(self):
        if self.raw_cluster_loss_s is None:
            return 0.0
        return self.raw_cluster_loss_s

    def get_raw_cluster_loss_t(self):
        if self.raw_cluster_loss_t is None:
            return 0.0
        return self.raw_cluster_loss_t

    def _select_single_ccm_input(self, x_raw, seasonal_init=None, trend_init=None):
        """
        Select the input used by single CCM cluster assigner.
        Only used when use_dual_ccm=False.
        """

        if self.ccm_input_type == "raw":
            return x_raw

        if self.ccm_input_type == "seasonal":
            if seasonal_init is None:
                return x_raw
            return seasonal_init

        if self.ccm_input_type == "trend":
            if trend_init is None:
                return x_raw
            return trend_init

        if self.ccm_input_type == "raw_plus_seasonal":
            if seasonal_init is None:
                return x_raw
            return x_raw + seasonal_init

        if self.ccm_input_type == "seasonal_plus_trend":
            if seasonal_init is None or trend_init is None:
                return x_raw
            return seasonal_init + trend_init

        raise ValueError("Unsupported ccm_input_type: {}".format(self.ccm_input_type))

    def _compute_ccm_losses(self):
        """
        Compute CCM auxiliary loss.

        Single CCM mode:
            aux = ccm_loss_weight * L_single

        Dual CCM mode:
            aux = ccm_loss_weight * L_s + ccm_trend_loss_weight * L_t
        """

        self.raw_cluster_loss = None
        self.raw_cluster_loss_s = None
        self.raw_cluster_loss_t = None
        self.aux_loss = None

        if not self.use_ccm_head or self.ccm_loss_fn is None:
            return

        aux_loss = 0.0
        raw_total = 0.0
        has_loss = False

        if self.use_dual_ccm:
            # Seasonal cluster loss
            if (
                self.ccm_loss_weight > 0
                and self.net.ccm_sim_matrix_s is not None
                and self.net.ccm_membership_s is not None
            ):
                loss_s = self.ccm_loss_fn(
                    self.net.ccm_sim_matrix_s,
                    self.net.ccm_membership_s,
                )
                self.raw_cluster_loss_s = loss_s
                aux_loss = aux_loss + self.ccm_loss_weight * loss_s
                raw_total = raw_total + loss_s
                has_loss = True

            # Trend cluster loss
            if (
                self.ccm_trend_loss_weight > 0
                and self.net.ccm_sim_matrix_t is not None
                and self.net.ccm_membership_t is not None
            ):
                loss_t = self.ccm_loss_fn(
                    self.net.ccm_sim_matrix_t,
                    self.net.ccm_membership_t,
                )
                self.raw_cluster_loss_t = loss_t
                aux_loss = aux_loss + self.ccm_trend_loss_weight * loss_t
                raw_total = raw_total + loss_t
                has_loss = True

            if has_loss:
                self.raw_cluster_loss = raw_total
                self.aux_loss = aux_loss

        else:
            # Backward-compatible single CCM loss
            if (
                self.ccm_loss_weight > 0
                and self.net.ccm_sim_matrix is not None
                and self.net.ccm_membership is not None
            ):
                loss = self.ccm_loss_fn(
                    self.net.ccm_sim_matrix,
                    self.net.ccm_membership,
                )
                self.raw_cluster_loss = loss
                self.aux_loss = self.ccm_loss_weight * loss

    def forward(self, x):
        """
        Args:
            x: [B, L, C]

        Returns:
            x: [B, pred_len, C]
        """

        self.aux_loss = None
        self.raw_cluster_loss = None
        self.raw_cluster_loss_s = None
        self.raw_cluster_loss_t = None

        # RevIN normalization
        if self.revin:
            x = self.revin_layer(x, "norm")

        # RevIN-normalized original input.
        x_raw = x

        # xPatch backbone
        if self.ma_type == "reg":
            # No decomposition. Dual-CCM degenerates to raw/raw.
            x = self.net(x, x, x_raw=x_raw)

        else:
            seasonal_init, trend_init = self.decomp(x)

            if self.use_dual_ccm:
                # Dual CCM:
                # seasonal_init -> CCM_s -> seasonal head
                # trend_init    -> CCM_t -> trend head
                #
                # Network.forward will internally pass s to seasonal assigner
                # and t to trend assigner.
                x = self.net(seasonal_init, trend_init, x_raw=x_raw)
            else:
                # Single CCM mode:
                # choose one input type for single assigner.
                x_ccm = self._select_single_ccm_input(
                    x_raw=x_raw,
                    seasonal_init=seasonal_init,
                    trend_init=trend_init,
                )
                x = self.net(seasonal_init, trend_init, x_raw=x_ccm)

        # Optional CCM cluster losses
        self._compute_ccm_losses()

        # RevIN denormalization
        if self.revin:
            x = self.revin_layer(x, "denorm")

        return x
