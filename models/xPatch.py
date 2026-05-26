import torch
import torch.nn as nn
import math

from layers.decomp import DECOMP
from layers.network import Network
from layers.revin import RevIN
from layers.ccm_xpatch import CCMClusterLoss


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.aux_loss = None
        self.raw_cluster_loss = None
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

        # Moving Average
        self.ma_type = configs.ma_type
        alpha = configs.alpha
        beta = configs.beta
        self.decomp = DECOMP(self.ma_type, alpha, beta)

        # CCM config
        self.use_ccm_head = getattr(configs, "use_ccm_head", False)
        self.ccm_loss_weight = getattr(configs, "ccm_loss_weight", 0.0)

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
        )

        if self.use_ccm_head and self.ccm_loss_weight > 0:
            self.ccm_loss_fn = CCMClusterLoss()
        else:
            self.ccm_loss_fn = None

        self.aux_loss = None

    def get_aux_loss(self):
        """
        Called by exp_main.py after forward.
        """
        if self.aux_loss is None:
            return 0.0
        return self.aux_loss

    def get_raw_cluster_loss(self):
        if self.raw_cluster_loss is None:
            return 0.0
        return self.raw_cluster_loss

    def forward(self, x):
        """
        Args:
            x: [B, L, C]

        Returns:
            x: [B, pred_len, C]
        """

        self.aux_loss = None
        self.raw_cluster_loss = None

        # RevIN normalization
        if self.revin:
            x = self.revin_layer(x, "norm")

        # normalized raw input for CCM cluster assignment
        x_raw = x

        # xPatch backbone
        if self.ma_type == "reg":
            x = self.net(x, x, x_raw=x_raw)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init, x_raw=x_raw)

        # Optional CCM cluster loss
        raw_cluster_loss = None
        
        if (
            self.use_ccm_head
            and self.ccm_loss_fn is not None
            and self.net.ccm_sim_matrix is not None
            and self.net.ccm_membership is not None
        ):
            raw_cluster_loss = self.ccm_loss_fn(
                self.net.ccm_sim_matrix,
                self.net.ccm_membership,
            )
        
        self.raw_cluster_loss = raw_cluster_loss
        
        if raw_cluster_loss is None:
            self.aux_loss = None
        else:
            self.aux_loss = self.ccm_loss_weight * raw_cluster_loss

        # RevIN denormalization
        if self.revin:
            x = self.revin_layer(x, "denorm")

        return x
