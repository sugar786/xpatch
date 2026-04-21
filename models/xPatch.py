import torch
import torch.nn as nn
import math

from layers.decomp import DECOMP
from layers.network import Network
from layers.revin import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Parameters
        seq_len = configs.seq_len   # lookback window L
        pred_len = configs.pred_len # prediction length (96, 192, 336, 720)
        c_in = configs.enc_in       # input channels

        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)

        # Moving Average
        self.ma_type = configs.ma_type
        alpha = configs.alpha       # smoothing factor for EMA (Exponential Moving Average)
        beta = configs.beta         # smoothing factor for DEMA (Double Exponential Moving Average)

        self.decomp = DECOMP(self.ma_type, alpha, beta)
        self.net = Network(
            seq_len,
            pred_len,
            patch_len,
            stride,
            padding_patch,
            use_trend_interactor=getattr(configs, 'use_trend_interactor', False),
            topk=getattr(configs, 'topk', 4),
            interactor_dropout=getattr(configs, 'interactor_dropout', 0.0)
        )
        

    def forward(self, x):
        # x: [Batch, Input, Channel]

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':   # If no decomposition, directly pass the input to the network
            x = self.net(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x
