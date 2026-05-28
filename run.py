import argparse
import os
import sys
import traceback
import torch
import random
import numpy as np

from exp.exp_main import Exp_Main


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True

    if v.lower() in ("no", "false", "f", "0", "n"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser():
    parser = argparse.ArgumentParser(description="xPatch")

    # basic config
    parser.add_argument("--is_training", type=int, required=True, default=1, help="status")
    parser.add_argument(
        "--train_only",
        type=str2bool,
        required=False,
        default=False,
        help="perform training on full input dataset without validation and testing",
    )
    parser.add_argument("--model_id", type=str, required=True, default="test", help="model id")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="xPatch",
        help="model name, options: [xPatch]",
    )

    # data loader
    parser.add_argument("--data", type=str, required=True, default="ETTh1", help="dataset type")
    parser.add_argument("--root_path", type=str, default="./dataset", help="root path of the data file")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]",
    )
    parser.add_argument("--target", type=str, default="OT", help="target feature in S or MS task")
    parser.add_argument("--freq", type=str, default="h", help="freq for time features encoding")
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction sequence length")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")

    # patching
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")
    parser.add_argument("--stride", type=int, default=8, help="stride")
    parser.add_argument(
        "--padding_patch",
        type=str,
        default="end",
        help="None: None; end: padding on the end",
    )

    # moving average
    parser.add_argument("--ma_type", type=str, default="ema", help="reg, ema, dema")
    parser.add_argument("--alpha", type=float, default=0.3, help="alpha")
    parser.add_argument("--beta", type=float, default=0.3, help="beta")

    # optimization
    parser.add_argument("--num_workers", type=int, default=10, help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=100, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="mse", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )
    parser.add_argument("--revin", type=int, default=1, help="RevIN; True 1 False 0")

    # GPU
    parser.add_argument("--use_gpu", type=str2bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu",
        action="store_true",
        help="use multiple gpus",
        default=False,
    )
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multiple gpus")
    parser.add_argument("--test_flop", action="store_true", default=False, help="See utils/tools for usage")

    # =========================
    # CCM-xPatch Head config
    # =========================
    parser.add_argument(
        "--use_ccm_head",
        type=int,
        default=0,
        help="whether to use CCM-style cluster-aware prediction head",
    )

    parser.add_argument(
        "--use_dual_ccm",
        type=int,
        default=0,
        help="whether to use dual CCM: seasonal_init->CCM_s and trend_init->CCM_t",
    )

    parser.add_argument(
        "--ccm_head_type",
        type=str,
        default="seasonal",
        choices=["seasonal", "trend", "both"],
        help="where to use CCM-aware head",
    )

    parser.add_argument("--n_cluster", type=int, default=2, help="number of CCM clusters")

    parser.add_argument(
        "--ccm_d_model",
        type=int,
        default=32,
        help="hidden dimension of CCM channel embedding",
    )

    parser.add_argument(
        "--ccm_sigma",
        type=float,
        default=5.0,
        help="sigma for RBF channel similarity in CCM loss",
    )

    parser.add_argument(
        "--ccm_epsilon",
        type=float,
        default=0.2,
        help="temperature/epsilon for cluster probability",
    )

    parser.add_argument(
        "--ccm_gumbel_temp",
        type=float,
        default=0.5,
        help="temperature for differentiable Bernoulli membership",
    )

    parser.add_argument(
        "--ccm_use_gumbel",
        type=int,
        default=0,
        help="whether to use Gumbel-Sigmoid membership during training",
    )

    parser.add_argument("--ccm_dropout", type=float, default=0.0, help="dropout for CCM module")

    parser.add_argument(
        "--ccm_loss_weight",
        type=float,
        default=0.001,
        help="weight for seasonal/single CCM cluster loss; 0 means disabled",
    )

    parser.add_argument(
        "--ccm_trend_loss_weight",
        type=float,
        default=0.001,
        help="weight for trend CCM cluster loss in dual CCM mode",
    )

    parser.add_argument(
        "--ccm_residual_weight",
        type=float,
        default=0.5,
        help="residual mixing weight for seasonal CCM head",
    )

    parser.add_argument(
        "--ccm_trend_residual_weight",
        type=float,
        default=0.3,
        help="residual mixing weight for trend CCM head",
    )

    parser.add_argument(
        "--ccm_use_prototype",
        type=int,
        default=1,
        help="whether to use prototype learning in CCM assigner",
    )

    parser.add_argument(
        "--ccm_prob_mode",
        type=str,
        default="learned",
        choices=["learned", "uniform", "shuffle", "random"],
        help="probability mode for CCM ablation",
    )

    # Only used in backward-compatible single-CCM mode.
    parser.add_argument(
        "--ccm_input_type",
        type=str,
        default="raw",
        choices=[
            "raw",
            "seasonal",
            "trend",
            "raw_plus_seasonal",
            "seasonal_plus_trend",
        ],
        help=(
            "single-CCM only: which signal is used for CCM cluster assignment"
        ),
    )

    parser.add_argument(
        "--train_loss_type",
        type=str,
        default="mae",
        choices=["mae", "mse"],
        help="training forecasting loss type",
    )

    parser.add_argument(
        "--vali_loss_type",
        type=str,
        default="mae",
        choices=["mae", "mse"],
        help="validation loss type for early stopping",
    )

    parser.add_argument(
        "--use_loss_ratio",
        type=int,
        default=1,
        help="whether to use original xPatch horizon weighted loss ratio",
    )

    parser.add_argument(
        "--ccm_diag_interval",
        type=int,
        default=5,
        help="print CCM diagnosis every N epochs; <=0 disables diagnosis",
    )

    parser.add_argument(
        "--ccm_save_visuals",
        type=int,
        default=0,
        help="whether to save CCM heatmaps and scatter plots",
    )

    parser.add_argument(
        "--ccm_save_pca",
        type=int,
        default=0,
        help="whether to save channel/prototype PCA plot",
    )

    return parser


def normalize_args(args):
    args.use_ccm_head = bool(args.use_ccm_head)
    args.use_dual_ccm = bool(args.use_dual_ccm)
    args.ccm_use_gumbel = bool(args.ccm_use_gumbel)
    args.ccm_use_prototype = bool(args.ccm_use_prototype)
    args.use_loss_ratio = bool(args.use_loss_ratio)
    args.revin = bool(args.revin)

    args.ccm_save_visuals = int(args.ccm_save_visuals)
    args.ccm_save_pca = int(args.ccm_save_pca)

    if args.use_dual_ccm:
        args.use_ccm_head = True

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    return args


def set_seed(seed=2021):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("[RUN.PY STARTED]", flush=True)
    print("python:", sys.executable, flush=True)
    print("cwd:", os.getcwd(), flush=True)
    print("torch:", torch.__version__, flush=True)
    print("cuda available:", torch.cuda.is_available(), flush=True)

    set_seed(2021)

    parser = build_parser()
    args = parser.parse_args()
    args = normalize_args(args)

    print("Args in experiment:", flush=True)
    print(args, flush=True)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}".format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.des,
                ii,
            )

            exp = Exp(args)

            print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting), flush=True)
            exp.train(setting)

            print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting), flush=True)
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}".format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.des,
            ii,
        )

        exp = Exp(args)
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting), flush=True)
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[RUN.PY ERROR]", flush=True)
        traceback.print_exc()
        sys.exit(1)
