#!/bin/bash
set -e

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ma_type=ema
alpha=0.3
beta=0.3

model_name=xPatch
seq_len=96

# =========================
# CCM branch control
# Options:
#   seasonal : only seasonal branch uses CCM
#   trend    : only trend branch uses CCM
#   both     : both seasonal and trend branches use CCM
# =========================
ccm_branch=trend

# Seasonal branch CCM strength
ccm_seasonal_loss_weight=0.01
ccm_seasonal_residual_weight=1.0

# Trend branch CCM strength
ccm_trend_loss_weight=0.003
ccm_trend_residual_weight=0.3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/$ma_type" ]; then
    mkdir ./logs/$ma_type
fi

if [ ! -d "./logs/$ma_type/dual_ccm_${ccm_branch}" ]; then
    mkdir ./logs/$ma_type/dual_ccm_${ccm_branch}
fi

for model_name in xPatch
do
for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_${pred_len}_${ma_type}_dual_ccm_${ccm_branch} \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des Exp \
    --itr 1 \
    --batch_size 512 \
    --learning_rate 0.0005 \
    --lradj sigmoid \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta \
    --use_ccm_head 1 \
    --use_dual_ccm 1 \
    --ccm_head_type $ccm_branch \
    --n_cluster 2 \
    --ccm_d_model 128 \
    --ccm_sigma 1.0 \
    --ccm_epsilon 0.3 \
    --ccm_gumbel_temp 0.5 \
    --ccm_use_gumbel 1 \
    --ccm_dropout 0.0 \
    --ccm_loss_weight $ccm_seasonal_loss_weight \
    --ccm_residual_weight $ccm_seasonal_residual_weight \
    --ccm_trend_loss_weight $ccm_trend_loss_weight \
    --ccm_trend_residual_weight $ccm_trend_residual_weight \
    --ccm_use_prototype 1 \
    --ccm_prob_mode learned \
    --train_loss_type mae \
    --vali_loss_type mae \
    --use_loss_ratio 1 \
    --ccm_diag_interval 10 \
    --ccm_save_visuals 1 \
    --ccm_save_pca 1 \
    > logs/$ma_type/dual_ccm_${ccm_branch}/${model_name}_ETTh1_${seq_len}_${pred_len}_dual_ccm_${ccm_branch}.log 2>&1
done
done
