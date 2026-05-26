ma_type=ema
alpha=0.3
beta=0.3

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/"$ma_type ]; then
    mkdir ./logs/$ma_type
fi

model_name=xPatch
seq_len=96

for model_name in xPatch
do
for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 512 \
    --learning_rate 0.0005 \
    --lradj 'sigmoid'\
    --ma_type $ma_type \
    --alpha $alpha \
    --use_ccm_head 1 \
    --ccm_head_type seasonal \
    --n_cluster 2 \
    --ccm_d_model 128 \
    --ccm_sigma 1.0 \
    --ccm_epsilon 0.3 \
    --ccm_gumbel_temp 0.5 \
    --ccm_use_gumbel 1 \
    --ccm_dropout 0.0 \
    --ccm_loss_weight 0.01 \
    --ccm_residual_weight 1.0 \
    --train_loss_type mae \
    --vali_loss_type mae \
    --use_loss_ratio 1 \
    --beta $beta > logs/$ma_type/$model_name'_ETTh1_'$seq_len'_'$pred_len.log
done
done
