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
    --data_path ETTm2.csv \
    --model_id ETTm2_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.0001 \
    --lradj 'sigmoid'\
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta > logs/$ma_type/$model_name'_ETTm2_'$seq_len'_'$pred_len.log
done
done
