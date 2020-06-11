#!/bin/bash

GPU=$1

CUDA_VISIBLE_DEVICES=$GPU python -m models.reg_lstm --dataset LitCovid --mode static --batch-size 8 --lr 0.001 --epochs 30 --bidirectional --num-layers 1 --hidden-dim 300 --wdrop 0.1 --embed-droprate 0.2 --dropout 0.5 --beta-ema 0.99 --seed 3435 --training_file train_0.01.tsv
CUDA_VISIBLE_DEVICES=$GPU python -m models.reg_lstm --dataset LitCovid --mode static --batch-size 8 --lr 0.001 --epochs 30 --bidirectional --num-layers 1 --hidden-dim 300 --wdrop 0.1 --embed-droprate 0.2 --dropout 0.5 --beta-ema 0.99 --seed 3435 --training_file train_0.05.tsv
CUDA_VISIBLE_DEVICES=$GPU python -m models.reg_lstm --dataset LitCovid --mode static --batch-size 8 --lr 0.001 --epochs 30 --bidirectional --num-layers 1 --hidden-dim 300 --wdrop 0.1 --embed-droprate 0.2 --dropout 0.5 --beta-ema 0.99 --seed 3435 --training_file train_0.10.tsv
CUDA_VISIBLE_DEVICES=$GPU python -m models.reg_lstm --dataset LitCovid --mode static --batch-size 8 --lr 0.001 --epochs 30 --bidirectional --num-layers 1 --hidden-dim 300 --wdrop 0.1 --embed-droprate 0.2 --dropout 0.5 --beta-ema 0.99 --seed 3435 --training_file train_0.20.tsv
CUDA_VISIBLE_DEVICES=$GPU python -m models.reg_lstm --dataset LitCovid --mode static --batch-size 8 --lr 0.001 --epochs 30 --bidirectional --num-layers 1 --hidden-dim 300 --wdrop 0.1 --embed-droprate 0.2 --dropout 0.5 --beta-ema 0.99 --seed 3435 --training_file train_0.50.tsv