#!/bin/bash

python -m models.kim_cnn --mode static --dataset LitCovid --batch-size 32 --lr 0.001 --epochs 30 --dropout 0.1 --seed 3435
python -m models.xml_cnn --mode static --dataset LitCovid --batch-size 32 --lr 0.001 --epochs 30 --dropout 0.7 --dynamic-pool-length 8 --seed 3435
python -m models.reg_lstm --dataset LitCovid --mode static --batch-size 16 --lr 0.001 --epochs 30 --bidirectional --num-layers 1 --hidden-dim 512 --wdrop 0.1 --embed-droprate 0.2 --dropout 0.1 --beta-ema 0.0 --seed 3435
python -m models.reg_lstm --dataset LitCovid --mode static --batch-size 8 --lr 0.001 --epochs 30 --bidirectional --num-layers 1 --hidden-dim 300 --wdrop 0.1 --embed-droprate 0.2 --dropout 0.5 --beta-ema 0.99 --seed 3435