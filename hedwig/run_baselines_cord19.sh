#!/bin/bash

python -m models.reg_lstm --dataset LitCovid --mode static --batch-size 8 --lr 0.001 --epochs 30 --bidirectional --num-layers 1 --hidden-dim 300 --wdrop 0.1 --embed-droprate 0.2 --dropout 0.5 --beta-ema 0.99 --seed 3435 --extra_test_set cord19_test.tsv
python -m models.bert --dataset LitCovid --model biobert --max-seq-length 512 --batch-size 6 --lr 2e-5 --epochs 30 --test_set cord19_test.tsv
python -m models.longformer --dataset LitCovid --model longformer-base --max-seq-length 1024 --batch-size 3 --lr 2e-5 --epochs 30 --test_set cord19_test.tsv