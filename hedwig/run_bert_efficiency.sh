#!/bin/bash

GPU=$1

CUDA_VISIBLE_DEVICES=$GPU python -m models.bert --dataset LitCovid --model bert-base-uncased --max-seq-length 512 --batch-size 6 --lr 2e-5 --epochs 30  --seed 3436 --training_file train_0.01.tsv
CUDA_VISIBLE_DEVICES=$GPU python -m models.bert --dataset LitCovid --model bert-large-uncased --max-seq-length 512 --batch-size 2 --lr 2e-5 --epochs 30 --seed 3436 --training_file train_0.01.tsv
CUDA_VISIBLE_DEVICES=$GPU python -m models.bert --dataset LitCovid --model biobert --max-seq-length 512 --batch-size 6 --lr 2e-5 --epochs 30 --seed 3436 --training_file train_0.01.tsv
CUDA_VISIBLE_DEVICES=$GPU python -m models.longformer --dataset LitCovid --model longformer-base --max-seq-length 1024 --batch-size 3 --lr 2e-5 --epochs 30 --seed 3436 --training_file train_0.01.tsv

CUDA_VISIBLE_DEVICES=$GPU python -m models.bert --dataset LitCovid --model bert-base-uncased --max-seq-length 512 --batch-size 6 --lr 2e-5 --epochs 30 --seed 3437 --training_file train_0.01.tsv
CUDA_VISIBLE_DEVICES=$GPU python -m models.bert --dataset LitCovid --model bert-large-uncased --max-seq-length 512 --batch-size 2 --lr 2e-5 --epochs 30 --seed 3437 --training_file train_0.01.tsv
CUDA_VISIBLE_DEVICES=$GPU python -m models.bert --dataset LitCovid --model biobert --max-seq-length 512 --batch-size 6 --lr 2e-5 --epochs 30 --seed 3437 --training_file train_0.01.tsv
CUDA_VISIBLE_DEVICES=$GPU python -m models.longformer --dataset LitCovid --model longformer-base --max-seq-length 1024 --batch-size 3 --lr 2e-5 --seed 3437 --epochs 30 --training_file train_0.01.tsv
