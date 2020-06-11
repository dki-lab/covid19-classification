#!/bin/bash

python -m models.bert --dataset LitCovid --model bert-base-uncased --max-seq-length 512 --batch-size 6 --lr 2e-5 --epochs 30
python -m models.bert --dataset LitCovid --model bert-large-uncased --max-seq-length 512 --batch-size 3 --lr 2e-5 --epochs 30
python -m models.bert --dataset LitCovid --model biobert --max-seq-length 512 --batch-size 6 --lr 2e-5 --epochs 30
python -m models.longformer --dataset LitCovid --model longformer-base --max-seq-length 1024 --batch-size 3 --lr 2e-5 --epochs 30