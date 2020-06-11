"""
EDIT NOTICE

File edited from original in https://github.com/castorini/hedwig
by Bernal Jimenez Gutierrez (jimenezgutierrez.1@osu.edu)
in May 2020
"""

import os

# Model categories
BERT_MODELS = ['BERT-Base', 'BERT-Large', 'HBERT-Base', 'HBERT-Large']

# String templates for logging results
LOG_HEADER = 'Split  Dev/Acc.  Dev/Pr.  Dev/Re.   Dev/F1   Dev/Loss'
LOG_TEMPLATE = ' '.join('{:>5s},{:>9.4f},{:>8.4f},{:8.4f},{:8.4f},{:10.4f}'.split(','))

# Path to pretrained model and vocab files
MODEL_DATA_DIR = os.path.join(os.pardir, 'hedwig-data', 'models')
PRETRAINED_MODEL_ARCHIVE_MAP = {
    'covid-bert': 'deepset/covid_bert_base',
    'bert-base-uncased': 'bert-base-uncased',
    'bert-large-uncased': 'bert-large-uncased',
    'bert-base-cased':'bert-base-cased',
    'bert-large-cased': 'bert-large-cased',
    'biobert': 'monologg/biobert_v1.1_pubmed',
    'longformer-base': 'allenai/longformer-base-4096',
    'longformer-large': 'allenai/longformer-large-4096',
    'reformer': "patrickvonplaten/reformer-tiny-random",
    'reformer-random': 'google/reformer-crime-and-punishment',
    'bert-base-multilingual-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-uncased'),
    'bert-base-multilingual-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-cased')
}
PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'covid-bert': 'deepset/covid_bert_base',
    'bert-base-uncased': 'bert-base-uncased',
    'bert-large-uncased': 'bert-large-uncased',
    'bert-base-cased': 'bert-base-cased',
    'bert-large-cased':'bert-large-cased',
    'biobert': 'monologg/biobert_v1.1_pubmed',
    'longformer-base':'allenai/longformer-base-4096',
    'longformer-large': 'allenai/longformer-large-4096',
    'reformer': "patrickvonplaten/reformer-tiny-random",
    'reformer-random':'google/reformer-crime-and-punishment',
    'bert-base-multilingual-uncased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-uncased-vocab.txt'),
    'bert-base-multilingual-cased': os.path.join(MODEL_DATA_DIR, 'bert_pretrained', 'bert-base-multilingual-cased-vocab.txt')
}
