# COVID-19 Document Classification

This repo provides a platform for testing document classification models on COVID-19 Literature. 

It is an extension of the [Hedwig](https://github.com/castorini/hedwig) library and contains all necessary code to reproduce the results of some document classification models on a COVID-19 dataset created from the [LitCovid](https://www.ncbi.nlm.nih.gov/research/coronavirus/) collection. More information about the models tested and experiments carried out can be found [here](https://arxiv.org/abs/2006.13816).

The Hedwig library was modified to work with a newer version of PyTorch and the Transformers library in order to import custom models. It was also extended to adapt [DocBERT](https://arxiv.org/abs/1904.08398) to use the Longformer model.

## Data

### LitCovid Data

You can download the LitCovid document classification dataset from August 1<sup>st</sup>, 2020 by following [this link](https://drive.google.com/drive/folders/1xaaf9ZYX5P8s_l4XlFaP2byXLL22jstZ?usp=sharing).


Replace the empty `hedwig-data` and `data` directories in this repository with the same directories downloaded from the link above. The data used for training will be under the following directory.

```
hedwig-data/datasets/LitCovid/
```

You can find a version compatible with the Hedwig library `train.tsv` and a raw version  `LitCovid.train.csv` which includes PMIDs for each article.

We have also included a script to download the most up-to-date version of the LitCovid dataset by running the following commands:
 
```bash
cd scripts
bash load_litcovid.sh
```

This script will download, process and save both the processed and raw versions into the `data/FullLitCovid` directory. Please move the files to `hedwig-data/datasets/LitCovid` for further training.
This new data will not maintain the train, dev, test split found in the paper.

### Word Embeddings

Along with the dataset, you must download the word2vec embeddings for the traditional deep learning models.

Follow these instructions from the Hedwig repo to download the embeddings. The only files needed for our purposes should be under:
```
hedwig-data/embeddings
```

Option 1. Our [Wasabi](https://wasabi.com/)-hosted mirror:

```bash
$ wget http://nlp.rocks/hedwig -O hedwig-data.zip
$ unzip hedwig-data.zip
```

Option 2. Our school-hosted repository, [`hedwig-data`](https://git.uwaterloo.ca/jimmylin/hedwig-data):

```bash
$ git clone https://github.com/castorini/hedwig.git
$ git clone https://git.uwaterloo.ca/jimmylin/hedwig-data.git
```

Next, organize your directory structure as follows:

```
.
├── hedwig
└── hedwig-data
```

After cloning the hedwig-data repo, you need to unzip the embeddings and run the preprocessing script:

```bash
cd hedwig-data/embeddings/word2vec 
tar -xvzf GoogleNews-vectors-negative300.tgz
```

### Pre-Trained Longformer

Finally, you'll need the Longformer Pre-trained model.

Run the following commands to download the pre-trained Longformer from their [original repo](https://github.com/allenai/longformer).

```bash
cd hedwig
wget https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-base-4096.tar.gz
tar -xvzf longformer-base-4096.tar.gz
```

### CORD-19 Test Dataset

The full articles, PMIDs and labels for the CORD-19 test dataset can be found under `data/cord19_annotations.tsv`.

The Jupyter notebook `scripts/clean_cord19_dataset.ipynb` creates the Hedwig compatible version found in `hedwig-data/dataset/LitCovid/cord19_test.tsv`.

## Requirements

We'd recommend creating a custom pip virtual environment and install all requirements via pip:

```
$ pip install -r requirements.txt
```

Code depends on data from NLTK (e.g., stopwords) so you'll have to download them. 
Run the Python interpreter and type the commands:

```python
>>> import nltk
>>> nltk.download()
```

## Model Training

Running the following commands in the hedwig directory to train the transformer based models and the conventional deep learning models respectively: 

```bash
bash run_all_bert.sh
bash run_all_deep.sh
```

To reproduce the results for traditional machine learning models on the LitCovid dataset run the Jupyter notebook `scripts/traditional_models.ipynb
`.

To reproduce the results on the CORD-19 test dataset run the following:

```bash
bash run_baselines_cord19.sh
```
