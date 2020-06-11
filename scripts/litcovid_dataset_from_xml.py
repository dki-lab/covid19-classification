import pandas as pd
import nltk
from tqdm import tqdm
import re
from utils import *
import bioc

## Loading LitCovid Data Directly

# Deserialize ``fp`` to a BioC collection object.
with open('../data/litcovid2BioCXML', 'r') as fp:
    collection = bioc.load(fp)

docs = collection.documents

articles = {}
xmls = {}

for doc in docs:
    try:
        pmid = doc.passages[0].infons['article-id_pmid']
    except:
        pmid = doc.id
        
    text = []
    for passage in doc.passages:
        p = passage.text
        text.append(p)
    
    articles[pmid] = '\n\n'.join(text)

    xmls[pmid] = doc

litcovid_xml_data = df_from_dict(articles,'pmid','text')
litcovid_xml_data['length'] = [len(t.split()) for t in litcovid_xml_data['text']]


print("Average Length of full text: " + str(litcovid_xml_data.length.mean()))

litcovid_xml_data['pmid'] = [int(pmid) for pmid in litcovid_xml_data.pmid]

## Using LitCovid Data Directly

litcovid = load_all_litcovid()
litcovid['category'] = litcovid['source']

litcovid_dataset = litcovid_xml_data.merge(litcovid,on='pmid',how='inner')

#Filtering Documents with no text
litcovid_dataset['title_len'] = [len(str(t).split()) for t in litcovid_dataset.title]
litcovid_dataset['diff'] = litcovid_dataset.length - litcovid_dataset.title_len
litcovid_dataset_no_text = litcovid_dataset[litcovid_dataset['diff'] < 25]
litcovid_dataset = litcovid_dataset[litcovid_dataset['diff'] > 25]

print("Number of Articles x Categories in LitCovid: {}".format(len(litcovid_dataset)))

litcovid_dataset['doc_type'] = 'text'
litcovid_dataset.loc[litcovid_dataset['length'] < 300,'doc_type'] = 'abs'

litcovid_dataset = litcovid_dataset[['pmid','text','title','category','length','doc_type']].reset_index(drop=True)


print("Number of Text Articles in LitCovid: {}".format(len(litcovid_dataset.pmid.unique())))

with open('../data/litcovid_category_order.txt','r') as f:
    categories = f.readlines()
    categories = [c.strip() for c in categories]


# Creating Unique Labels

concat_dfs = []

for pmid, df in litcovid_dataset.groupby('pmid'):
    row = df[['pmid','title','text','doc_type']].drop_duplicates()
    
    label = []
    cats = []
    for cat in categories:
        if cat in df.category.values:
            label.append('1')
            cats.append(cat)
        else:
            label.append('0')

    if len(row) > 1:
        i = row.index.values[0]
        row = row[row.index == i]
        
    row['human_label'] = ', '.join(cats)
    row['label'] = ''.join(label)
    
    concat_dfs.append(row)
    
litcovid_dataset_labelled = pd.concat(concat_dfs)

def tokenize_lines(text):
    ms = re.finditer('\n+|$',text)

    st = 0

    tokenized_text = []

    for m in ms:
        line = text[st:m.start()]

        sents = nltk.sent_tokenize(line)

        for sent in sents:
            sent = sent.lower()
            words = nltk.word_tokenize(sent)
            tokenized_text.append(words)
            
        try:
            if len(tokenized_text) == 0:
                tokenized_text.append([''])
            
            tokenized_text[-1].extend([m.group()])
        except:
            print(tokenized_text)
            raise
            
        st = m.end()

    return tokenized_text

def tokenize_text_w_para(df, column):
    tokenized_pubs = []

    for text in tqdm(df[column]):
        tokenized_text = tokenize_lines(text)
        tokenized_pubs.append(tokenized_text)
    
    return tokenized_pubs

litcovid_dataset_labelled['clean_doc_tokenized'] = tokenize_text_w_para(litcovid_dataset_labelled,'text')
litcovid_dataset_labelled['clean_doc'] = [' '.join([' '.join(sent) for sent in doc]).replace('\n','\\n') for doc in litcovid_dataset_labelled['clean_doc_tokenized']]


litcovid_dataset_labelled['sequence_len'] = [len(t.split()) for t in litcovid_dataset_labelled['text']]
litcovid_dataset_labelled['num_sents'] = [len(t) for t in litcovid_dataset_labelled.clean_doc_tokenized]

print("LitCovid Document Classification Dataset Statistics:")
print("Avg. Sents: {}, Avg. Tokens: {}, 'Total Tokens: {}".format(litcovid_dataset_labelled.num_sents.mean(), litcovid_dataset_labelled.sequence_len.mean(),litcovid_dataset_labelled.sequence_len.sum()))

# ## Splitting Dataset IID Across Different Document Types (Abstract, Full Text and Both)

def split_df_to_train(df, train,dev,test):
    df = df.sample(frac=1,random_state=42).reset_index()
    
    train_border = int(train*len(df))
    dev_border = int(train*len(df)) + int(dev*len(df)) + 1
    
    train_df = df[:train_border]
    dev_df = df[train_border:dev_border]
    test_df = df[dev_border:]
    
    assert len(train_df) + len(dev_df) + len(test_df) == len(df)
    return train_df, dev_df, test_df

data_split = {'train':[],'dev':[],'test':[]}

for doc_type, df in litcovid_dataset_labelled.groupby('doc_type'):
    df.loc[:,'doc_type'] = doc_type
    split = split_df_to_train(df,0.7,0.1,0.2)
    
    data_split['train'].append(split[0])
    data_split['dev'].append(split[1])
    data_split['test'].append(split[2])

train = pd.concat(data_split['train'])
dev = pd.concat(data_split['dev'])
test = pd.concat(data_split['test'])

train[['label','clean_doc']].to_csv('../data/FullLitCovid/train.tsv',header=False,index=False,sep='\t')
dev[['label','clean_doc']].to_csv('../data/FullLitCovid/dev.tsv',header=False,index=False,sep='\t')
test[['label','clean_doc']].to_csv('../data/FullLitCovid/test.tsv',header=False,index=False,sep='\t')

#Saving Train/Dev/Test With All Information

train[['pmid','title','text','clean_doc','clean_doc_tokenized','human_label','label']].to_csv('../data/FullLitCovid/LitCovid.train.csv')
dev[['pmid','title','text','clean_doc','clean_doc_tokenized','human_label','label']].to_csv('../data/FullLitCovid/LitCovid.dev.csv')
test[['pmid','title','text','clean_doc','clean_doc_tokenized','human_label','label']].to_csv('../data/FullLitCovid/LitCovid.test.csv')

print("Split Statistics:")
print("Train: {}, Dev: {}, Test: {}".format(len(train),len(dev),len(test)))

# Creating Class Balanced Subsets

data_eff_train = {}

for frac in [0.01,0.05,0.10,0.20,0.50]:
    
    subset = []
    
    for i,df in train.groupby('label'):
        n = max(int(len(df) * frac),1)
        subset.append(train.sample(n=n,random_state=42).reset_index())
        
    data_eff_train['{:.2f}'.format(frac)] = pd.concat(subset)

for frac in data_eff_train:
    df = data_eff_train[frac]
    df = df[['label','clean_doc']]
    
    df.to_csv('../data/FullLitCovid/train_{}.tsv'.format(frac),header=False,index=False,sep='\t')

print('Done')