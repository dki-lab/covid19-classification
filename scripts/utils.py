import pandas as pd
import glob
import re
from tqdm import tqdm
import nltk

def tokenize_text(df, column):
    tokenized_pubs = []

    for text in tqdm(df[column]):
        tokenized_text = []
        sents = nltk.sent_tokenize(text)

        for sent in sents:
            sent = sent.lower()
            words = nltk.word_tokenize(sent)
            tokenized_text.append(words)

        tokenized_pubs.append(tokenized_text)
    
    return tokenized_pubs

def tokenize_text_bert(df, column):
    tokenized_pubs = []

    for text in tqdm(df[column]):
        tokenized_text = []
        sents = nltk.sent_tokenize(text)

        for sent in sents:
            sent = sent.lower()
            words = nltk.word_tokenize(sent)
            if words[-1] == '.':
                words = words[:-1]
            words.append('[SEP]')
            tokenized_text.append(words)

        tokenized_pubs.append(tokenized_text)
    
    return tokenized_pubs


def load_all_cord(source_file_path="../data/cord-19-sources"): 

    files = glob.glob(source_file_path+"/clean*")

    dfs = []

    for file in files:
        print(file)
        df = pd.read_csv(file)
        df["source"] = file.split('/')[-1][:-4]

        dfs.append(df)

    covid_pubs = pd.concat(dfs)
    
    return covid_pubs


def load_all_litcovid(source_file_path = "../data"):

    files = glob.glob(source_file_path+"/litcovid_source*.tsv")

    dfs = []

    for file in files:
        df = pd.read_csv(file,sep='\t',comment='#')
        df["source"] = file.split('/')[-1][16:-4]
        dfs.append(df)
        
    litcovid = pd.concat(dfs)
    
    return litcovid

def add_count_to_dict(term_dict, term, count = 1):
    
    if term in term_dict:
        term_dict[term] += count
    else:
        term_dict[term] = count
        
def extract_ngrams_from_sent_list(sent_list, relevant_ngrams, max_length=6):

    freq_dict = {}
    
    for token_list in sent_list:
    
        num_tokens = len(token_list)

        token_list = [token.lower() for token in token_list]

        for st in range(num_tokens-1):

            for ngram_len in range(min(max_length, num_tokens-st-1)):
                end = st+ngram_len+1

                ngram = ' '.join(token_list[st:end])
                
                if ngram in relevant_ngrams:
                    add_count_to_dict(freq_dict, ngram)
                               
    return freq_dict    

def df_from_dict(d, keys, vals):
    
    df = pd.DataFrame()
    
    df[keys] = list(d.keys())
    df[vals] = list(d.values())
    
    return df
