# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 06:13:26 2022

@author: Pei-Hsun
"""

from nltk.tokenize import TweetTokenizer
from emoji import demojize
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import pandas as pd
import os
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start:", current_time)

reading_path = ''
writing_path = ''

tokenizer = TweetTokenizer()

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token

def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = normTweet.replace("cannot ", "can not ").replace("n't ", " n't ").replace("n 't ", " n't ").replace("ca n't", "can't").replace("ai n't", "ain't")
    normTweet = normTweet.replace("'m ", " 'm ").replace("'re ", " 're ").replace("'s ", " 's ").replace("'ll ", " 'll ").replace("'d ", " 'd ").replace("'ve ", " 've ")
    normTweet = normTweet.replace(" p . m .", "  p.m.") .replace(" p . m ", " p.m ").replace(" a . m .", " a.m.").replace(" a . m ", " a.m ")

    normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)
    
    return " ".join(normTweet.split())

def data_stream(texts):
    for text in texts:
        text = normalizeTweet(text)
        yield text


bert_tokenizer = AutoTokenizer.from_pretrained("bertweet20221020/tokenizer")
model = AutoModelForSequenceClassification.from_pretrained("bertweet20221020/trained").to('cuda')
bertpipe  = pipeline("text-classification", model=model, tokenizer=bert_tokenizer, device=0, return_all_scores=True, truncation=True, padding=True, max_length=100)


df_reader = pd.read_csv(reading_path, usecols = ['id','text'],chunksize=1000)
for df in df_reader:
    idx = df['id'].to_list()
    texts = df['text'].to_list()
    predictions = []
    
    for i, out in zip(idx,bertpipe(data_stream(texts))):
      predictions.append({'id':i,'score0': out[0]['score'], 'score1':out[1]['score'],'pred': 1 if out[1]['score']>out[0]['score'] else 0})

    pd.DataFrame(predictions).to_csv(writing_path,index=False,mode='a', header=not os.path.exists(writing_path))

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End:", current_time)