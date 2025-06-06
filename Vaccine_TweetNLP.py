# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 22:52:14 2024

@author: Pei-Hsun Hsieh
"""

from pathlib import Path
import pandas as pd
import os
from datetime import datetime
from transformers import pipeline

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start:", current_time)

reading_path = Path("")
csv_files = reading_path.rglob("*.csv")
writing_path = ""

pipe_sentiment = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment-latest", tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest", padding=True, truncation=True, max_length=512, return_all_scores=True, device=0)
pipe_emotion = pipeline('text-classification', model="cardiffnlp/twitter-roberta-base-emotion-multilabel-latest", tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest", padding=True, truncation=True, max_length=512, return_all_scores=True, device=0)

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def data_stream(texts):
    for text in texts:
        text = preprocess(text)
        yield text

idx_exceptions = []
for file in csv_files:
    print(file)
    writing_file = writing_path+f'USA_roberta_Vax_tweets_{"_".join(Path(file).stem.split("_")[-2:])}.csv'
    df_reader = pd.read_csv(file, usecols = ['id','text'],chunksize=1000)
    for df in df_reader:
        idx = df['id'].to_list()
        texts = df['text'].to_list()
        predictions = []
        try:            
            for i, s, e in zip(idx,pipe_sentiment(data_stream(texts)),pipe_emotion(data_stream(texts))):
                  temp_dict = {'id':i}
                  temp_dict.update({ c['label']:c['score'] for c in s})
                  temp_dict.update({ c['label']:c['score'] for c in e})
                  predictions.append(temp_dict)
            
            pd.DataFrame(predictions).to_csv(writing_file,index=False,mode='a', header=not os.path.exists(writing_file))
        except Exception as e:
            print(e)
            idx_exceptions.extend(idx)
            
    print(f'{writing_file} : {datetime.now().strftime("%H:%M:%S")}')

pd.DataFrame(idx_exceptions).to_csv(writing_path+"idx_exceptions.csv")

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End:", current_time)