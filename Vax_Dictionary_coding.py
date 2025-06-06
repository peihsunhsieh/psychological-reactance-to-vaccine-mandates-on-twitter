# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 00:21:21 2022

@author: Pei-Hsun
"""

import re
import spacy
import spacymoji
import pandas as pd
import json
import concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import os
import traceback
from PHtexttool import TextAnalyzer


reading_path = ""
writing_path = ''
flog_path = ''

start_time = datetime.now()

start_time = start_time.strftime("%m %d %Y %H:%M:%S")
with open(flog_path, 'w', encoding='utf8') as flog:
    flog.write(f'Start Time: {start_time}\n')
    flog.flush()



# use spacy for tokenization ,'tagger','parser','ner'
nlp = spacy.load("en_core_web_sm",exclude=['ner'])
nlp.add_pipe('emoji', first=True)
# get default pattern for tokens that don't get split
re_token_match = spacy.tokenizer._get_regex_pattern(nlp.Defaults.token_match)
# add your patterns (here: hashtags and in-word hyphens)
re_token_match = f"({re_token_match}|[a-zA-Z]+\'[a-zA-Z]+|[a-zA-Z]+\'[a-zA-Z]+\'[a-zA-Z])"
# overwrite token_match function of the tokenizer
nlp.tokenizer.token_match = re.compile(re_token_match).match


with open('LIWC2015.json', 'r') as f:
  liwc = json.load(f)
  
with open('EmoLex.json', 'r') as f:
  EmoLex = json.load(f)

ta = TextAnalyzer(nlp)
ta.add_lexicon(liwc,'Anx', 'Anger', 'Sad', 'Family', 'Friend', 'Risk', 'FocusPast', 'FocusPresent','FocusFuture',name='liwc',wildcard=True,exist_check=False)
ta.add_lexicon(EmoLex,'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust',name='NRCLex')

pandemic_words = {'vaccine':['vaccin*'],
                  'mandate':['mandat*'],
                  'freedom':['freedom','liberty','rights','choice']}

ta.add_lexicon(pandemic_words,'vaccine','mandate','freedom',name='pandemic', wildcard=True,exist_check=True)



df_reader = pd.read_csv(reading_path, usecols = ['id','text'],chunksize=2000)
r_time_l = datetime.now()
for df in df_reader:
    r_time = datetime.now()
    with open(flog_path, 'a', encoding='utf8') as flog:
        flog.write(f'Time spending: {(r_time-r_time_l).total_seconds()}\n')
        flog.flush()
    r_time_l = r_time
    with ThreadPoolExecutor() as executor:
        temp_dic = {executor.submit(ta, text, id=i):i for i,text in zip(df['id'],df['text'])}
        results = []
        for future in as_completed(temp_dic):
            try:
                results.append(future.result(timeout=20)) 
            except Exception as e:
                with open(flog_path, 'a', encoding='utf8') as flog:
                    flog.write(f'failed id: {temp_dic[future]}, exception: {e}\n')
                    flog.flush()
                
    pd.DataFrame(results).to_csv(writing_path,index=False,mode='a', header=not os.path.exists(writing_path))


end_time = datetime.now()

end_time = end_time.strftime("%H:%M:%S")
with open(flog_path, 'w', encoding='utf8') as flog:
    flog.write(f'End Time: {end_time}\n')
    flog.flush()