# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 00:32:14 2022

@author: Pei-Hsun Hsieh
"""

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import urllib
import requests
requests.packages.urllib3.disable_warnings()
from collections import Counter
import spacymoji


class TextAnalyzer:
  def __init__(self, tokenizer, vader=True, url='domain',userhandle='collect',emoji='collect'):
    self.tokenizer = tokenizer
    self.vader=vader
    self.num_lex = 0
    self.num_text_exist = 0
    self.url=url
    self.userhandle=userhandle
    self.emoji = emoji
    self.lexi_param={}
    self.lexicons={}
    self.text_exist_list={}
    if self.vader==True:
      self.vader_analyzer = SentimentIntensityAnalyzer()

  def vader_analysis(self,text):
    vader_scores = self.vader_analyzer.polarity_scores(text)
    vader_scores = {'vader_'+key:value for key, value in vader_scores.items()}

    return vader_scores      

  def tokenize(self,text):
    tokens = []
    # if self.url=='domain' or self.url=='keep':
    urls = []
    # if self.userhandle=='collect':
    userhandles = []
    # if self.emoji=='collect':
    emojis = []
    for token in self.tokenizer(text):
      # emoji
      if token._.is_emoji:
        if self.emoji=='collect':
          emojis.append(token.text)
        elif self.emoji=='demojize':
          tokens.append(token._.emoji_desc)
        else:
          tokens.append(token.text)

      # user mentions
      elif token.text.startswith('@'):
        if self.userhandle=='collect':
          userhandles.append(token.text)
        elif self.userhandle=='remove':
          pass
        elif self.userhandle=='keep':
          tokens.append(token.text)
      # URL
      elif token.like_url:
        if self.url=='domain':
          try:
            urls.append(urllib.parse.urlparse(requests.get(token.text, verify=False, timeout=10).url).netloc)
          except:
            urls.append("URL_NA")
        else:
          pass
      # skip stopwords and empty strings, punctuations and numbers
      elif token.is_punct or token.like_email or token.like_num or (token.is_ascii is False):
        pass
      elif token.text:
        tokens.append(token.lemma_)
    word_count = len(tokens)

    return tokens, word_count, userhandles, urls, emojis

  def add_lexicon(self,lexicon,*args,name=None,wildcard=False,exist_check=False):
    self.num_lex += 1
    name=name if name!=None else 'dict'+str(self.num_lex)
    self.lexi_param.update({name:(wildcard,exist_check)})
    if wildcard:
      self.lexicons.update({name:{category:re.compile('|'.join(['(^'+w[:-1]+'.*)' if re.match('.*\*$', w) else '('+w+')' for w in lexicon[category]])) for category in args}})
    else:
      self.lexicons.update({name:{category:lexicon[category] for category in args}})

  def lexicon_matcher(self,tokens,lexi_name):
    wildcard = self.lexi_param[lexi_name][0]
    exist_check = self.lexi_param[lexi_name][1]
    lexi_score = {}
    try:
        if wildcard:
          for category, word_regex in self.lexicons[lexi_name].items():
            if exist_check:
              lexi_score.update({lexi_name+'_'+category:any([bool(word_regex.match(token.lower())) for token in tokens])})
            else:
              lexi_score.update({lexi_name+'_'+category:sum([bool(word_regex.match(token.lower())) for token in tokens])/len(tokens)})
        else:
          tokens = [token.lower() for token in tokens]
          for category, word_list in self.lexicons[lexi_name].items():
            if exist_check:
              lexi_score.update({str(lexi_name)+'_'+str(category):not set(tokens).isdisjoint(set(word_list))})
            else:
              token_count = Counter(tokens)
              lexi_score.update({str(lexi_name)+'_'+str(category):sum(Counter({ key:value for key, value in token_count.items() if key in word_list}).values())/len(tokens)})
    finally:
        return lexi_score

  def __call__(self, text, rep_tokens=False, **kwargs):
    variables = kwargs
    if self.vader:
      variables.update(self.vader_analysis(text))
    tokens, word_count, userhandles, urls, emojis = self.tokenize(text)
    if len(tokens)!=0:
        for lexi_name in self.lexi_param.keys():
          variables.update(self.lexicon_matcher(tokens,lexi_name))
    if rep_tokens:
      variables.update({'word_count':word_count,'userhandles':';'.join(userhandles),'urls':';'.join(urls),'emojis':';'.join(emojis),'tokens':tokens})
    else:
      variables.update({'word_count':word_count,'userhandles':';'.join(userhandles),'urls':';'.join(urls),'emojis':';'.join(emojis)})

    return variables