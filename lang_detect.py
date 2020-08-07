import os
import pickle
import numpy as np
import pandas as pd

datasets = ['Datasets/Humour/data_frame_22.pkl', 'Datasets/Sarcasm/data_frame_22.pkl', 'Datasets/Sentiment/data_frame_22.pkl', 'Datasets/Hate/data_frame_22.pkl']


# print(data.head())

en_dict = {}
hi_dict = {}
def add_dict1(tweets):
    for tweet in tweets:
        # print(tweet)
        # break
        tweet = str(tweet)
        tweet = tweet.split()
        for word in tweet:
            if len(word.split('\\')) !=2:
                continue
            word, lang = word.split('\\')
            lang = lang.lower()
            if lang=='en':
                en_dict[word] = en_dict.get(word,0)+1
            if lang=='hi':
                hi_dict[word] = hi_dict.get(word,0)+1

def add_dict(tweets):
    for tweet in tweets:
        for word in tweet:
            if len(word.split('_')) !=2:
                continue
            word, lang = word.split('_')
            if lang=='en':
                en_dict[word] = en_dict.get(word,0)+1
            if lang=='hi':
                hi_dict[word] = hi_dict.get(word,0)+1

def check_ans(tweets):
    total = 0
    correct = 0
    not_in_dict = 0
    mistake = 0
    for tweet in tweets:
        for word in tweet:
            if len(word.split('_')) !=2:
                continue
            
            word, lang = word.split('_')
            if lang!='en' and lang!='hi': continue
            total+=1
            en_count = en_dict.get(word,0)
            hi_count = hi_dict.get(word,0)
            if en_count==0 and hi_count==0:
                not_in_dict+=1
                continue
            # elif en_count==hi_count:
            #     print(word)
            #     print(lang)
            #     print(hi_count)
            if lang=='en':
                if en_count>=hi_count: correct+=1
                else: print(word)
            if lang=='hi':
                if en_count<=hi_count: correct+=1
                # else: print(word)
    print(total)
    print(correct)
    print(not_in_dict)
    print(mistake)

with open(datasets[1], 'rb') as f:
    data = pickle.load(f)
add_dict(data['tweet'].values)

with open(datasets[3], 'rb') as f:
    data = pickle.load(f)
add_dict(data['tweet'].values)

with open(datasets[2], 'rb') as f:
    data = pickle.load(f)
add_dict1(data['tweet'].values)

with open(datasets[0], 'rb') as f:
    data = pickle.load(f)
add_dict(data['tweet'].values)

lang_dict = {'en': en_dict, 'hi': hi_dict}
with open('lang_dict.pkl', 'wb') as f:
    pickle.dump(lang_dict, f)
