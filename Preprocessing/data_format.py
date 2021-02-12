import os
import pickle
import numpy as np
import pandas as pd

datasets = ['Datasets/Humour/data_frame_22.pkl', 'Datasets/Sarcasm/data_frame_22.pkl', 'Datasets/Sentiment/data_frame_22.pkl', 'Datasets/Hate/data_frame_22.pkl']
datasets = ['Datasets/Humour/data_frame_22.pkl', 'Datasets/Sarcasm/data_frame_22.pkl', 'Datasets/Hate/data_frame_22.pkl']

with open('lang_dict.pkl', 'rb') as f:
    lang_dict = pickle.load(f)
# print(lang_dict['en'])

for dataset in datasets:
    with open(dataset, 'rb') as f:
        data = pickle.load(f)
    print(data.columns)
    print(data['clean_devanagari'])
