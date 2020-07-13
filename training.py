import sys
from data import *
from models import *
from tqdm import tqdm
from transformers import BertModel, RobertaModel, BertTokenizer, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, random_split, DataLoader, IterableDataset, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from sklearn.model_selection import train_test_split
import io
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
import pickle
import time
import datetime
from sklearn.metrics import f1_score 
import sklearn 
import copy
import random
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def get_predicted(preds):
    pred_flat = np.argmax(preds, axis=1).flatten()
    return pred_flat

def evaluate(test_dataloader, nmodel):
    nmodel.eval()
    total_eval_accuracy=0
    y_preds = np.array([])
    y_test = np.array([])
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device).long()
        b_input_mask = batch[1].to(device).long()
        b_tokens = batch[2].to(device).long()
        b_features = batch[3].to(device).long()
        b_labels = batch[4].to(device).long()
        with torch.no_grad():        
            ypred = nmodel(b_input_ids, b_features, b_input_mask, b_tokens)
        loss = criterion(ypred, b_labels)
        total_loss += loss
        ypred = ypred.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(ypred, label_ids)
        ypred = get_predicted(ypred)
        y_preds = np.hstack((y_preds, ypred))
        y_test = np.hstack((y_test, label_ids))
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    macro_f1, micro_f1 = f1_score(y_test, y_preds, average='macro'), f1_score(y_test, y_preds, average='micro')
    return avg_val_accuracy, micro_f1, macro_f1, total_loss

def train(training_dataloader, validation_dataloader, nmodel, epochs = 4, lr1=2e-5, lr2=1e-4):
    total_steps = len(training_dataloader) * epochs
    bert = nmodel.embeddings
#    params = list(nmodel.linear.parameters())
    params = list(nmodel.linear.parameters())+list(nmodel.fc.parameters())+list(nmodel.final.parameters())

    optimizer1 = AdamW(bert.parameters(), lr=lr1, eps = 1e-8)
    optimizer2 = AdamW(params, lr=lr2, eps = 1e-8)
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    criterion = nn.CrossEntropyLoss()
    best_model = copy.deepcopy(nmodel)
    best_acc = 0
    best_micro = 0
    best_macro = 0
    for epoch_i in tqdm(range(0, epochs)):
        total_train_loss = 0
        nmodel.train()
        for step, batch in enumerate(training_dataloader):
            b_input_ids = batch[0].to(device).long()
            b_input_mask = batch[1].to(device).long()
            b_tokens = batch[2].to(device).long()
            b_features = batch[3].to(device).long()
            b_labels = batch[4].to(device).long()
            ypred = nmodel(b_input_ids, b_features, b_input_mask, b_tokens)
            loss = criterion(ypred, b_labels)
            if step%50==0:
                print('Loss = '+str(total_train_loss/(step+1.00)))
            total_train_loss += loss
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()

        print()

        print(f'Total Train Loss = {total_train_loss}')
        print('#############    Validation Set Stats')
        avg_val_accuracy, micro_f1, macro_f1, val_loss = evaluate(validation_dataloader, nmodel)
        print(f'Total Validation Loss = {val_loss}')
        print("  Accuracy: {0:.4f}".format(avg_val_accuracy))
        print("  Micro F1: {0:.4f}".format(micro_f1))
        print("  Macro F1: {0:.4f}".format(macro_f1))
        if macro_f1 > best_macro:
            best_model = copy.deepcopy(nmodel)
            best_macro = macro_f1
            best_acc = avg_val_accuracy
            best_micro = micro_f1

        print()

    return best_model, best_acc, best_micro, best_macro

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)
print(n_gpu)
print(torch.cuda.get_device_name(0))

# print(sys.argv)
folder, file_name = sys.argv[3].split('/')
print(folder, file_name)
# exit()
f = open(sys.argv[3], 'rb')
data = pickle.load(f)
f.close()

inputs, masks, tokens, labels, features = process_data(data)

lr = float(sys.argv[1])
hs = int(sys.argv[2])
file_name = file_name.replace('.pkl','')
f = open(folder + '/' + file_name +'.csv', 'a')
f.write('Hidden Size \t , Learning Rate \t , Accuracy \t , Micro F1 \t , Macro F1 \n')

k = 5
inputs, masks, tokens, labels, features = sklearn.utils.shuffle(inputs, masks, tokens, labels, features, random_state=42)
kf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
kf.get_n_splits(inputs, labels)

total_acc = 0
total_micro = 0
total_macro = 0

D_in, hidden_size,num_labels, feature_dim = 768, hs, 2, features.shape[1]

if len(file_name.split('_'))==2 : feature_dim = 0
if file_name.split('_')[0]=='data' : num_labels = 3
batch_size = 32
print(D_in, hidden_size,num_labels, feature_dim)
# exit()
nmodel = BERT_Linear_Feature(hidden_size, D_in, num_labels, feature_dim).to(device)

for train_index, test_index in kf.split(inputs, labels):
    training_inputs = torch.tensor(inputs[train_index])
    test_inputs = torch.tensor(inputs[test_index])

    training_labels = torch.tensor(labels[train_index])
    test_labels = torch.tensor(labels[test_index])

    training_masks = torch.tensor(masks[train_index])
    test_masks = torch.tensor(masks[test_index])

    training_features = torch.tensor(features[train_index])
    test_features = torch.tensor(features[test_index])

    training_tokens = torch.tensor(tokens[train_index])
    test_tokens = torch.tensor(tokens[test_index])

    # Create an iterator of our data with torch DataLoader 
    training_data = TensorDataset(training_inputs, training_masks, training_tokens,training_features, training_labels)
    training_sampler = RandomSampler(training_data)
    training_dataloader = DataLoader(training_data, sampler=training_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_masks,test_tokens, test_features, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    # print(training_data[0])
    
    
    # nmodel = BERT_Linear( D_in, num_labels, feature_dim).to(device)

    best_model, acc, micro, macro = train(training_dataloader, test_dataloader, copy.deepcopy(nmodel), epochs = 4, lr2=lr)
    total_acc += acc
    total_micro += micro
    total_macro += macro
    
print("==================FINAL RESULTS====================")
print("  Accuracy: {0:.4f}".format(total_acc/k))
print("  Micro F1: {0:.4f}".format(total_micro/k))
print("  Macro F1: {0:.4f}".format(total_macro/k))

acc, micro, macro = (total_acc/k), (total_micro/k), (total_macro/k)
f.write(str(hs)+'\t , ' + str(lr) + '\t , '+str(acc)+'\t , '+str(micro)+'\t , '+str(macro)+'\n')
