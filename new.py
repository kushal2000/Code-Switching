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

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)
print(n_gpu)
print(torch.cuda.get_device_name(1))

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

def train(training_dataloader, validation_dataloader, test_dataloader, nmodel, epochs = 5, lr1=2e-5, lr2=2e-4):
    total_steps = len(training_dataloader) * epochs
    bert = nmodel.embeddings
#    params = list(nmodel.linear.parameters())
    params = list(nmodel.linear.parameters())+list(nmodel.fc.parameters())
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
        
        print('#############    Test Set Stats')
        avg_val_accuracy, micro_f1, macro_f1, test_loss = evaluate(test_dataloader, nmodel)
        print(f'Total Test Loss = {test_loss}')
        print("  Accuracy: {0:.4f}".format(avg_val_accuracy))
        print("  Micro F1: {0:.4f}".format(micro_f1))
        print("  Macro F1: {0:.4f}".format(macro_f1))

        print()

    return best_model, best_acc, best_micro, best_macro

if __name__=="__main__":
    batch_size = int(sys.argv[1])
    filenames = ['humour_dataset_23.pkl','humour_dataset_9.pkl','humour_dataset_29_with_sel.pkl']
    #filenames = ['humour_dataset_23.pkl']
    for fil in filenames:
        folder = './Humour_Datasets/'
        filename = folder + fil
        f = open(filename, 'rb')
        data = pickle.load(f)
        f.close()

        signals = data['signal'].values
        labels = data['sentiment'].values
        blank = np.zeros((len(signals),len(signals[0])))
        for i in range(len(signals)):
            blank[i] = np.asarray(signals[i])
        signals = blank
        inputs, masks, tokens = tokenize(data['clean_devanagari'])
        inputs = np.asarray(inputs)
        masks = np.asarray(masks)
        tokens = np.asarray(tokens)
        labels = np.asarray(data['sentiment'])
        features = signals

        results = {}
        lr1s = [2e-5, 3e-5]
        lr2s = [1e-3, 1e-4]

        f = open(fil+'results.csv', 'a')
        f.write('batch_size,lr1,lr2,test_acc,test_micro,test_macro,val_acc,val_micro,val_macro\n')

        #inputs, masks, tokens, labels, features = sklearn.utils.shuffle(inputs, masks, tokens, labels, features, random_state=42)
        inputs_train, inputs_test, masks_train, masks_test, tokens_train, tokens_test, labels_train, labels_test, features_train, features_test = train_test_split(inputs, masks,tokens,labels, features,stratify=labels, test_size=0.1, random_state=42)
        inputs_train, inputs_val, masks_train, masks_val, tokens_train, tokens_val, labels_train, labels_val, features_train, features_val = train_test_split(inputs_train, masks_train,tokens_train,labels_train, features_train,stratify=labels_train, test_size=0.11, random_state=42)
        for lr1 in lr1s:
            for lr2 in lr2s:
                training_inputs = torch.tensor(inputs_train)
                validation_inputs = torch.tensor(inputs_val)
                test_inputs = torch.tensor(inputs_test)

                training_labels = torch.tensor(labels_train)
                validation_labels = torch.tensor(labels_val)
                test_labels = torch.tensor(labels_test)

                training_masks = torch.tensor(masks_train)
                validation_masks = torch.tensor(masks_val)
                test_masks = torch.tensor(masks_test)

                training_features = torch.tensor(features_train)
                validation_features = torch.tensor(features_val)
                test_features = torch.tensor(features_test)

                training_tokens = torch.tensor(tokens_train)
                validation_tokens = torch.tensor(tokens_val)
                test_tokens = torch.tensor(tokens_test)

                # Create an iterator of our data with torch DataLoader 
                training_data = TensorDataset(training_inputs, training_masks, training_tokens,training_features, training_labels)
                training_sampler = RandomSampler(training_data)
                training_dataloader = DataLoader(training_data, sampler=training_sampler, batch_size=batch_size)

                validation_data = TensorDataset(validation_inputs, validation_masks,validation_tokens, validation_features, validation_labels)
                validation_sampler = SequentialSampler(validation_data)
                validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

                test_data = TensorDataset(test_inputs, test_masks,test_tokens, test_features, test_labels)
                test_sampler = SequentialSampler(test_data)
                test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
                # print(training_data[0])
                D_in, hidden_size,num_labels, feature_dim = 768, 100, 3, training_data[0][3].shape[0]
                nmodel = BERT_Linear_Feature( hidden_size, D_in, num_labels, feature_dim).to(device)
               # nmodel = BERT_Linear( D_in, num_labels, feature_dim).to(device)

                best_model, best_acc, best_micro, best_macro = train(training_dataloader, validation_dataloader, copy.deepcopy(nmodel), epochs = 4, lr1=lr1, lr2=lr2)
                acc, micro, macro = evaluate(test_dataloader, best_model)   
                print((acc,micro,macro))
                results[(lr1,lr2)]=(batch_size,acc, micro, macro, best_acc, best_micro, best_macro)
                f.write(str(batch_size)+','+str(lr1)+','+str(lr2)+','+str(acc)+','+str(micro)+','+str(macro)+','+str(best_acc)+','+str(best_micro)+','+str(best_macro)+'\n')
        f.close()
