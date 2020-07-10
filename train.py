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
from tqdm import tqdm

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(n_gpu)
print(torch.cuda.get_device_name(1))

def normalize(x, m, s): return (x-m)/s

def conv_np(data):
    blank = np.zeros((len(data), len(data[0])))
    for i in range(len(data)):
        blank[i] = np.asarray([float(f) for f in data[i]])
    return blank

def tokenize(sentences, use_type_tokens = True, padding = True):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    input_ids = []
    attention_masks = []
    token_type_ids = []
    max_len = 128
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(sent,
                                                add_special_tokens=True,
                                                max_length=max_len, 
                                                pad_to_max_length=padding, 
                                                return_attention_mask = True,
                                                return_tensors = 'pt', 
                                                return_token_type_ids = use_type_tokens,
                                                truncation = True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        if use_type_tokens :
            token_type_ids.append(encoded_dict['token_type_ids'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    if use_type_tokens :
        token_type_ids = torch.cat(token_type_ids, dim=0)

    if use_type_tokens :
        return (input_ids, attention_masks, token_type_ids)

def get_dataset(data):
    input_ids, attention_masks, token_type_ids = tokenize(data['clean_devanagari'])
    signals = list(data['signal'])
    features = torch.Tensor(signals)
    labels = torch.Tensor(list(data['sentiment']))
    dataset = TensorDataset(input_ids, attention_masks, token_type_ids, features, labels)
    return dataset

def get_dataloader(dataset, batch_size = 32, train=True, split=True):
    if train:
        dataloader = DataLoader(
            dataset,
            sampler=RandomSampler(dataset),   
            batch_size=batch_size,
            num_workers=8
        )
    else:
        dataloader = DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),  
            batch_size=batch_size,
            num_workers=8
        )
    return dataloader

class BERT_Linear(torch.nn.Module):
    def __init__(self, D_in, num_labels, feature_dim):
        super(BERT_Linear, self).__init__()
        self.embeddings = BertModel.from_pretrained('bert-base-multilingual-cased',  output_hidden_states = True)
        self.linear = nn.Linear(D_in, num_labels, bias = True)

    def forward(self, x, x_feature, x_mask, tokens):
        embeddings = self.embeddings(x,x_mask, tokens)[1]
        y_pred = (self.linear(embeddings))
        return y_pred

class BERT_Linear_Feature(torch.nn.Module):
    def __init__(self, D_in, num_labels, feature_dim):
        super(BERT_Linear_Feature, self).__init__()
        self.embeddings = BertModel.from_pretrained('bert-base-multilingual-cased',  output_hidden_states = True)
        self.linear = nn.Linear(D_in+feature_dim, num_labels, bias = True)
        self.fc = nn.Linear(feature_dim, feature_dim, bias = True)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, x_feature, x_mask, token):
        embeddings = self.embeddings(x,x_mask, token)[1]
        feat = F.relu(self.fc(x_feature.float()))
        feat = self.dropout(feat)
        embed = torch.cat([embeddings, feat], 1)
        y_pred = (self.linear(embed))
        return y_pred

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def get_predicted(preds):
    pred_flat = np.argmax(preds, axis=1).flatten()
    return pred_flat

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def evaluate(test_dataloader, nmodel):
    nmodel.eval()
    total_eval_accuracy=0
    y_preds = np.array([])
    y_test = np.array([])
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device).long()
        b_input_mask = batch[1].to(device).long()
        b_tokens = batch[2].to(device).long()
        b_features = batch[3].to(device).long()
        b_labels = batch[4].to(device).long()
        with torch.no_grad():        
            ypred = nmodel(b_input_ids, b_features, b_input_mask, b_tokens)
        ypred = ypred.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(ypred, label_ids)
        ypred = get_predicted(ypred)
        y_preds = np.hstack((y_preds, ypred))
        y_test = np.hstack((y_test, label_ids))
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    macro_f1, micro_f1 = f1_score(y_test, y_preds, average='weighted'), f1_score(y_test, y_preds, average='micro')
    return avg_val_accuracy, micro_f1, macro_f1

def train(training_dataloader, validation_dataloader, nmodel, epochs = 5, lr1=2e-5, lr2=1e-4):
    total_steps = len(training_dataloader) * epochs
    bert = nmodel.embeddings
    params = list(nmodel.linear.parameters())
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
        nmodel.eval()
        total_eval_accuracy = 0
        nb_eval_steps = 0
        y_preds = np.array([])
        y_test = np.array([])
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device).long()
            b_input_mask = batch[1].to(device).long()
            b_tokens = batch[2].to(device).long()
            b_features = batch[3].to(device).long()
            b_labels = batch[4].to(device).long()
            with torch.no_grad():        
                ypred = nmodel(b_input_ids, b_features, b_input_mask, b_tokens)
            ypred = ypred.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(ypred, label_ids)
            ypred = get_predicted(ypred)
            y_preds = np.hstack((y_preds, ypred))
            y_test = np.hstack((y_test, label_ids))
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        macro_f1, micro_f1 = f1_score(y_test, y_preds, average='weighted'), f1_score(y_test, y_preds, average='micro')
        print("  Accuracy: {0:.4f}".format(avg_val_accuracy))
        print("  Micro F1: {0:.4f}".format(micro_f1))
        print("  Macro F1: {0:.4f}".format(macro_f1))
        if macro_f1 > best_macro:
            best_model = copy.deepcopy(nmodel)
            best_macro = macro_f1
            best_acc = avg_val_accuracy
            best_micro = micro_f1

    return best_model, best_acc, best_micro, best_macro

if __name__=="__main__":
    batch_size = int(sys.argv[1])
    filenames = ['sentiment_dataset_23.pkl']
    for fil in filenames:
        folder = './Sentiment_Datasets/'
        filename = folder + fil
        f = open(filename, 'rb')
        train_data, validation_data, test_data = pickle.load(f)
        f.close()

        train_signal = train_data['signal'].values
        train_signal = conv_np(train_signal)
        m,s = train_signal.mean(axis=0), train_signal.std(axis=0)
        train_signal = normalize(train_signal, m, s)
        train_data['signal'] = list(train_signal)

        val_signal = validation_data['signal'].values
        val_signal = conv_np(val_signal)
        val_signal = normalize(val_signal, m,s)
        validation_data['signal'] = list(val_signal)

        test_signal = test_data['signal'].values
        test_signal = conv_np(test_signal)
        test_signal = normalize(test_signal, m,s)
        test_data['signal'] = list(test_signal)

        train_dataset = get_dataset(train_data)
        val_dataset = get_dataset(validation_data)
        test_dataset = get_dataset(test_data)

        training_dataloader = get_dataloader(train_dataset, batch_size=batch_size)
        validation_dataloader = get_dataloader(val_dataset, batch_size=batch_size, train=False)
        test_dataloader = get_dataloader(test_dataset, batch_size = batch_size, train = False)

        results = {}
        lr1s = [2e-5, 3e-5, 5e-5]
        lr2s = [5e-3, 1e-3, 1e-4]

        f = open('Sentiment_results.csv', 'a')
        f.write('batch_size,lr1,lr2,test_acc,test_micro,test_macro,val_acc,val_micro,val_macro\n')

        for lr1 in lr1s:
            for lr2 in lr2s:
                D_in, hidden_size,num_labels, feature_dim = 768, 100, 3, train_dataset[0][3].shape[0]
                nmodel1 = BERT_Linear(D_in, num_labels, feature_dim)
                nmodel1.to('cpu')
                nmodel = copy.deepcopy(nmodel1)
                nmodel.to(device)
                # nmodel = torch.nn.DataParallel(nmodel, device_ids = [1,0])
                best_model, best_acc, best_micro, best_macro = train(training_dataloader, validation_dataloader, (nmodel), epochs = 6, lr1=lr1, lr2=lr2)
                acc, micro, macro = evaluate(test_dataloader, best_model)   
                print((acc,micro,macro))
                results[(lr1,lr2)]=(batch_size,acc, micro, macro, best_acc, best_micro, best_macro)
                f.write(str(batch_size)+','+str(lr1)+','+str(lr2)+','+str(acc)+','+str(micro)+','+str(macro)+','+str(best_acc)+','+str(best_micro)+','+str(best_macro)+'\n')

        f.close()
        
        f = open('Sentiment_results_dic.pkl', 'wb')
        pickle.dump(results, f)
        f.close()
        # for (lr,size) in results.keys():
        #     acc, micro, macro, best_acc, best_micro, best_macro = results[(lr,size)]
        #     f.write(str(lr)+','+str(size)+','+str(acc)+','+str(micro)+','+str(macro)+str(best_acc)+','+str(best_micro)+','+str(best_macro)+',\n')
        

