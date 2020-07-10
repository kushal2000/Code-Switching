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

def run_feature_selection(signal_feature, feature_list, labels):
    feature_dict =  {}
    for j, i in enumerate(feature_list):
        feature_dict[i]=j
    results = {}
    # BORUTA
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=1, random_state=1)
    feat_selector.fit(signal_feature, labels)
    df = pd.DataFrame(np.array([feature_list, feat_selector.support_, feat_selector.ranking_]).T, columns=['feature', 'T/F', 'Rank'])
    print(df.sort_values(by=['Rank']))
    print('------ BORUTA ------')
    return feat_selector

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

    
    #TODO: Pass dictionary instead of tuple
    if use_type_tokens :
        # print("input ids: {} attention_masks: {} token_type_ids: {}".format(input_ids.shape, attention_masks.shape, token_type_ids.shape))
        return (input_ids, attention_masks, token_type_ids)

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
    def __init__(self, hidden_size, D_in, num_labels, feature_dim):
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

def train(training_dataloader, validation_dataloader, nmodel, epochs = 5, lr1=2e-5, lr2=2e-4):
    total_steps = len(training_dataloader) * epochs
    bert = nmodel.embeddings
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
        macro_f1, micro_f1 = f1_score(y_test, y_preds, average='macro'), f1_score(y_test, y_preds, average='micro')
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
    filenames = ['hate_dataset_23.pkl', 'hate_dataset_15.pkl', 'hate_dataset_9.pkl', 'hate_dataset_29_with_sel.pkl', 'hate_dataset_23_with_sel.pkl']
    for fil in filenames:
        folder = './Hate_Datasets/'
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
        lr1s = [2e-5, 3e-5, 5e-5]
        lr2s = [1e-3, 1e-4]

        f = open(fil+'results.csv', 'a')
        f.write('batch_size,lr1,lr2,test_acc,test_micro,test_macro\n')

        for lr1 in lr1s:
            for lr2 in lr2s:
                k = 10
                inputs, masks, tokens, labels, features = sklearn.utils.shuffle(inputs, masks, tokens, labels, features, random_state=42)
                kf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
                kf.get_n_splits(inputs, labels)
                total_acc = 0
                total_micro = 0
                total_macro = 0
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
                    D_in, hidden_size,num_labels, feature_dim = 768, 100, 3, training_data[0][3].shape[0]
                    nmodel = BERT_Linear_Feature( hidden_size, D_in, num_labels, feature_dim).to(device)
                    # nmodel = BERT_Linear( D_in, num_labels, feature_dim).to(device)

                    best_model, acc, micro, macro = train(training_dataloader, test_dataloader, copy.deepcopy(nmodel), epochs = 6, lr1=lr1, lr2=lr2)
                    print("  Accuracy: {0:.4f}".format(acc))
                    print("  Micro F1: {0:.4f}".format(micro))
                    print("  Macro F1: {0:.4f}".format(macro))
                    total_acc += acc
                    total_micro += micro
                    total_macro += macro
                print("==================FINAL RESULTS====================")
                print("  Accuracy: {0:.4f}".format(total_acc/k))
                print("  Micro F1: {0:.4f}".format(total_micro/k))
                print("  Macro F1: {0:.4f}".format(total_macro/k))
                acc, micro, macro = (total_acc/k), (total_micro/k), (total_macro/k)
                f.write(str(batch_size)+','+str(lr1)+','+str(lr2)+','+str(acc)+','+str(micro)+','+str(macro)+'\n')
        f.close()
