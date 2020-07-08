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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
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
    max_len = 512
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

def get_dataset(data):
    input_ids, attention_masks, token_type_ids = tokenize(data['clean_devanagari'])
    signals = list(data['signal'])
    features = torch.Tensor(signals)
    labels = torch.Tensor(list(data['sentiment']))
    dataset = TensorDataset(input_ids, attention_masks, token_type_ids, features, labels)
    return dataset

def get_dataloader(dataset, batch_size = 8, train=True, split=True):
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

class Bert(nn.Module) :
    def __init__(self) :
        super(Bert, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased',  output_hidden_states = True)
    def forward(self, input_ids, attn_masks, tokens):
        outputs = self.model(input_ids, token_type_ids=tokens, attention_mask=attn_masks)
        outputs = outputs[1]
        return outputs

class BERT_Linear(torch.nn.Module):
    def __init__(self, D_in, num_labels, feature_dim):
        super(BERT_Linear, self).__init__()
        self.embeddings = Bert()
        self.linear1 = nn.Linear(D_in, num_labels, bias = True)

    def forward(self, x, x_feature, x_mask, tokens):
        embeddings = self.embeddings(x,x_mask, tokens)
        y_pred = (self.linear1(embeddings))
        return y_pred

class BERT_Linear_Feature(torch.nn.Module):
    def __init__(self, hidden_size, D_in, num_labels, feature_dim):
        super(BERT_Linear_Feature, self).__init__()
        self.embeddings = Bert()
        self.linear1 = nn.Linear(D_in, hidden_size, bias = True)
        self.linear2 = nn.Linear(hidden_size+feature_dim, num_labels, bias = True)
        # self.linear3 = nn.Linear(feature_dim, reduced_size, bias = True)
        self.fc1 = nn.Linear(feature_dim, feature_dim, bias = True)
        self.fc2 = nn.Linear(feature_dim, feature_dim, bias = True)
        # self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x, x_feature, x_mask, token):
        # hidden_states = self.embeddings(x, x_mask)
        # embed = torch.cat([hidden_states, x_feature.float()], 1)
        embeddings = self.embeddings(x,x_mask, token)
        hidden_states = self.linear1(embeddings)
        feat = F.relu(self.fc1(x_feature.float()))
        feat = F.relu(self.fc2(feat))
        # reduced = self.linear3(x_feature.float())
        # hidden_states = self.dropout(hidden_states)
        hidden_states = F.relu(hidden_states)
        embed = torch.cat([hidden_states, feat], 1)
        y_pred = (self.linear2(embed))
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

def train(training_dataloader, validation_dataloader, nmodel, epochs = 5, lr = 2e-5):
    total_steps = len(training_dataloader) * epochs
    optimizer = AdamW(nmodel.parameters(),
                  lr = lr, # args.learning_rate - default is 5e-5, 
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
    scheduler = get_linear_schedule_with_warmup(optimizer, 
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
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nmodel.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
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
    filenames = ['sentiment_dataset_23.pkl', 'sentiment_dataset_15.pkl', 'sentiment_dataset_9.pkl', 'sentiment_dataset_23_with_sel.pkl', 'sentiment_dataset_29_with_sel.pkl']
    for fil in filenames:
        folder = './Sentiment_Datasets/'
        filename = folder + fil
        f = open(filename, 'rb')
        train_data, validation_data, test_data = pickle.load(f)
        f.close()

        train_dataset = get_dataset(train_data)
        val_dataset = get_dataset(validation_data)
        test_dataset = get_dataset(test_data)

        training_dataloader = get_dataloader(train_dataset, batch_size=batch_size)
        validation_dataloader = get_dataloader(val_dataset, batch_size=batch_size, train=False)
        test_dataloader = get_dataloader(test_dataset, batch_size = batch_size, train = False)

        results = {}
        learning_rates = [1.5e-5, 2e-5, 2.5e-5]
        learning_rates = [2e-5]
        hidden_sizes = [25,50,100,200,500]
        for lr in learning_rates:
            for size in hidden_sizes:
                D_in, hidden_size,num_labels, feature_dim = 768, size, 3, train_dataset[0][3].shape[0]
                nmodel = BERT_Linear_Feature( hidden_size, D_in, num_labels, feature_dim).to(device)
                best_model, best_acc, best_micro, best_macro = train(training_dataloader, validation_dataloader, copy.deepcopy(nmodel), epochs = 8, lr=lr)
                acc, micro, macro = evaluate(test_dataloader, best_model)   
                print((acc,micro,macro))
                results[(lr,size)]=(acc, micro, macro)

        f = open(fil+'results_dic.pkl', 'wb')
        pickle.dump(results, f)
        f.close()

        f = open(fil+'results.csv', 'w')
        f.write('lr,hidden_size,acc,micro, macro\n')
        for (lr,size) in results.keys():
            acc, micro, macro = results[(lr,size)]
            f.write(str(lr)+','+str(size)+','+str(acc)+','+str(micro)+','+str(macro)+',\n')
        f.close()

