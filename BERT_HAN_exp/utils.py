from data import *
from models import *
from transformers import BertModel, RobertaModel, BertTokenizer, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from sklearn.metrics import f1_score 
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, random_split, DataLoader, IterableDataset, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from tqdm import tqdm
import pandas as pd
import numpy as np
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    for batch in test_dataloader:
        b_input_ids = batch[0].to(device).long()
        b_input_mask = batch[1].to(device).long()
        b_tokens = batch[2].to(device).long()
        b_features = batch[3].to(device).long()
        b_labels = batch[4].to(device).long()
        with torch.no_grad():        
            ypred, _att_weight = nmodel(b_input_ids, b_features, b_input_mask, b_tokens)
        ypred = ypred.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(ypred, label_ids)
        ypred = get_predicted(ypred)
        y_preds = np.hstack((y_preds, ypred))
        y_test = np.hstack((y_test, label_ids))
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    macro_f1, micro_f1 = f1_score(y_test, y_preds, average='macro'), f1_score(y_test, y_preds, average='micro')
    print("  Accuracy: {0:.4f}".format(avg_val_accuracy))
    print("  Micro F1: {0:.4f}".format(micro_f1))
    print("  Macro F1: {0:.4f}".format(macro_f1))

def train(training_dataloader, validation_dataloader, nmodel, epochs = 5):
    total_steps = len(training_dataloader) * epochs
    bert_params = nmodel.embeddings
    bert_named_params = ['embeddings.'+name_ for name_, param_ in bert_params.named_parameters()]
    model_named_params = [name_ for name_, param_ in nmodel.named_parameters()]
    other_named_params = [i for i in model_named_params if i not in bert_named_params]
    params = []
    for name, param in nmodel.named_parameters():
        if name in other_named_params:
            params.append(param)
    
    optimizer1 = AdamW(bert_params.parameters(), lr=2e-5, eps = 1e-8)
    optimizer2 = AdamW(params, lr=1e-4, eps = 1e-8)
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()
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
            ypred, _att_weight = nmodel(b_input_ids, b_features, b_input_mask, b_tokens)
            loss = criterion(ypred, b_labels)
            if step%50==0:
                print('Loss = '+str(total_train_loss/(step+1.00)))
            total_train_loss += loss
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_params.parameters(), 1.0)
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
                ypred, _att_weight = nmodel(b_input_ids, b_features, b_input_mask, b_tokens)
                if epoch_i==epochs-1:
                    ATTENTION_WEIGHTS.append(_att_weight)
                    IDS_CORRESPONDING_TO_ATT.append(b_labels)
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