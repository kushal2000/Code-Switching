from transformers import BertModel, RobertaModel, BertTokenizer, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, random_split, DataLoader, IterableDataset, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

class Fusion_Net(torch.nn.Module):
    def __init__(self, hidden_size, D_in, num_labels, feature_dim):
        super(Fusion_Net, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.embeddings = BertModel.from_pretrained('bert-base-multilingual-cased',  output_hidden_states = True)
        if hidden_size==0:
            self.linear = nn.Linear(D_in+feature_dim, max(num_labels, hidden_size), bias = True)
        else:
            self.linear = nn.Linear(D_in, max(num_labels, hidden_size), bias = True)
        self.fc = nn.Linear(max(feature_dim,1) , max(int(feature_dim),1) , bias = True)
        self.final = nn.Linear(max(hidden_size + int(feature_dim), 1), num_labels, bias = True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, x_feature, x_mask, token):
        embed = self.embeddings(x,x_mask, token)[1]

        if self.feature_dim == 0:
            hidden = F.relu(self.linear(embed))
            hidden = self.dropout(hidden)
            y_pred = self.final(hidden)
            return y_pred

        if self.feature_dim>0:
            feat = F.relu(self.fc(x_feature.float()))
            feat = self.dropout(feat)
            if self.hidden_size > 0:
                embed = F.relu(self.linear(embed))
                embed = self.dropout(embed)
            embed = torch.cat([embed, feat], 1)

        if self.hidden_size == 0:
            y_pred = (self.linear(embed))
            return y_pred

        y_pred = self.final(embed)
        return y_pred
        
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        
        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)
        
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))
        
    def forward(self, x, mask=None):
        feature_dim = self.feature_dim 
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim), 
            self.weight
        ).view(-1, step_dim)
        
        if self.bias:
            eij = eij + self.b
            
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        
        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1), a

class BERT_HAN(torch.nn.Module):
    def __init__(self, hidden_size, D_in, num_labels, feature_dim):
        super(BERT_HAN, self).__init__()
        self.embeddings = BertModel.from_pretrained('bert-base-multilingual-cased',  output_hidden_states = True)
        self.bert_encoder = nn.LSTM(input_size=D_in,
                                    hidden_size=hidden_size,
                                    num_layers=1, 
                                    bidirectional=True)
        self.attent = Attention(hidden_size*2, 128)
        self.linear = nn.Linear(hidden_size*2, num_labels, bias = True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, x_feature, x_mask, token):
        embeddings = self.embeddings(x,x_mask, token)[2][-1]
        bert_encoder, (h_n, c_n) = self.bert_encoder(embeddings)
        bert_encoder = self.dropout(bert_encoder)
        attent, attention_weight = self.attent(bert_encoder)
        y_pred = self.linear(attent)
        return y_pred, attention_weight

class BERT_HAN_feature_old(torch.nn.Module):
    def __init__(self, hidden_size, D_in, num_labels, feature_dim):
        super(BERT_HAN_feature_old, self).__init__()
        self.embeddings = BertModel.from_pretrained('bert-base-multilingual-cased',  output_hidden_states = True)
        self.bert_encoder = nn.LSTM(input_size=D_in,
                                    hidden_size=hidden_size,
                                    num_layers=1, 
                                    bidirectional=True)
        #self.bert_encoder.flatten_parameters()
        self.attent = Attention(hidden_size*2, 128)
        self.lin_feat = nn.Linear(feature_dim, feature_dim, bias = True)
        self.linear = nn.Linear(hidden_size*2+feature_dim, num_labels, bias = True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, x_feature, x_mask, token):
        embeddings = self.embeddings(x,x_mask, token)[2][-1]
        bert_encoder, (h_n, c_n) = self.bert_encoder(embeddings)
        bert_encoder = self.dropout(bert_encoder)
        #attent = bert_encoder[:,:1,:].squeeze()
        attent, attention_weight = self.attent(bert_encoder)
        feat = F.relu(self.lin_feat(x_feature.float()))
        # feat = self.dropout(feat)
        embed = torch.cat([attent, feat], 1)
        y_pred = self.linear(embed)
        return y_pred, attention_weight

class BERT_Linear_Feature(torch.nn.Module):
    def __init__(self, hidden_size, D_in, num_labels, feature_dim):
        super(BERT_Linear_Feature, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.embeddings = BertModel.from_pretrained('bert-base-multilingual-cased',  output_hidden_states = True)
        if hidden_size==0:
            self.linear = nn.Linear(D_in+feature_dim, max(num_labels, hidden_size), bias = True)
        else:
            self.linear = nn.Linear(D_in, max(num_labels, hidden_size), bias = True)
        self.fc = nn.Linear(max(feature_dim,1) , max(feature_dim,1) , bias = True)
        self.final = nn.Linear(max(hidden_size + feature_dim, 1), num_labels, bias = True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, x_feature, x_mask, token):
        embed = self.embeddings(x,x_mask, token)[1]

        if self.hidden_size == 0 and self.feature_dim==0:
            y_pred = (self.linear(embed))
            return y_pred

        if self.feature_dim == 0:
            hidden = F.relu(self.linear(embed))
            hidden = self.dropout(hidden)
            y_pred = self.final(hidden)
            return y_pred

        if self.feature_dim>0:
            feat = F.relu(self.fc(x_feature.float()))
            feat = self.dropout(feat)
            if self.hidden_size > 0:
                embed = F.relu(self.linear(embed))
                embed = self.dropout(embed)
            embed = torch.cat([embed, feat], 1)

        if self.hidden_size == 0:
            y_pred = (self.linear(embed))
            return y_pred

        y_pred = self.final(embed)
        return y_pred

class BERT_HAN_feature(torch.nn.Module):
    def __init__(self, hidden_size, D_in, num_labels, feature_dim):
        super(BERT_HAN_feature, self).__init__()
        self.embeddings = BertModel.from_pretrained('bert-base-multilingual-cased',  output_hidden_states = True)
        self.bert_encoder = nn.LSTM(input_size=D_in,
                                    hidden_size=hidden_size,
                                    num_layers=1, 
                                    bidirectional=True)
        self.bert_encoder_2 = nn.LSTM(input_size=hidden_size*2,
                                    hidden_size=feature_dim//2,
                                    num_layers=1, 
                                    bidirectional=True)
        self.attent = Attention(feature_dim, 128+1)
        self.lin_feat = nn.Linear(feature_dim, feature_dim, bias = True)
        self.linear = nn.Linear(feature_dim, num_labels, bias = True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, x_feature, x_mask, token):
        embeddings = self.embeddings(x,x_mask, token)[2][-1]
        bert_encoder, (h_n, c_n) = self.bert_encoder(embeddings)
        bert_encoder = self.dropout(bert_encoder)
        bert_encoder, (h_n, c_n) = self.bert_encoder_2(bert_encoder)
        bert_encoder = self.dropout(bert_encoder)
        
        feat = F.relu(self.lin_feat(x_feature.float()))
        feat = feat.view(feat.size()[0], 1, feat.size()[1])

        embed = torch.cat([bert_encoder, feat], 1)

        attent, attention_weight = self.attent(embed)
        y_pred = self.linear(attent)
        return y_pred, attention_weight
