from transformers import BertModel, RobertaModel, BertTokenizer, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, random_split, DataLoader, IterableDataset, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BERT_Linear_Feature(torch.nn.Module):
    def __init__(self, hidden_size, D_in, num_labels, feature_dim):
        super(BERT_Linear_Feature, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.embeddings = BertModel.from_pretrained('bert-base-multilingual-cased',  output_hidden_states = True)
        self.linear = nn.Linear(D_in, max(num_labels, hidden_size), bias = True)
        self.fc = nn.Linear(max(feature_dim,1) , max(feature_dim,1) , bias = True)
        self.final = nn.Linear(max(hidden_size + feature_dim, 1), num_labels, bias = True)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, x_feature, x_mask, token):
        embed = self.embeddings(x,x_mask, token)[1]

        if self.hidden_size == 0 and self.feature_dim==0:
            y_pred = (self.linear(embed))
            return y_pred

        if self.feature_dim == 0:
            hidden = self.linear(embed)
            hidden = self.dropout(hidden)
            y_pred = self.final(hidden)
            return y_pred

        if self.feature_dim>0:
            feat = F.relu(self.fc(x_feature.float()))
            feat = self.dropout(feat)
            if self.hidden_size > 0:
                embed = self.linear(embed)
                embed = self.dropout(embed)
            embed = torch.cat([embed, feat], 1)

        if self.hidden_size == 0:
            y_pred = (self.linear(embed))
            return y_pred
            
        y_pred = self.final(embed)
        return y_pred