from data import *
from models import *
from utils import *
import sys
import os
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import sklearn 
from sklearn.metrics import f1_score 
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
ids, inputs, masks, tokens, labels, features = sklearn.utils.shuffle(inputs, masks, tokens, labels, features, random_state=42)
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

    training_ids = torch.tensor(ids[train_index])
    test_ids = torch.tensor(ids[test_index])

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
