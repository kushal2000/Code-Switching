from transformers import BertModel, RobertaModel, BertTokenizer
import numpy as np
import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, random_split, DataLoader, IterableDataset, ConcatDataset
from sklearn.preprocessing import normalize

def tokenize(sentences, use_type_tokens = True, padding = True, max_len = 128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    input_ids = []
    attention_masks = []
    token_type_ids = []
    max_len = max_len
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

def process_data(data, num_features=22, mode = 'dev'):
    signals = data['signal'].values
    labels = data['sentiment'].values
    blank = np.zeros((len(signals),len(signals[0])))
    for i in range(len(signals)):
        blank[i] = np.asarray(signals[i])
    signals = blank
    if signals.shape[1]==29:
        if num_features == 22:
            idxs = np.array([1,2,3,4,5,8,9,10,11,12,13,14,17,18,19,20,21,22,25,26,27,28])
            signals = signals[:,idxs]
        if num_features == 9:
            signals = signals[:,6:15]
    if mode == 'dev':
        inputs, masks, tokens = tokenize(data['clean_devanagari'])
    else:
        inputs, masks, tokens = tokenize(data['clean_romanized'])
    inputs = np.asarray(inputs)
    masks = np.asarray(masks)
    tokens = np.asarray(tokens)
    labels = np.asarray(data['sentiment'])
    ids = np.arange(len(labels))
    return ids, inputs, masks, tokens, labels, signals 