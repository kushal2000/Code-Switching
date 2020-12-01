from data import *
from models import *
import copy
# from utils import *
from tqdm import tqdm
import pickle
import sklearn
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score 
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams.update({'font.size': 16})
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
filename = 'Datasets/HATE.pkl'
mode = 'dev'
num_labels = 2
hidden_size = 50
feature_dim = 22
k=5
epochs = 6
learning_rate = 1e-4
batch_size = 32

def metric_loss(preds,labels):
    m = nn.Softmax(dim=1)
    soft = m(preds)
    batch_size = int(preds.shape[0]/2)
    preds1, preds2 = preds[:batch_size], preds[batch_size:batch_size*2]
    labels1, labels2 = labels[:batch_size], labels[batch_size:batch_size*2]
    soft1, soft2 = soft[:batch_size], soft[batch_size:batch_size*2]
    p1 = 1-soft1[:,labels1]
    p2 = 1-soft2[:,labels2]
    p = 1+(p1+p2)/2
    loss1 = (labels1==labels2)*(p*torch.norm(preds1-preds2, dim=1))
    loss2 = (labels1!=labels2)*(torch.log(p*torch.exp(1-torch.norm(preds1-preds2, dim=1))))
    return torch.mean(torch.square(torch.max(loss1,loss2)))

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
        
class BERT_HAN_feature1(torch.nn.Module):
    def __init__(self, hidden_size, D_in, num_labels, feature_dim):
        super(BERT_HAN_feature1, self).__init__()
        self.embeddings = BertModel.from_pretrained('bert-base-multilingual-cased',  output_hidden_states = True)
        self.bert_encoder = nn.LSTM(input_size=D_in,
                                    hidden_size=hidden_size,
                                    num_layers=1, 
                                    bidirectional=True)
        self.bert_encoder_2 = nn.LSTM(input_size=hidden_size*2,
                                    hidden_size=feature_dim//2,
                                    num_layers=1, 
                                    bidirectional=True)
        self.attent = Attention(2*(feature_dim//2), 128+1)
        self.lin_feat = nn.Linear(feature_dim, 2*(feature_dim//2), bias = True)
        self.linear = nn.Linear(2*(feature_dim//2), num_labels, bias = True)
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
            ypred, attn = nmodel(b_input_ids, b_features, b_input_mask, b_tokens)
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

def train(training_dataloader, validation_dataloader, nmodel, epochs = 6, alpha = 0.25, beta = 1.0):
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
    optimizer2 = AdamW(params, lr=2e-4, eps = 1e-8)
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # TensorDataset(training_inputs, training_masks, training_tokens,training_features, training_labels, training_ids)
    criterion = nn.CrossEntropyLoss()
    best_model = copy.deepcopy(nmodel)
    best_acc = 0
    best_micro = 0
    best_macro = 0
    for epoch_i in tqdm(range(0, epochs)):
        total_train_loss = 0
        nmodel.train()
        for step, batch in enumerate(training_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_tokens = batch[2].to(device)
            b_features = batch[3].to(device)
            b_labels = batch[4].to(device).long()
            ypred, attention_weight = nmodel(b_input_ids, b_features, b_input_mask, b_tokens)
            # t = metric_loss(ypred, b_labels)
            loss = beta*(alpha*metric_loss(ypred, b_labels) + (1-alpha)*criterion(ypred, b_labels))

            total_train_loss += loss
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert_params.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer1.step()
            optimizer2.step()
            scheduler1.step()
            scheduler2.step()

        # print()

        # print(f'Total Train Loss = {total_train_loss}')
        # print('#############    Validation Set Stats')
        avg_val_accuracy, micro_f1, macro_f1, val_loss = evaluate(validation_dataloader, nmodel)
        # print(f'Total Validation Loss = {val_loss}')
        # print("  Accuracy: {0:.4f}".format(avg_val_accuracy))
        # print("  Micro F1: {0:.4f}".format(micro_f1))
        # print("  Macro F1: {0:.4f}".format(macro_f1))
        if macro_f1 > best_macro:
            best_model = copy.deepcopy(nmodel)
            best_macro = macro_f1
            best_acc = avg_val_accuracy
            best_micro = micro_f1

        # print()

    return best_model, best_acc, best_micro, best_macro

hidden_sizes = [25, 50, 100, 200]
alphas = [0.1, 0.25, 0.50]
betas = [1, 5]
feature_dims = [9, 22]

def get_dataloaders(inputs, masks, tokens, labels, features, ids):
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

    return training_dataloader, test_dataloader

results = {}
try:
    with open('hate_results.pkl', 'rb') as f:
        results = pickle.load(f)
except:
    with open('hate_results.pkl', 'wb') as f:
        pickle.dump(results, f)

results

model_name = 'Fusion_HAN_Bert'
dataset_name = 'Hate'

for hidden_size in hidden_sizes:
    for alpha in alphas:
        for beta in betas:
            for feature_dim in feature_dims:
                if model_name not in results: results[model_name] = {}
                if dataset_name not in results[model_name]: results[model_name][dataset_name] = {}
                if feature_dim not in results[model_name][dataset_name]: results[model_name][dataset_name][feature_dim] = {}
                if (hidden_size, alpha, beta) in results[model_name][dataset_name][feature_dim].keys(): 
                    print('Skipping')
                    print((hidden_size, alpha, beta))
                    continue
                model = BERT_HAN_feature1(hidden_size, 768, num_labels, feature_dim).to(device)
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                ids, inputs, masks, tokens, labels, features = process_data(data, feature_dim, mode)
                # print(features.shape)
                inputs, masks, tokens, labels, features, ids = sklearn.utils.shuffle(inputs, masks, tokens, labels, features, ids, random_state=42)
                kf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
                kf.get_n_splits(inputs, labels)
                total_acc, total_micro, total_macro   = 0,0,0
                scores = []
                for train_index, test_index in kf.split(inputs, labels):
                    training_dataloader, test_dataloader = get_dataloaders(inputs, masks, tokens, labels, features, ids)
                    best_model, acc, micro, macro = train(training_dataloader, test_dataloader, copy.deepcopy(model), epochs = epochs, alpha = alpha, beta = beta)
                    total_acc += acc
                    total_micro += micro
                    total_macro += macro
                    print(macro, acc, micro)
                score = (total_macro/k, total_acc/k, total_micro/k)
                print(hidden_size, alpha, beta)
                print(score)
                results[model_name][dataset_name][feature_dim][(hidden_size, alpha, beta)] = score
                with open('hate_results.pkl', 'wb') as f:
                    pickle.dump(results, f)

