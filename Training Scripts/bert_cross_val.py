from data import *
from models import *
from utils import *
import pickle
import sklearn
import argparse
from sklearn.model_selection import train_test_split, StratifiedKFold

#### Set Seed for reproducibility of results
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", type=str, default='Datasets/Humour/data_frame_22.pkl',
                    help="Expects a .pkl file dataset")
parser.add_argument("-d", "--feature_dim", type=int, default=0,
                    help="Dimension of features")
parser.add_argument("-hs", "--hidden_size", type=int, default=0,
                    help="Hidden Size of Linear Layer")
parser.add_argument("-lr", "--learning_rate", type=float, default=2e-5,
                    help="Learning Rate of non-embedding params")
parser.add_argument("-n", "--num_labels", type=int, default=2,
                    help="Number of Labels")
parser.add_argument("-bs", "--batch_size", type=int, default=32,
                    help="Batch Size for Training Model")
parser.add_argument("-e", "--epochs", type=int, default=6,
                    help="Epochs to run Model for")
parser.add_argument("-k", type=int, default=5,
                    help="Number of folds for k-fold Cross Validation")
parser.add_argument("-m", "--mode", type=str, default='dev',
                    help="Romanized or Devanagari Mode")
args = parser.parse_args()

#### Set Up Dataset - Tokenise the data
data = {}
with open(args.filename, 'rb') as f:
    data = pickle.load(f)
ids, inputs, masks, tokens, labels, features = process_data(data, args.feature_dim, args.mode)

#### Initialise The Model
model = BERT_Linear_Feature(args.hidden_size, 768, args.num_labels, args.feature_dim).to(device)

#### Create K fold splits
inputs, masks, tokens, labels, features, ids = sklearn.utils.shuffle(inputs, masks, tokens, labels, features, ids, random_state=42)
kf = StratifiedKFold(n_splits=args.k, random_state=42, shuffle=True)
kf.get_n_splits(inputs, labels)

total_acc = 0
total_micro = 0
total_macro = 0

scores = []

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
    training_dataloader = DataLoader(training_data, sampler=training_sampler, batch_size=args.batch_size)

    test_data = TensorDataset(test_inputs, test_masks,test_tokens, test_features, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)
    
    best_model, acc, micro, macro = train(training_dataloader, test_dataloader, copy.deepcopy(model), epochs = args.epochs, lr2 = args.learning_rate)
    total_acc += acc
    total_micro += micro
    total_macro += macro
    
    score = (macro, acc, micro)
    scores.append(score)
    
print("==================FINAL RESULTS====================")
print("  Accuracy: {0:.4f}".format(total_acc/args.k))
print("  Micro F1: {0:.4f}".format(total_micro/args.k))
print("  Macro F1: {0:.4f}".format(total_macro/args.k))
print('===================================================')
print("==================F1 SCORES====================")
for score in scores: print(score[0])
print('===================================================')
print("==================ACC SCORES====================")
for score in scores: print(score[1])
print('===================================================')



