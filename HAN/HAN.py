import numpy as np
import pickle
import argparse
from utils import load_data, build_sentence_char_embeddings
from model import HAN_Linear_Feature, HAN
from sklearn.model_selection import train_test_split
import sklearn
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random 
from sklearn.metrics import f1_score 

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

#### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-df", "--dataset_filename", type=str,
                    help="Expects a .pkl file (dataset file)")
parser.add_argument("-ef", "--embedding_filename", type=str,
                    help="Expects a .pkl file (embedding file)")
parser.add_argument("-hs", "--hidden_layer_size", type=int, default=30,
                    help="Size of hidden layer")
parser.add_argument("-cs", "--combined_layer_size", type=int, default=30,
                    help="Size of combined layer size")
parser.add_argument("-li", "--late_incoporation_type", type=str, default='MakeEqual',
                    help="Size of hidden layer")
parser.add_argument("-us", "--use_switching", type=bool, default=True,
                    help="use_switching")
parser.add_argument("-no", "--num_outputs", type=int, default=2,
                    help="use_switching")
args = parser.parse_args()

if __name__=="__main__":

	MAX_LENGTH_SENTENCE = 30
	MAX_LENGTH_CHAR = 10
	n_folds=10

	embedding, signals, sentences, labels, ids = load_data(args.dataset_filename, args.embedding_filename)
	padded_sents, encoded_sentence_char, embedding_matrix, vocab_size_char, vocab_size_word = build_sentence_char_embeddings(sentences, embedding, MAX_LENGTH_SENTENCE, MAX_LENGTH_CHAR)
	FEATURE_LENGTH = len(signals[0])
	labels = keras.utils.to_categorical(labels)
	padded_sents, encoded_sentence_char, signals, labels, ids = sklearn.utils.shuffle(padded_sents, encoded_sentence_char, signals, labels, ids,  random_state=42)
	X_train_LSTM, X_test_LSTM, X_train_char, X_test_char, X_train_NN, X_test_NN, y_train, y_test = train_test_split(padded_sents, encoded_sentence_char, signals, labels, stratify=labels, test_size=0.2, random_state=RANDOM_SEED)

	filepath="Best_Models/weights.best."
	if args.use_switching:
		filepath+="switching."
	filepath+= args.dataset_filename.split('/')[-1].replace('.pkl','.hdf5')

	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
	earlystop = EarlyStopping(monitor = 'val_acc',
	                        min_delta = 0,
	                        patience = 15,
	                        verbose = 0,
	                        restore_best_weights = True)

	callbacks_list = [earlystop]
	if args.use_switching:
		model = HAN_Linear_Feature(FEATURE_LENGTH, args.late_incoporation_type, args.combined_layer_size, args.hidden_layer_size,  MAX_LENGTH_SENTENCE, MAX_LENGTH_CHAR, embedding_matrix, vocab_size_char, vocab_size_word, args.num_outputs)
	else:
		model = HAN(args.combined_layer_size, args.hidden_layer_size,  MAX_LENGTH_SENTENCE, MAX_LENGTH_CHAR, embedding_matrix, vocab_size_char, vocab_size_word, args.num_outputs)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
	model.fit([X_train_LSTM,X_train_char, X_train_NN],y_train, validation_split=0.2, epochs=2,callbacks=callbacks_list, verbose=1, batch_size=32)
	loss, accuracy = model.evaluate([X_test_LSTM,X_test_char, X_test_NN], y_test, verbose=0)
	y_preds = model.predict([X_test_LSTM,X_test_char, X_test_NN], verbose=0)
	y_preds = np.argmax(y_preds, axis=1)
	y_test = np.argmax(y_test, axis=1)
	macro_f1, micro_f1 = f1_score(y_test, y_preds, average='macro'), f1_score(y_test, y_preds, average='micro')
	print("Loss:%f accuracy:%f macro_f1:%f micro_f1:%f", (loss,accuracy,macro_f1, micro_f1))
