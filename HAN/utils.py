import pickle
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import random
import numpy as np
from sklearn.preprocessing import normalize

def load_data(data_file, embedding_path):
    with open(embedding_path,'rb') as F:
        embedding=pickle.load(F, encoding="bytes")
        embedding['oov']=[random.random() for i in range(50)]

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    signals = data['signal'].values
    labels = data['sentiment'].values
    blank = np.zeros((len(signals),len(signals[0])))
    for i in range(len(signals)):
        blank[i] = np.asarray(signals[i])
    signals = normalize(blank,axis=0)
    sentences = data['tweet'].values
    ids = np.arange(len(labels))    
    return embedding, signals, sentences, labels, ids

def build_sentence_char_embeddings(sentences, embedding, MAX_LENGTH_SENTENCE, MAX_LENGTH_CHAR):
    t = Tokenizer()
    t.fit_on_texts(sentences)
    encoded_sents = t.texts_to_sequences(sentences)
    padded_sents = pad_sequences(encoded_sents, maxlen=MAX_LENGTH_SENTENCE, padding='post')

    vocab_size_word = len(embedding.keys()) + 1
    embedding_matrix = np.zeros((vocab_size_word, 50))
    sentences_char=[]
    for sentence in sentences:
        sentences_char.append(" ".join([a.split("_")[0] for a in sentence]))
    
    t_char=Tokenizer(char_level=True)
    t_char.fit_on_texts(sentences_char)
    vocab_size_char = len(t_char.word_index)+1

    encoded_sentence_char = []
    for sentence in sentences:
        temp = []
        for word in sentence[:MAX_LENGTH_SENTENCE]:
            temp.append(pad_sequences(
                    t_char.texts_to_sequences([word.split("_")[0]]), 
                    maxlen=MAX_LENGTH_CHAR, 
                    padding='post'
                )[0]
            )
        for i in range(30-len(temp)):
            temp.append([0 for _ in range(10)])
        encoded_sentence_char.append(np.array(temp))
    encoded_sentence_char = np.array(encoded_sentence_char)
    assert(encoded_sentence_char.shape==(len(sentences),MAX_LENGTH_SENTENCE,MAX_LENGTH_CHAR))        

    count=0
    for word, i in t.word_index.items():
        try:
            embedding_matrix[i] = embedding[word]
        except:
            embedding_matrix[i] = embedding['oov']
            count+=1
    print("OOV count:",count)

    return padded_sents, encoded_sentence_char, embedding_matrix, vocab_size_char, vocab_size_word

def run_k_fold(n_folds, signal_feature, padded_sents, labels, encoded_sentence_char):
    skf = StratifiedKFold(n_splits=n_folds ,shuffle=True,random_state=SEED)
    FEATURE_LENGTH = len(signal_feature[0])
    acc_sum,f1_sum=0,0
    for i, (train, test) in tqdm(enumerate(skf.split(padded_sents,labels)), total=n_folds):
        # Split into test_train
        X_train_LSTM, X_test_LSTM = padded_sents[train], padded_sents[test]
        X_train_char, X_test_char = encoded_sentence_char[train], encoded_sentence_char[test]
        y_train, y_test  = labels[train], labels[test]
        X_train_NN, X_test_NN = signal_feature[train], signal_feature[test]        

        filepath="weights.best.switching.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        earlystop = EarlyStopping(monitor = 'val_acc',
                                min_delta = 0,
                                patience = 15,
                                verbose = 0,
                                restore_best_weights = True)
        
        callbacks_list = [earlystop,checkpoint, TQDMNotebookCallback()]
        model = build_model(FEATURE_LENGTH, LATE_INCORP_MODE, COMBINED_LAYER_SIZE, HIDDEN_SIZE_ATT_LAYER,  MAX_LENGTH_SENTENCE, MAX_LENGTH_CHAR, embedding_matrix, vocab_size_char, vocab_size_word)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m])
        model.fit([X_train_LSTM,X_train_char, X_train_NN],y_train, validation_split=0.2, epochs=100,callbacks=callbacks_list, verbose=0, batch_size=32)
        loss, accuracy, f1_score = model.evaluate([X_test_LSTM,X_test_char, X_test_NN], y_test, verbose=0)
        print("Loss:%f accuracy:%f f1_score:%f", (loss,accuracy,f1_score))
        acc_sum+=accuracy
        f1_sum+=f1_score
        gc.collect()
        
    print("Mean Results:",acc_sum/(n_folds*1.00),"Macro_F1_mean:",f1_sum/(n_folds*1.00))

def run_80_20(signal_feature, padded_sents, labels, encoded_sentence_char):
    X_train_LSTM, X_test_LSTM, X_train_char, X_test_char, X_train_NN, X_test_NN, y_train, y_test = train_test_split(padded_sents, encoded_sentence_char, signal_feature, labels, stratify=labels, test_size=0.2, random_state=SEED)
    FEATURE_LENGTH = len(signal_feature[0])
    filepath="weights.best.switching.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    earlystop = EarlyStopping(monitor = 'val_acc',
                            min_delta = 0,
                            patience = 15,
                            verbose = 0,
                            restore_best_weights = True)
    
    callbacks_list = [earlystop, TQDMNotebookCallback()]
    model = build_model(FEATURE_LENGTH, LATE_INCORP_MODE, COMBINED_LAYER_SIZE, HIDDEN_SIZE_ATT_LAYER,  MAX_LENGTH_SENTENCE, MAX_LENGTH_CHAR, embedding_matrix, vocab_size_char, vocab_size_word)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m])
    model.fit([X_train_LSTM,X_train_char, X_train_NN],y_train, validation_split=0.2, epochs=100,callbacks=callbacks_list, verbose=0, batch_size=32)
    loss, accuracy, f1_score = model.evaluate([X_test_LSTM,X_test_char, X_test_NN], y_test, verbose=0)
    print("Loss:%f accuracy:%f f1_score:%f", (loss,accuracy,f1_score))
    return model
