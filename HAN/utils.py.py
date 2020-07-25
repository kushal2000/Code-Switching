def load_data(data_file, embedding_path):
    DATA_FILES = {
        'HUMOUR':'/content/ACL20-Code-switching-patterns/Humour/Dataset/dataset_humour_processed.pkl',
        'HATE':'/content/ACL20-Code-switching-patterns/Hate/Dataset/dataset_hate_processed_manual_annotated.pkl',
        'SARCASM':'/content/ACL20-Code-switching-patterns/Sarcasm/Datasets/dataset_sarcasm_processed_6.pkl'
    }
    EMBEDDING_PATH = 'embedding_50d.pkl'
    DATA_FILE_PATH = DATA_FILES[task]

    with open(EMBEDDING_PATH,'rb') as F:
        embedding=pickle.load(F, encoding="bytes")
        embedding['oov']=[random.random() for i in range(50)]

    with open(DATA_FILE_PATH,'rb') as F:
        u = pickle._Unpickler(F)
        u.encoding = 'latin1'
        data = u.load()
    
    return embedding, data

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def build_sentence_char_embeddings(data, MAX_LENGTH_SENTENCE, MAX_LENGTH_CHAR):
    
    sentences=[data[key]['tweet'] for key in data]

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
            #print(word)
            embedding_matrix[i] = embedding['oov']
            print("OOV here")
            count+=1
    print("OOV count:",count)

    return padded_sents, encoded_sentence_char, embedding_matrix, vocab_size_char, vocab_size_word

