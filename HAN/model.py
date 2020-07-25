from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.layers import GlobalMaxPool1D,  concatenate

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from keras.models import Model,Sequential
from keras.preprocessing import text as keras_text, sequence as keras_seq
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
import gc
from keras import optimizers

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights.append([self.W, self.b, self.u])
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def HAN_Linear_Feature(FEATURE_LENGTH, LATE_INCORP_MODE, COMBINED_LAYER_SIZE, HIDDEN_SIZE_ATT_LAYER,  MAX_LENGTH_SENTENCE, MAX_LENGTH_CHAR, embedding_matrix, vocab_size_char, vocab_size_word, num_outputs):
    input_feature_vec = Input(shape=(FEATURE_LENGTH,), name='input_feature_vector')

    word_in = Input(shape=(MAX_LENGTH_SENTENCE,), name='Input_for_sentence_level_HAN')
    emb_word = Embedding(vocab_size_word, 50, weights=[embedding_matrix], input_length=MAX_LENGTH_SENTENCE, trainable=False, name='word_embeddings')(word_in)

    char_in = Input(shape=(MAX_LENGTH_SENTENCE, MAX_LENGTH_CHAR,), name='Input_for_charecter_level_HAN')
    emb_char = TimeDistributed(Embedding(input_dim=vocab_size_char, output_dim=10,input_length=MAX_LENGTH_CHAR),name='Char_to_token_embedding')(char_in)

    char_enc = TimeDistributed(GRU(units=10, return_sequences=False,recurrent_dropout=0), name='Char_encoder')(emb_char)
    concat_word_char = concatenate([emb_word, char_enc], name='Concate_word_char_embeddings')
    main_lstm = Bidirectional(GRU(units=50, return_sequences=True,recurrent_dropout=0), name='BiLSTM_for_word_char_concat_embed')(concat_word_char)

    l_att_sent = AttLayer(HIDDEN_SIZE_ATT_LAYER)(main_lstm)
    HAN_output = Dense(COMBINED_LAYER_SIZE, activation='relu', name='Dense_layer_after_attention')(l_att_sent)
    model_HAN = Model([word_in,char_in], HAN_output)

    if LATE_INCORP_MODE=='ReduceOne':
        y = Dense(FEATURE_LENGTH, activation='relu', name='Dense_layer_1_for_feature_input')(input_feature_vec)
        y = Dense(1, activation="sigmoid", name='Dense_layer_2_for_feature_input')(y)    
    elif LATE_INCORP_MODE=='MakeEqual':
        y = Dense(FEATURE_LENGTH, activation="relu", name='Dense_layer_for_feature_input')(input_feature_vec)
    
    model_feature_vec = Model(inputs=input_feature_vec, outputs=y)

    combined = concatenate([model_HAN.output, model_feature_vec.output], name='Concat_HAN_and_feature_vector')
    z = Dense(num_outputs, activation="sigmoid", name='Output_layer')(combined)

    model = Model(inputs=[model_HAN.input[0],model_HAN.input[1], model_feature_vec.input], outputs=z)
    return model

def HAN(COMBINED_LAYER_SIZE, HIDDEN_SIZE_ATT_LAYER,  MAX_LENGTH_SENTENCE, MAX_LENGTH_CHAR, embedding_matrix, vocab_size_char, vocab_size_word, num_outputs):
    input_feature_vec = Input(shape=(FEATURE_LENGTH,), name='input_feature_vector')

    word_in = Input(shape=(MAX_LENGTH_SENTENCE,), name='Input_for_sentence_level_HAN')
    emb_word = Embedding(vocab_size_word, 50, weights=[embedding_matrix], input_length=MAX_LENGTH_SENTENCE, trainable=False, name='word_embeddings')(word_in)

    char_in = Input(shape=(MAX_LENGTH_SENTENCE, MAX_LENGTH_CHAR,), name='Input_for_charecter_level_HAN')
    emb_char = TimeDistributed(Embedding(input_dim=vocab_size_char, output_dim=10,input_length=MAX_LENGTH_CHAR),name='Char_to_token_embedding')(char_in)

    char_enc = TimeDistributed(GRU(units=10, return_sequences=False,recurrent_dropout=0), name='Char_encoder')(emb_char)
    concat_word_char = concatenate([emb_word, char_enc], name='Concate_word_char_embeddings')
    main_lstm = Bidirectional(GRU(units=50, return_sequences=True,recurrent_dropout=0), name='BiLSTM_for_word_char_concat_embed')(concat_word_char)

    l_att_sent = AttLayer(HIDDEN_SIZE_ATT_LAYER)(main_lstm)
    HAN_output = Dense(COMBINED_LAYER_SIZE, activation='relu', name='Dense_layer_after_attention')(l_att_sent)
    z = Dense(num_outputs, activation="sigmoid", name='Output_layer')(HAN_output)
    
    model = Model([word_in,char_in], z)
    return model