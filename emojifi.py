import numpy as np 
import glove as glove
from utils import read_csv

from keras.models import Model 
from keras.layers import Dense, Input, Dropout, LSTM, Activation 
from keras.layers.embeddings import Embedding 
from keras.preprocessing import sequence 
from keras.initializers import glorot_uniform

VOCAB_SIZE = 1200001
EMBEDDING_DIMS = 50

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    emb_matrix = np.zeros([VOCAB_SIZE, EMBEDDING_DIMS])
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    embedding_layer = Embedding(VOCAB_SIZE, EMBEDDING_DIMS, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer

def load_model(input_shape, word_to_vec_map, word_to_index):
    input_indices = Input(shape=input_shape, dtype="int32")
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(input_indices)

    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(5, activation='softmax')(X)
    X = Activation('softmax')(X)

    model = Model(inputs=input_indices, outputs=X)
    return model

def load_dataset():
    X_train, Y_train = read_csv("./data/train_emoji.csv")
    X_test, Y_test = read_csv("./data/test.csv")
    max_word_count = len(max(X_train, key=len).split())

    return (max_word_count, X_train, Y_train, X_test, Y_test)

def words_to_indices(input, word_to_index, max_word_count):
    m = input.shape[0]
    input_indices = np.zeros([m, max_word_count])
    for i in range(m):
        statement_words = list(word.lower() for word in input[i].split())
        j = 0

        for w in statement_words:
            input_indices[i, j] = word_to_index[w]
            j = j + 1
    return input_indices
