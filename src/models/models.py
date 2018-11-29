import sys
sys.path.append('/usr/users/oliverren/meng/check-worthy')

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense, Input, Merge
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import Sequential
from src.data import debates
import numpy as np

# take a set of texts and returns a tokenizer and word_index trained on that text
# texts: a list of sentences (str) from which to create the word dictionary
#
# returns: tokenizer (fitted tokenizer), word_index (dictionary mapping word to index)
def create_tokenizer(texts, MAX_NUM_WORDS = 35000):
    tokenizer = Tokenizer(num_words= MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    word_index = tokenizer.word_index
    return tokenizer, word_index


# create an embedding layer
# embedding_filepath: path to a pretrained embedding from which to intialize the embedding layer
# word_index: dictionary mapping the words that should be included in the embedding to an index
# EMBEDDING_DIM: int, the number of dimensions to embed words into
# trainable: bool, whether or not the embedding weights should be updated during training
# INPUT_LENGTH: int, the number of words in the input of our samples
#
# returns: embedding_layer (a keras embedding layer to include in models)
def create_embedding(embedding_filepath, word_index, EMBEDDING_DIM = 50, trainable = True, INPUT_LENGTH = 100):
    embeddings_index = {}
    f = open(embedding_filepath)

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # indexes are positive integers
    embedding_matrix = np.random.standard_normal((len(word_index) + 1, EMBEDDING_DIM))
    embedding_matrix[0] = np.zeros(EMBEDDING_DIM)
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not in embedding index are randomly intialized
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights = [embedding_matrix],
                                input_length = INPUT_LENGTH,
                                trainable = trainable)

    return embedding_layer


# texts: a list of sentences (str) from which to create the word dictionary
# embedding_filepath: path to a pretrained embedding from which to intialize the embedding layer
def bilstm(word_index, embedding_filepath, INPUT_LENGTH = 100, EMBEDDING_DIM = 50, embedding_trainable = True, features = False, feat_size = 1,
        LSTM_OUTPUT_DIM = 100, hidden_layer = True, num_hidden_layers = 1, hidden_activation = 'relu', dropout = None, final_activation = 'softmax',
        learning_rate = 0.00001):
    
    embedding_layer = create_embedding(embedding_filepath, word_index, EMBEDDING_DIM, embedding_trainable, INPUT_LENGTH)

    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(LSTM_OUTPUT_DIM)))
    if features:
        inputs = Input(shape=(feat_size, ))
        features_model = Model(inputs = inputs, outputs = inputs)
        final_model = Sequential()
        final_model.add(Merge([model, features_model], mode = 'concat'))

        if hidden_layer:
            for i in range(num_hidden_layers):
                final_model.add(Dense(2*LSTM_OUTPUT_DIM + feat_size, activation = hidden_activation))
                if dropout != None:
                    final_model.add(Dropout(dropout))

            final_model.add(Dense(2,activation=final_activation))
            final_model.compile(loss = 'binary_crossentropy', optimizer=Adam(learning_rate), metrics = ['accuracy'])

            print(final_model.summary())
            return final_model


    if hidden_layer:
        for i in range(num_hidden_layers):
            model.add(Dense(2*LSTM_OUTPUT_DIM, activation = hidden_activation))
            if dropout != None:
                model.add(Dropout(dropout))

    model.add(Dense(2,activation=final_activation))
    model.compile(loss = 'binary_crossentropy', optimizer=Adam(learning_rate), metrics = ['accuracy'])

    print(model.summary())
    return model


def lstm(word_index, embedding_filepath, INPUT_LENGTH = 100, EMBEDDING_DIM = 50, embedding_trainable = True, features = False, feat_size = 1,
        LSTM_OUTPUT_DIM = 100, hidden_layer = True, num_hidden_layers = 1, hidden_activation = 'relu', dropout = None, final_activation = 'softmax',
        learning_rate = 0.00001):
    
    embedding_layer = create_embedding(embedding_filepath, word_index, EMBEDDING_DIM, embedding_trainable, INPUT_LENGTH)

    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(LSTM_OUTPUT_DIM))
    if features:
        inputs = Input(shape=(feat_size, ))
        features_model = Model(inputs = inputs, outputs = inputs)
        final_model = Sequential()
        final_model.add(Merge([model, features_model], mode = 'concat'))

        if hidden_layer:
            for i in range(num_hidden_layers):
                final_model.add(Dense(2*LSTM_OUTPUT_DIM + feat_size, activation = hidden_activation))
                if dropout != None:
                    final_model.add(Dropout(dropout))

            final_model.add(Dense(2,activation=final_activation))
            final_model.compile(loss = 'binary_crossentropy', optimizer=Adam(learning_rate), metrics = ['accuracy'])

            print(final_model.summary())
            return final_model

    if hidden_layer:
        for i in range(num_hidden_layers):
            model.add(Dense(LSTM_OUTPUT_DIM, activation = hidden_activation))
            if dropout != None:
                model.add(Dropout(dropout))
                
    model.add(Dense(2,activation=final_activation))
    model.compile(loss = 'binary_crossentropy', optimizer=Adam(learning_rate), metrics = ['accuracy'])

    print(model.summary())
    return model

def cnn(word_index, embedding_filepath, INPUT_LENGTH = 100, EMBEDDING_DIM = 50, embedding_trainable = True, features = False, feat_size = 1,
        CNN_OUTPUT_DIM = 128, filters = 5, pool_size = 2, hidden_layer = True, num_hidden_layers = 1, hidden_activation = 'relu', dropout = None, final_activation = 'softmax',
        learning_rate = 0.00001):
    
    embedding_layer = create_embedding(embedding_filepath, word_index, EMBEDDING_DIM, embedding_trainable, INPUT_LENGTH)

    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(CNN_OUTPUT_DIM, filters, activation = 'relu'))
    model.add(MaxPooling1D(pool_size))
    # model.add(Conv1D(CNN_OUTPUT_DIM, filters, activation = 'relu'))
    # model.add(MaxPooling1D(pool_size))
    # model.add(Conv1D(CNN_OUTPUT_DIM, filters, activation = 'relu'))
    # model.add(MaxPooling1D(pool_size))
    model.add(Flatten())
    if features:
        inputs = Input(shape=(feat_size, ))
        features_model = Model(inputs = inputs, outputs = inputs)
        final_model = Sequential()
        final_model.add(Merge([model, features_model], mode = 'concat'))

        if hidden_layer:
            for i in range(num_hidden_layers):
                final_model.add(Dense(2*CNN_OUTPUT_DIM + feat_size, activation = hidden_activation))
                if dropout != None:
                    final_model.add(Dropout(dropout))

            final_model.add(Dense(2,activation=final_activation))
            final_model.compile(loss = 'binary_crossentropy', optimizer=Adam(learning_rate), metrics = ['accuracy'])

            print(final_model.summary())
            return final_model

    if hidden_layer:
        for i in range(num_hidden_layers):
            model.add(Dense(CNN_OUTPUT_DIM, activation = hidden_activation))
            if dropout != None:
                model.add(Dropout(dropout))
                
    model.add(Dense(2,activation=final_activation))
    model.compile(loss = 'binary_crossentropy', optimizer=Adam(learning_rate), metrics = ['accuracy'])

    print(model.summary())
    return model



def main():
    data_sets = debates.get_for_crossvalidation()

    texts = [sentence.text for sentence in data_sets[0][1]]
    texts.extend([sentence.text for sentence in data_sets[0][2]])

    tokenizer, word_index = create_tokenizer(texts)
    create_embedding('/usr/users/oliverren/meng/check-worthy/data/glove/glove.6B.50d.txt', word_index)

if __name__ == '__main__':
    main()