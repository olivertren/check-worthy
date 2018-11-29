import sys
sys.path.append('/usr/users/oliverren/meng/check-worthy')

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Bidirectional, Dropout, Dense
from keras import Sequential
from src.data import debates
import numpy as np

MAX_NUM_WORDS = 1000
# data_set[i] is the ith crossvalidation split, data_set[i][0] says which debate is the test debate
# data_set[i][1] are the sentences in the test set
# data_set[i][2] are the sentences in the training set
data_sets = debates.get_for_crossvalidation()

texts = [sentence.text for sentence in data_sets[0][1]]
texts.extend([sentence.text for sentence in data_sets[0][2]])
MAX_SEQUENCE_LENGTH = max([len(sentence.split()) for sentence in texts])
# print(MAX_SEQUENCE_LENGTH)


# the embedding is already pretrained, so whenever we go to a different dataset, we should reset the embedding layer
# so that the embedding layer uses the words in the vocab of the dataset being tested
tokenizer = Tokenizer(num_words= MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
# print(sequences)
# print(texts[0])
# print(tokenizer.word_index)
word_index = tokenizer.word_index
# print(word_index)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Create Embedding layer
embeddings_index = {}
f = open('/usr/users/oliverren/meng/check-worthy/data/glove/glove.6B.50d.txt')
count = 0
for line in f:
    values = line.split()
    if count == 0:
        # print(values)
        count += 1
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

EMBEDDING_DIM = 50
# + 1 because indexes are positive integers
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not in embedding index will be all-zeros
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights = [embedding_matrix],
                            input_length = MAX_SEQUENCE_LENGTH,
                            trainable = False)


# bi-directional
LSTM_OUTPOUT_DIM = 200
HIDDEN_LAYER_DIM = 200
BATCH_SIZE = 32

x_train = [sentence.text for sentence in data_sets[0][2]]
y_train = [sentence.label for sentence in data_sets[0][2]]
x_test = [sentence.text for sentence in data_sets[0][1]]
y_test = [sentence.label for sentence in data_sets[0][1]]


x_train = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
x_test = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)


model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(LSTM_OUTPOUT_DIM)))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=200,
validation_data=[x_test, y_test])

model_2 = Sequential()
model_2.add(embedding_layer)
model_2.add(Bidirectional(LSTM(LSTM_OUTPOUT_DIM)))
model_2.add(Dense(HIDDEN_LAYER_DIM*4,activation='relu'))
model_2.add(Dropout(0.5))
model_2.add(Dense(HIDDEN_LAYER_DIM,activation='relu'))
model_2.add(Dropout(0.5))
model_2.add(Dense(1,activation='sigmoid'))
model_2.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model_2.summary())

model_2.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=200,
validation_data=[x_test, y_test])


from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)

def f1(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))

def accuracy(y_true, y_pred):
    num_correct = len([1 for true, pred in zip(y_true, y_pred) if true == pred])
    return num_correct/len(y_true)

print('model 1')
print('f1')
print(f1(y_test, model.predict_classes(x_test).reshape(-1)))
print('accuracy')
print(accuracy(y_test, model.predict_classes(x_test).reshape(-1)))


print('model 2')
print('f1')
print(f1(y_test, model_2.predict_classes(x_test).reshape(-1)))
print('accuracy')
print(accuracy(y_test, model_2.predict_classes(x_test).reshape(-1)))
