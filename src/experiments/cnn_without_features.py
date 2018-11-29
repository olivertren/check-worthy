import copy
import json
import os
from copy import deepcopy
from math import log2
from operator import attrgetter, itemgetter
from os.path import join
import pickle
import csv

import sys
sys.path.append('/usr/users/oliverren/meng/check-worthy')

import numpy as np
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

from src.data import debates
from src.models import models
from src.stats import rank_metrics as rm
from src.features.feature_sets import get_serialized_pipeline

data_sets = debates.get_for_crossvalidation()

texts = [sentence.text for sentence in data_sets[0][1]]
texts.extend([sentence.text for sentence in data_sets[0][2]])
texts.extend([sentence.text for sentence in data_sets[0][3]])

tokenizer, word_index = models.create_tokenizer(texts)

MAX_SENTENCE_LENGTH = max([len(sentence.split()) for sentence in texts])


# tests: one hidden layer, dropout 0.3, softmax activation, fixed vs dynamic word embedding (300d glove)
embedding_filepath = '/usr/users/oliverren/meng/check-worthy/data/glove/glove.6B.300d.txt'
results = []
exp_num = 'cnn_f'
for test_deb, test, val, train in data_sets:
    print(test_deb)

    with open(join('results', exp_num, 'tokenizer.pickle'), 'wb') as file:
        pickle.dump(tokenizer, file, protocol = pickle.HIGHEST_PROTOCOL)

    x_train = [sentence.text for sentence in train]
    y_train = np.array([sentence.label for sentence in train])
    x_val = [sentence.text for sentence in val]
    y_val = np.array([sentence.label for sentence in val])
    x_test = [sentence.text for sentence in test]
    y_test = np.array([sentence.label for sentence in test])

    model = models.cnn(word_index, embedding_filepath, embedding_trainable = False, INPUT_LENGTH = MAX_SENTENCE_LENGTH, EMBEDDING_DIM = 300, dropout=0.3)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
    x_val = tokenizer.texts_to_sequences(x_val)
    x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)

    filepath= join('results', exp_num, str(test_deb) + 'weights.best.hdf5')
    checkpoint_best = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    filepath= join('results', exp_num, str(test_deb) + 'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoint_update = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint_best, checkpoint_update]
    
    epochs = 200
    model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2, batch_size=16)
    model.save(join('results', exp_num, str(test_deb) + 'model.h5'))

    # predict        
    model.load_weights(join('results', exp_num, str(test_deb) + 'weights.best.hdf5'))
    predictions = model.predict_classes(x_test)
    pred_probs = model.predict(x_test)

    for i, sent in enumerate(test):
        sent.pred_label = predictions[i]
        sent.pred = pred_probs[i]

    results.append(test)
    rm.get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
rm.get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))

results = []
exp_num = 'cnn_d'
for test_deb, test, val, train in data_sets:
    print(test_deb)

    with open(join('results', exp_num, 'tokenizer.pickle'), 'wb') as file:
        pickle.dump(tokenizer, file, protocol = pickle.HIGHEST_PROTOCOL)

    x_train = [sentence.text for sentence in train]
    y_train = np.array([sentence.label for sentence in train])
    x_val = [sentence.text for sentence in val]
    y_val = np.array([sentence.label for sentence in val])
    x_test = [sentence.text for sentence in test]
    y_test = np.array([sentence.label for sentence in test])

    model = models.cnn(word_index, embedding_filepath, embedding_trainable = True, INPUT_LENGTH = MAX_SENTENCE_LENGTH, EMBEDDING_DIM = 300, dropout=0.3)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
    x_val = tokenizer.texts_to_sequences(x_val)
    x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)

    filepath= join('results', exp_num, str(test_deb) + 'weights.best.hdf5')
    checkpoint_best = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    filepath= join('results', exp_num, str(test_deb) + 'weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5')
    checkpoint_update = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint_best, checkpoint_update]
    
    epochs = 200
    model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2, batch_size=16)
    model.save(join('results', exp_num, str(test_deb) + 'model.h5'))

    # predict        
    model.load_weights(join('results', exp_num, str(test_deb) + 'weights.best.hdf5'))
    predictions = model.predict_classes(x_test)
    pred_probs = model.predict(x_test)

    for i, sent in enumerate(test):
        sent.pred_label = predictions[i]
        sent.pred = pred_probs[i]

    results.append(test)
    rm.get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
rm.get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))