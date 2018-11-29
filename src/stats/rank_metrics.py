import copy
import json
import os
from copy import deepcopy
from math import log2
from operator import attrgetter, itemgetter
from os.path import join


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

def precision(y_true, y_pred):
    tp = sum([1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1])
    fp = sum([1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1])
    return tp/tp+fp


def f1(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


def accuracy(y_true, y_pred):
    num_correct = len([1 for true, pred in zip(y_true, y_pred) if true == pred])
    return num_correct/len(y_true)


def r_precision(y_true, pred_probas, agreement=1):
    r = sum([1 if label >= agreement else 0 for label in y_true])
    return precision_at_n(y_true, pred_probas, n=r, agreement=agreement)


def average_precision(y_true, pred_probas, agreement=1):
    sorted_indexes = np.argsort(pred_probas)[::-1]
    relevant = sum([1 if _label >= agreement else 0 for _label in y_true])
    avg_p = 0
    for i, ind in enumerate(sorted_indexes):
        if y_true[ind] >= agreement:
            avg_p += precision_at_n(y_true, pred_probas, n=i + 1, agreement=agreement)
    return avg_p/relevant


def precision_at_n(y_true, pred_probas, n=10, agreement=1):
    sorted_indexes = np.argsort(pred_probas)[::-1]
    relevant = sum([1 if y_true[ind] >= agreement else 0 for ind in sorted_indexes[:n]])
    return relevant / n


def recall_at_n(y_true, pred_probas, n=10, agreement=1):
    sorted_indexes = np.argsort(pred_probas)[::-1]
    relevant = sum([1 if y_true[ind] >= agreement else 0 for ind in sorted_indexes[:n]])
    all_relevant = sum([1 if y_true[ind] >= agreement else 0 for ind in sorted_indexes])
    return relevant / all_relevant


def dcg(y_true, pred_probas, agreement=1):
    sorted_indexes = np.argsort(pred_probas)[::-1]
    result = 0
    for i, ind in enumerate(sorted_indexes):
        reli = 2 ** (1 if y_true[ind] >= agreement else 0) - 1
        denom = log2(i+2)
        result += reli / denom
    return result


def ndcg(y_true, pred_probas, agreement=1):
    result = dcg(y_true, pred_probas)
    idcg = dcg(y_true, y_true, agreement=agreement)
    return result/idcg


def get_mrr(y_true, pred_probas, agreement=1):
    mrr = 0

    sorted_indexes = np.argsort(pred_probas)[::-1]
    for i, ind in enumerate(sorted_indexes):
        if y_true[ind] >= agreement:
            mrr += 1/(i+1)
    return mrr


def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)


def get_all_metrics(sentences, agreement=1, test_deb = ''):
    """
    Prints list of results for a ranker.
    :param sentences:
    :return:
    """
    metrics = {'RR': [], 'AvgP': [], 'ROC': [], 'R_Prec':[], 'nDCG': [],
               'Precision': [], 'Recall': [], 'Accuracy':[], 'F1':[],
               'Recall@10': [], 'Recall@100': [],'Recall@150': [], 'Recall@200': [], 'Recall@50':[],
               'PR@1': [], 'PR@3': [], 'PR@5': [],'PR@20': [], 'PR@10': [], 'PR@50': [], 'PR@100': [], 'PR@200': []}

    for sentence_set in sentences:
        y_true = copy.deepcopy([1 if t.label_test >= agreement else 0 for t in sentence_set])
        y_pred = copy.deepcopy([s.pred[1] for s in sentence_set])
        y_pred_label = copy.deepcopy([s.pred_label for s in sentence_set])

        metrics['AvgP'].append(average_precision(y_true, y_pred, agreement=agreement))
        metrics['ROC'].append(roc_auc_score(y_true, y_pred))
        metrics['RR'].append(get_mrr(y_true, y_pred, agreement=agreement))
        metrics['nDCG'].append(ndcg(y_true, y_pred, agreement=agreement))
        metrics['R_Prec'].append(r_precision(y_true, y_pred, agreement=agreement))
        metrics['Precision'].append(precision_score(y_true, y_pred_label))
        metrics['Recall'].append(recall_score(y_true, y_pred_label))
        metrics['F1'].append(f1(y_true, y_pred_label))
        metrics['Accuracy'].append(accuracy(y_true, y_pred_label))

        for i in [1, 3, 5, 10, 20, 50, 100, 200]:
            metrics['PR@'+str(i)].append(precision_at_n(y_true, y_pred, i, agreement=agreement))

        for i in [10, 50, 100, 150, 200]:
            metrics['Recall@'+str(i)].append(recall_at_n(y_true, y_pred, i, agreement=agreement))

    for key, value in sorted(metrics.items(), key=itemgetter(0)):
        print("{0}\t\t {1:.4f}".format(key, mean(value)))

    print_for_table(metrics)

    with open(test_deb + '.tsv', 'w') as f:
        writer = csv.writer(f,delimiter='\t')
        for key, value in sorted(metrics.items(), key=itemgetter(0)):
            writer.writerow([key, value])


def print_for_table(metrics):
    order = ["Accuracy", "Precision", "Recall", "F1", "AvgP", "ROC",
             "RR", "PR@1", "PR@3", "PR@5", "PR@10", "PR@20", "PR@50",
             "PR@100", "PR@200", "Recall@10", "Recall@50", "Recall@100", "Recall@150", "Recall@200",
             "R_Prec", "nDCG"]
    labels = "\t".join([l for l in order])
    out = "\t".join(["{0:.4f}".format(mean(metrics[key])) for key in order])
    print(out)
    print(labels)


def get_metrics_for_plot(agreement, ranks):
    """
    Calculates the precision, recalls and avg. precisions at various threshold levels.
    :param agreement: level of agreement for a cliam to be counted as positive
    :param ranks: ranks given by a classifier/ranker
    :return: average precisions, precisions, recalls, thresholds
    """
    precision = []
    recall = []
    av_p = []

    step = 0.01
    thresholds = [i for i in np.arange(0, 1 + step, step)]

    for threshold in thresholds:
        av_p_th = []
        precision_th = []
        recall_th = []

        for debate_sents in ranks:
            # get for positives claims with agreement above 'agreement'
            y_score = [1 if sent.label >= agreement else 0 for sent in debate_sents]

            # get for positive predictions those with rank above threshold
            y_test = [1 if sent.pred > threshold else 0 for sent in debate_sents]
            precision_th.append(precision_score(y_true=y_score, y_pred=y_test))
            recall_th.append(recall_score(y_true=y_score, y_pred=y_test))
            av_p_th.append(average_precision_score(y_true=y_score, y_score=y_test))

        av_p.append(mean(av_p_th))
        precision.append(mean(precision_th))
        recall.append(mean(recall_th))

    return av_p, precision, recall, thresholds


if __name__ == '__main__':
    data_sets = debates.get_for_crossvalidation()

    texts = [sentence.text for sentence in data_sets[0][1]]
    texts.extend([sentence.text for sentence in data_sets[0][2]])
    texts.extend([sentence.text for sentence in data_sets[0][3]])

    embedding_filepath = '/usr/users/oliverren/meng/check-worthy/data/glove/glove.6B.50d.txt'
    MAX_SENTENCE_LENGTH = max([len(sentence.split()) for sentence in texts])


    # tests: one hidden layer, dropout 0.3, softmax activation, fixed vs dynamic word embedding (300d glove)
    # results = []
    # exp_num = 'bilstm_d'
    # embedding_filepath = '/usr/users/oliverren/meng/check-worthy/data/glove/glove.6B.300d.txt'
    # for test_deb, test, val, train in data_sets:
    #     print(test_deb)

    #     x_train = [sentence.text for sentence in train]
    #     y_train = np.array([sentence.label for sentence in train])
    #     x_val = [sentence.text for sentence in val]
    #     y_val = np.array([sentence.label for sentence in val])
    #     x_test = [sentence.text for sentence in test]
    #     y_test = np.array([sentence.label for sentence in test])

    #     model, tokenizer = models.bilstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH, EMBEDDING_DIM = 300, LSTM_OUTPUT_DIM=600, dropout=0.3)

    #     x_train = tokenizer.texts_to_sequences(x_train)
    #     x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
    #     x_val = tokenizer.texts_to_sequences(x_val)
    #     x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
    #     x_test = tokenizer.texts_to_sequences(x_test)
    #     x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)

    #     filepath= join('results', exp_num, 'weights.best.hdf5')
    #     checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #     filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
    #     checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #     callbacks_list = [checkpoint_best, checkpoint_update]

    #     epochs = 200
    #     model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
    #     model.save(join('results', exp_num, 'model.h5'))

    #     # predict        
    #     model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
    #     predictions = model.predict_classes(x_test)
    #     pred_probs = model.predict(x_test)

    #     for i, sent in enumerate(test):
    #         sent.pred_label = predictions[i]
    #         sent.pred = pred_probs[i]

    #     results.append(test)
    #     get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    # get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))

    # results = []
    # exp_num = 'lstm_d'
    # for test_deb, test, val, train in data_sets:
    #     print(test_deb)

    #     x_train = [sentence.text for sentence in train]
    #     y_train = np.array([sentence.label for sentence in train])
    #     x_val = [sentence.text for sentence in val]
    #     y_val = np.array([sentence.label for sentence in val])
    #     x_test = [sentence.text for sentence in test]
    #     y_test = np.array([sentence.label for sentence in test])

    #     model, tokenizer = models.lstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH, EMBEDDING_DIM = 300, LSTM_OUTPUT_DIM=600, dropout=0.3)

    #     x_train = tokenizer.texts_to_sequences(x_train)
    #     x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
    #     x_val = tokenizer.texts_to_sequences(x_val)
    #     x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
    #     x_test = tokenizer.texts_to_sequences(x_test)
    #     x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)

    #     filepath= join('results', exp_num, 'weights.best.hdf5')
    #     checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #     filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
    #     checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #     callbacks_list = [checkpoint_best, checkpoint_update]
        
    #     epochs = 200
    #     model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
    #     model.save(join('results', exp_num, 'model.h5'))

    #     # predict        
    #     model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
    #     predictions = model.predict_classes(x_test)
    #     pred_probs = model.predict(x_test)

    #     for i, sent in enumerate(test):
    #         sent.pred_label = predictions[i]
    #         sent.pred = pred_probs[i]

    #     results.append(test)
    #     get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    # get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))


    # results = []
    # exp_num = 'bilstm_f'
    # embedding_filepath = '/usr/users/oliverren/meng/check-worthy/data/glove/glove.6B.300d.txt'
    # for test_deb, test, val, train in data_sets:
    #     print(test_deb)

    #     x_train = [sentence.text for sentence in train]
    #     y_train = np.array([sentence.label for sentence in train])
    #     x_val = [sentence.text for sentence in val]
    #     y_val = np.array([sentence.label for sentence in val])
    #     x_test = [sentence.text for sentence in test]
    #     y_test = np.array([sentence.label for sentence in test])

    #     model, tokenizer = models.bilstm(texts, embedding_filepath, embedding_trainable = False, INPUT_LENGTH = MAX_SENTENCE_LENGTH, EMBEDDING_DIM = 300, LSTM_OUTPUT_DIM=600, dropout=0.3)

    #     x_train = tokenizer.texts_to_sequences(x_train)
    #     x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
    #     x_val = tokenizer.texts_to_sequences(x_val)
    #     x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
    #     x_test = tokenizer.texts_to_sequences(x_test)
    #     x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)

    #     filepath= join('results', exp_num, 'weights.best.hdf5')
    #     checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #     filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
    #     checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #     callbacks_list = [checkpoint_best, checkpoint_update]

    #     epochs = 200
    #     model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
    #     model.save(join('results', exp_num, 'model.h5'))

    #     # predict        
    #     model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
    #     predictions = model.predict_classes(x_test)
    #     pred_probs = model.predict(x_test)

    #     for i, sent in enumerate(test):
    #         sent.pred_label = predictions[i]
    #         sent.pred = pred_probs[i]

    #     results.append(test)
    #     get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    # get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))

    # results = []
    # exp_num = 'lstm_f'
    # for test_deb, test, val, train in data_sets:
    #     print(test_deb)

    #     x_train = [sentence.text for sentence in train]
    #     y_train = np.array([sentence.label for sentence in train])
    #     x_val = [sentence.text for sentence in val]
    #     y_val = np.array([sentence.label for sentence in val])
    #     x_test = [sentence.text for sentence in test]
    #     y_test = np.array([sentence.label for sentence in test])

    #     model, tokenizer = models.lstm(texts, embedding_filepath, embedding_trainable = False, INPUT_LENGTH = MAX_SENTENCE_LENGTH, EMBEDDING_DIM = 300, LSTM_OUTPUT_DIM=600, dropout=0.3)

    #     x_train = tokenizer.texts_to_sequences(x_train)
    #     x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
    #     x_val = tokenizer.texts_to_sequences(x_val)
    #     x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
    #     x_test = tokenizer.texts_to_sequences(x_test)
    #     x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)

    #     filepath= join('results', exp_num, 'weights.best.hdf5')
    #     checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #     filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
    #     checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #     callbacks_list = [checkpoint_best, checkpoint_update]
        
    #     epochs = 200
    #     model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
    #     model.save(join('results', exp_num, 'model.h5'))

    #     # predict        
    #     model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
    #     predictions = model.predict_classes(x_test)
    #     pred_probs = model.predict(x_test)

    #     for i, sent in enumerate(test):
    #         sent.pred_label = predictions[i]
    #         sent.pred = pred_probs[i]

    #     results.append(test)
    #     get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    # get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))


    # parameters
    # texts, embedding_filepath, MAX_NUM_WORDS = 1000, INPUT_LENGTH = 100, EMBEDDING_DIM = 50, 
    # embedding_trainable = True, features = False,
    # LSTM_OUTPUT_DIM = 100, hidden_layer = True, num_hidden_layers = 1, hidden_activation = 'relu', 
    # dropout = None, final_activation = 'softmax'):
    

    # 1: one hidden layer, softmax activation
    results = []
    exp_num = '1b'
    for test_deb, test, val, train in data_sets:
        print(test_deb)

        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.bilstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))

    results = []
    exp_num = '1l'
    for test_deb, test, val, train in data_sets:
        print(test_deb)
     
        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.lstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))


    # 2: no hidden layer, softmax activation
    results = []
    exp_num = '2b'
    for test_deb, test, val, train in data_sets:
        print(test_deb)

        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.bilstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH, hidden_layer=False)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))

    results = []
    exp_num = '2l'
    for test_deb, test, val, train in data_sets:
        print(test_deb)

        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.lstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH, hidden_layer=False)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))


    # 3: one hidden layer, dropout, softmax activation
    results = []
    exp_num = '3b'
    for test_deb, test, val, train in data_sets:
        print(test_deb)

        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.bilstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH, dropout=0.3)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))

    results = []
    exp_num = '3l'
    for test_deb, test, val, train in data_sets:
        print(test_deb)

        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.lstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH, dropout=0.3)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))


    #4: two hidden layer, softmax activation
    results = []
    exp_num = '4b'
    for test_deb, test, val, train in data_sets:
        print(test_deb)

        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.bilstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH, num_hidden_layers=2)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))



    results = []
    exp_num = '4l'
    for test_deb, test, val, train in data_sets:
        print(test_deb)

        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.lstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH, num_hidden_layers=2)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))


    # 5: fixed word embeddings, one hidden layer, softmax activation
    results = []
    exp_num = '5b'
    for test_deb, test, val, train in data_sets:
        print(test_deb)

        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.bilstm(texts, embedding_filepath, embedding_trainable = False ,INPUT_LENGTH = MAX_SENTENCE_LENGTH)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))


    results = []
    exp_num = '5l'
    for test_deb, test, val, train in data_sets:
        print(test_deb)

        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.lstm(texts, embedding_filepath, embedding_trainable = False ,INPUT_LENGTH = MAX_SENTENCE_LENGTH)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))


    # 6: one hidden layer, softmax activation, 100d word embedding
    results = []
    exp_num = '6b'
    embedding_filepath = '/usr/users/oliverren/meng/check-worthy/data/glove/glove.6B.100d.txt'
    for test_deb, test, val, train in data_sets:
        print(test_deb)

        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.bilstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH, EMBEDDING_DIM=100, LSTM_OUTPUT_DIM=200)
        
        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))

    results = []
    exp_num = '6l'
    embedding_filepath = '/usr/users/oliverren/meng/check-worthy/data/glove/glove.6B.100d.txt'
    for test_deb, test, val, train in data_sets:
        print(test_deb)

        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.lstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH, EMBEDDING_DIM=100, LSTM_OUTPUT_DIM=200)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))


    # 7: one hidden layer, softmax activation, 200d word embedding
    results = []
    exp_num = '7b'
    embedding_filepath = '/usr/users/oliverren/meng/check-worthy/data/glove/glove.6B.200d.txt'
    for test_deb, test, val, train in data_sets:
        print(test_deb)

        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.bilstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH, EMBEDDING_DIM=200, LSTM_OUTPUT_DIM=400)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))

    results = []
    exp_num = '7l'
    embedding_filepath = '/usr/users/oliverren/meng/check-worthy/data/glove/glove.6B.200d.txt'
    for test_deb, test, val, train in data_sets:
        print(test_deb)

        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.lstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH, EMBEDDING_DIM=200, LSTM_OUTPUT_DIM=400)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))



    # 8: one hidden layer, softmax activation, 300d word embedding
    results = []
    exp_num = '8b'
    embedding_filepath = '/usr/users/oliverren/meng/check-worthy/data/glove/glove.6B.300d.txt'
    for test_deb, test, val, train in data_sets:
        print(test_deb)

        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.bilstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH, EMBEDDING_DIM=300, LSTM_OUTPUT_DIM=600)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))

    results = []
    exp_num = '8l'
    embedding_filepath = '/usr/users/oliverren/meng/check-worthy/data/glove/glove.6B.300d.txt'
    for test_deb, test, val, train in data_sets:
        print(test_deb)
        x_train = [sentence.text for sentence in train]
        y_train = np.array([sentence.label for sentence in train])
        x_val = [sentence.text for sentence in val]
        y_val = np.array([sentence.label for sentence in val])
        x_test = [sentence.text for sentence in test]
        y_test = np.array([sentence.label for sentence in test])

        model, tokenizer = models.lstm(texts, embedding_filepath, INPUT_LENGTH = MAX_SENTENCE_LENGTH, EMBEDDING_DIM=300, LSTM_OUTPUT_DIM=600)

        x_train = tokenizer.texts_to_sequences(x_train)
        x_train = pad_sequences(x_train, maxlen = MAX_SENTENCE_LENGTH)
        x_val = tokenizer.texts_to_sequences(x_val)
        x_val = pad_sequences(x_val, maxlen = MAX_SENTENCE_LENGTH)
        x_test = tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen = MAX_SENTENCE_LENGTH)
        

        filepath= join('results', exp_num, 'weights.best.hdf5')
        checkpoint_best = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        filepath= join('results', exp_num, 'weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
        checkpoint_update = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint_best, checkpoint_update]
        
        epochs = 200
        model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks = callbacks_list, epochs=epochs, verbose=2)
        model.save(join('results', exp_num, 'model.h5'))

        # predict        
        model.load_weights(join('results', exp_num, 'weights.best.hdf5'))
        predictions = model.predict_classes(x_test)
        pred_probs = model.predict(x_test)

        for i, sent in enumerate(test):
            sent.pred_label = predictions[i]
            sent.pred = pred_probs[i]

        results.append(test)
        get_all_metrics(copy.deepcopy([test]), agreement=1, test_deb=join('results', str(exp_num),str(test_deb)))
    get_all_metrics(results, agreement=1, test_deb=join('results', str(exp_num), 'Debate.ALL'))

