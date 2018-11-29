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
from src.models import models
import numpy as np

data_sets = debates.get_for_crossvalidation()
texts = [sentence.text for sentence in data_sets[0][1]]
tokenizer, word_index = models.create_tokenizer(texts)
MAX_SEQUENCE_LENGTH = max([len(sentence.split()) for sentence in texts])

inputs = Input(shape=(MAX_SEQUENCE_LENGTH, ))
encoder1 = models.create_embedding('/usr/users/oliverren/meng/check-worthy/data/glove/glove.6B.50d.txt', word_index, trainable=False, INPUT_LENGTH = MAX_SEQUENCE_LENGTH)(inputs)
encoder2 = LSTM(128)(encoder1)