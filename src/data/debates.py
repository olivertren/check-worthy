from enum import Enum
from os.path import join
import numpy as np
from nltk.tokenize import word_tokenize
# from src.data.models import Sentence
# from src.utils.config import get_config

# CONFIG = get_config()
FILE_EXT = "_ann.tsv"
# CB_FILE_EXT = '_cb.tsv'
SEP = "\t"
DATA_PATH = "/usr/users/oliverren/meng/check-worthy/data/claim-rank/transcripts_all_sources"
# DATA_PATH = "/usr/users/oliverren/meng/check-worthy/data/test"


class Sentence(object):
    def __init__(self, id, text, label, debate, speaker):
        self.id = id
        self.text = text
        self.label = label
        self.label_test = label[1] # label = [0,1] for check worthy claims and [1,0] for non check worthy claims
        self.tokens = word_tokenize(text)
        self.debate = debate
        self.features = {}
        self.speaker = speaker


        

Config = {'FIRST':'26_09_2016',
            'VP':'04_10_2016',
            'SECOND':'09_10_2016',
            'THIRD':'19_10_2016',
            'NinthDem':'9th_democratic',
            'TRUMP_S':'trump_acceptance_speech',
            'TRUMP_I':'trump_inauguration',
            'CLINTON_S':'clinton_acceptance_speech'}

class Debate(Enum):
    NinthDem = 1
    FIRST = 2
    VP = 3
    SECOND = 4
    THIRD = 5
    TRUMP_S = 6
    CLINTON_S = 7
    TRUMP_I = 8



DEBATES = [Debate.FIRST, Debate.VP, Debate.SECOND, Debate.THIRD]
SPEECHES = [Debate.NinthDem, Debate.TRUMP_S, Debate.TRUMP_I, Debate.CLINTON_S]

def read_all_debates(use_label = lambda x: [0,1] if int(x[2])>=1 else [1,0], sep_by_deb = True):
    """
    :return: a list of all sentences said in the debates
    """

    sentences = []
    if sep_by_deb:
        sentences.append(read_debates(Debate.FIRST, use_label))
        sentences.append(read_debates(Debate.VP, use_label))
        sentences.append(read_debates(Debate.SECOND, use_label))
        sentences.append(read_debates(Debate.THIRD, use_label))

        return sentences
    else:
        sentences += read_debates(Debate.FIRST, use_label)
        sentences += read_debates(Debate.VP, use_label)
        sentences += read_debates(Debate.SECOND, use_label)
        sentences += read_debates(Debate.THIRD, use_label)

        return sentences

def read_all_speeches(use_label = lambda x: [0,1] if int(x[2])>=1 else [1,0], sep_by_deb = True):
    """
    :return: a list of all sentences said in the debates
    """
    sentences = []
    if sep_by_deb:
        sentences.append(read_debates(Debate.NinthDem, use_label))
        sentences.append(read_debates(Debate.TRUMP_S, use_label))
        sentences.append(read_debates(Debate.TRUMP_I, use_label))
        sentences.append(read_debates(Debate.CLINTON_S, use_label))

        return sentences
    else:
        sentences += read_debates(Debate.NinthDem, use_label)
        sentences += read_debates(Debate.TRUMP_S, use_label)
        sentences += read_debates(Debate.TRUMP_I, use_label)
        sentences += read_debates(Debate.CLINTON_S, use_label)

        return sentences

def read_debates(debate, use_label='sum_all'):
    """
    Reads the debate transcripts data.
    :param debate: debates (Debate enum) to return the sentences for.
    :param use_label: how to form the gold label for the sentences
    - sum_all : label is the number of annotators that have agreed
    - lambda function : a custom function for a label, with input - the columns from file
    :return:

    Examples:
    1. Take for claims only those that more than one annotator agrees on it
    #>>> read_debates(Debate.FIRST, lambda x: 1 if int(x[2])>1 else 0)
    2. Take the number of annotators that have agreed on it
    #>>> read_debates(Debate.FIRST)
    """
    sentences = []
    name = debate.name
    debate_file_name = join(DATA_PATH, Config[name] + FILE_EXT)
    debate_file = open(debate_file_name)
    debate_file.readline()
    for line in debate_file:
        line = line.strip()
        columns = line.split(SEP)

        if use_label == 'sum_all':
            label = int(columns[2].strip())
        else:
            label = use_label(columns)
        labels = columns[3:-1]
        s = Sentence(columns[0], columns[-1], label, debate, columns[1])

        sentences.append(s)

    return sentences

def get_for_crossvalidation(use_label = lambda x: [0,1] if int(x[2])>=1 else [1,0]):
    """
    Splits the debates into four cross-validation sets.
    One of the debates is a test set at each cross validation.
    :return: test , val, and train sets
    """
    data_sets = []
    for i, debate in enumerate(DEBATES):
        train = read_all_debates()
        test = train.pop(i)
        if i < 3:
            val = train.pop(i)
        else:
            val = train.pop(0)
        train = [sentence for debate in train for sentence in debate]
        train += read_all_speeches(sep_by_deb=False)
        data_sets.append((debate, test, val, train))
        # print(len(test), len(val), len(train))
    return data_sets