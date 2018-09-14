from nltk.tokenize import word_tokenize

class Sentence(object):
    def __init__(self, id, text, label, speaker, debate, labels):
        self.id = id
        self.text = text
        self.label = label
        self.speaker = speaker
        self.debate = debate
        self.features = {}
        self.tokens = word_tokenize(text)
        self.labels = labels

def import_dataset(path):
    return "TODO"
