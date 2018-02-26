import os
import random
from nltk import word_tokenize


def get_tokens(t):
    return [w.lower() for w in word_tokenize(t)]


def read_data(dirname, sent):
    data = []
    vocab = []
    for i, fn in enumerate(os.listdir(dirname + sent)):
        # For efficiency, let's just take the first 1000 pos,
        # and 1000 neg documents
        if i > 1000:
            break
        if fn.endswith(".txt"):
            print("Reading %s" % fn)
            with open(dirname + sent + "/" + fn) as f:
                tokens = get_tokens(f.read())
                data.append((tokens, sent))
                vocab = vocab + tokens

    return (data, vocab)

def prepare_data(dirname):
    print("Reading Positive examples...")
    posdata, posvocab = read_data(dirname, "pos")
    print("Reading Negative examples...")
    negdata, negvocab = read_data(dirname, "neg")
    data = posdata + negdata
    vocab = posvocab + negvocab

    random.shuffle(data)

    return (data, list(set(vocab)))
