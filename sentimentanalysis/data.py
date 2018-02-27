import os
import random
from nltk import word_tokenize


def get_tokens(t):
    return [w.lower() for w in word_tokenize(t)]


def read_data(dirname, sent, sample_size=None):
    data = []
    vocab = []
    for i, fn in enumerate(os.listdir(dirname + sent)):
        # For efficiency, let's just take the first 1000 pos,
        # and 1000 neg documents
        if sample_size is not None and i > sample_size:
            break
        if fn.endswith(".txt"):
            print("Reading %s" % fn)
            with open(dirname + sent + "/" + fn) as f:
                tokens = get_tokens(f.read())
                data.append((tokens, sent))
                vocab = vocab + tokens

    return (data, vocab)

def prepare_data(dirname, sample_size=None):
    if sample_size is not None:
        """
        Split sample size in half so we get ~half pos half neg
        """
        sample_size = int(sample_size/2)
    print("Reading Positive examples...")
    posdata, posvocab = read_data(dirname, "pos", sample_size=sample_size)
    print("Reading Negative examples...")
    negdata, negvocab = read_data(dirname, "neg", sample_size=sample_size)
    data = posdata + negdata
    vocab = posvocab + negvocab

    random.shuffle(data)

    return (data, list(set(vocab)))
