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

    posdata, posvocab = read_data(dirname, "pos", sample_size=sample_size)
    negdata, negvocab = read_data(dirname, "neg", sample_size=sample_size)
    data = posdata + negdata
    vocab = posvocab + negvocab + ["UNK"]

    random.shuffle(data)

    return (data, list(set(vocab)))

def prepare_sequence(seq, to_id):
    """
    Get a tensor of words
    """
    ids = []
    for w in seq:
        if w not in to_id.keys():
            w = "UNK"

        ids.append(to_id[w])

    tensor = torch.LongTensor(ids)
    return autograd.Variable(tensor)

def prepare_label(label, to_id):
    return autograd.Variable(torch.LongTensor([to_id[label]]))

def eval_preds(preds, golds):
    acc = 0
    for pred, gold in zip(preds, golds):
        print(pred, gold)
        if pred == gold:
            acc += 1

    return acc / len(golds)
