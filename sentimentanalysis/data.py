import os
import random
from nltk import word_tokenize

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PADDING_SYMBOL = "#"

def get_tokens(t):
    return [w.lower() for w in word_tokenize(t)]

def read_data(dirname, sent, sample_size=None, randomize=False):
    data = []
    vocab = []
    dircontents = os.listdir(dirname + sent)
    if randomize:
        # Randomize the order in which we parse files
        # so that when sampling small sets of docs, it is not deterministic.
        random.shuffle(dircontents)
    for i, fn in enumerate(dircontents):
        if sample_size is not None and i > sample_size:
            break
        if fn.endswith(".txt"):
            with open(dirname + sent + "/" + fn) as f:
                tokens = get_tokens(f.read())
                data.append((tokens, sent))
                vocab = vocab + tokens

    return (data, vocab)

def prepare_data(dirname, sample_size=None, randomize=False):
    if sample_size is not None:
        """
        Split sample size in half so we get ~half pos half neg
        """
        sample_size = int(sample_size/2)

    posdata, posvocab = read_data(dirname, "pos", sample_size=sample_size, randomize=randomize)
    negdata, negvocab = read_data(dirname, "neg", sample_size=sample_size, randomize=randomize)
    data = posdata + negdata
    # get all words, plu sunkown symbol, and padding symbol,
    vocab = [PADDING_SYMBOL, "UNK"] + posvocab + negvocab

    random.shuffle(data)

    return (data, list(set(vocab)))

def prepare_test_sequence(seq, to_id):
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

def get_train_ids(seq, to_id):
    ids = []

    # Randomly select 15% of the sequence to replace with UNK
    unk_sample_size = int(len(seq) * .15)
    UNK_indices = random.sample(range(len(seq)), unk_sample_size)

    for i, w in enumerate(seq):
        if i in UNK_indices:
            ids.append(to_id["UNK"])
        else:
            ids.append(to_id[w])

    return ids

def prepare_train_sequence(seq, to_id):
    """
    Get a tensor of words
    """
    ids = get_train_ids(seq, to_id)

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
