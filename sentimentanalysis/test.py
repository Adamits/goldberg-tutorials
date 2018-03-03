"""
TEST THE MODEL OVER THE FIRST 5000 pos examples and 5000 neg examples from the test docs
"""
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from classifier import *

import pickle

from data import prepare_data, prepare_sequence, prepare_label, eval_preds

if __name__=='__main__':
    # LOAD DICTS
    word2id_file = open('./models/word2id.pkl', 'rb')
    word2id = pickle.load(word2id_file)
    # Label dicts should be the same always..
    sentiment2id = {"neg": 0, "pos": 1}
    id2sentiment = {0: "neg", 1: "pos"}

    # LOAD THE MODEL
    model = torch.load("./models/sentiment_model")

    train_data, train_vocab = prepare_data("./aclimdb/test/", sample_size=100)

    # For tracking preds
    preds = []
    golds = []
    for doc, sentiment in train_data:
        doc_in = prepare_sequence(doc, word2id)
        if USE_CUDA:
            doc_in = doc_in.cuda()
        pred=model(doc_in)
        # Get the predicted  class, from the log_softmax distribution
        pred_label = pred.data.max(1)[1][0]
        preds.append(pred_label)
        golds.append(sentiment2id[sentiment])

        print(doc)
        print("TRUE SENTIMENT ID: %i" % sentiment2id[sentiment])
        print("PREDICTED SENTIMENT SCORE: %i" % pred_label)

    print("Accuracy: %.4f%%" % eval_preds(preds, golds))
