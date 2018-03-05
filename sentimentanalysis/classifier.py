# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle
import random

from data import prepare_data, prepare_train_sequence, \
                    prepare_test_sequence, get_train_ids, prepare_label, eval_preds

USE_CUDA = torch.cuda.is_available()

dtype = torch.FloatTensor
if USE_CUDA:
        dtype = torch.cuda.FloatTensor

NUM_SAMPLES = 100

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # Instantiate empty embeddings layer
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # You can initialize with pretrained by:
        # self.word_embeddings.weight = nn.Parameter(embeddings)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to sentiment space,
        # in our case, binary (pos/neg)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden(1)

    def init_hidden(self, batch_size=1):
        # Before we've done anything, we dont have any hidden state.
        # Pytorch’s LSTM expects all of its inputs to be 3D tensors.
        # Note the tuple is of (C, H)
        if USE_CUDA:
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).type(dtype)),
                                    autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim).type(dtype)))
        else:
            return (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)),
                                    autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)))

    def forward(self, docs, batched=False):
        # Lookup the embeddings for each word.
        embeds = self.word_embeddings(docs)
        # Run the LSTM
        if batched:
            lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        else:
            lstm_out, self.hidden = self.lstm(embeds.view(len(docs), 1, -1), self.hidden)

        # Compute score distribution
        sentiment_space = self.hidden2label(lstm_out[-1])
        sentiment_scores = F.log_softmax(sentiment_space)

        return sentiment_scores


class Batch():
    def __init__(self, data, padding_symbol):
        self.symbol = padding_symbol
        self.size = (len(data))
        # Because it is sorted, first input in the batch should have max_length
        self.max_length = len(data[0][0])
        self.input = [d[0] for d in data]
        self.lengths = [len(i) for i in self.input]
        self.output = [d[1] for d in data]

    def input_variable(self, word2id):
        """
        Turn the input into a tensor of batch_size x batch_length

        Returns the input
        """
        tensor = torch.LongTensor(self.size, self.max_length)

        for i, doc in enumerate(self.input):
            ids = get_train_ids(doc, word2id)
            # Pad the difference with symbol
            ids = ids + [word2id[self.symbol]] * (self.max_length - len(doc))
            tensor[i] = torch.LongTensor(ids)

        self.input =  autograd.Variable(tensor)
        return self.input

    def output_variable(self, sent2id):
        """
        Turn the output into a tensor of batch_size

        Returns the output
        """
        tensor = torch.LongTensor([sent2id[s] for s in self.output])

        self.output =  autograd.Variable(tensor)
        return self.output


def train(batch, model, optimizer, loss_function):
    # Clear old gradients
    model.zero_grad()
    # Clear out the hidden state of the LSTM,
    # detaching it from its history on the last instance.
    model.hidden = model.init_hidden(batch.size)

    # Need to transpose so that batch_size is second dim
    preds = model(batch.input.t(), batched=True)

    # Step 4. Pass in the computed scores, and true targets
    # to compute the loss, gradients, and update the parameters by
    # calling optimizer.step().
    loss = loss_function(preds, batch.output)

    # Pytorch knows how to call backward on this apparently..
    loss.backward()
    optimizer.step()

    return loss

def train_iter(model, data, word2id, sentiment2id, loss_function, optimizer, epochs, batch_size):
    # Symbol for padding instances in the same batch to some max_length
    PADDING_SYMBOL = "#"

    # Sort the data by the length of the document, so that batches are of similar length
    sorted_data = data.copy()
    sorted_data.sort(key=lambda x: len(x[0]), reverse=True)

    # Split sorted_data into n batches each of size batch_length
    batches = [sorted_data[i:i+batch_size] for i in range(0, len(sorted_data), batch_size)]
    # Loop over indices so we can modify batches in place
    for i in range(len(batches)):
        batches[i] = Batch(batches[i], PADDING_SYMBOL)
        batches[i].input_variable(word2id)
        batches[i].output_variable(sentiment2id)

    for epoch in range(1, epochs + 1):
        print("EPOCH %i" % epoch)
        random.shuffle(batches)
        losses = []
        for batch in batches:
            loss = train(batch, model, optimizer, loss_function)
            losses.append(loss.data[0])

        # Save the model at the end of each epoch, overwriting the last each time..
        torch.save(model, "./models/sentiment_model_%i" % NUM_SAMPLES)
        # Save the encoding dict for words
        word2id_out = open('./models/word2id_%i.pkl' % NUM_SAMPLES, 'wb')
        pickle.dump(word2id, word2id_out)

        print(sum(losses) / len(losses))

if __name__=='__main__':
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 300

    # Prepare data, and store ids
    print("Preparing data...")
    data, vocab = prepare_data("./aclimdb/train/", sample_size=NUM_SAMPLES, randomize=False)

    # Compute dicts for getting ids from word/label
    word2id = {w:i for i,w in enumerate(vocab)}
    sentiment2id = {"neg": 0, "pos": 1}
    id2sentiment = {0: "neg", 1: "pos"}

    # Initialze the LSTM
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2id), len(sentiment2id))
    # Negative Log Likelihood Loss
    loss_function = nn.NLLLoss()

    if USE_CUDA:
        model = model.cuda()
        loss_function = loss_function.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.08)

    train_iter(model, data, word2id, sentiment2id, loss_function, optimizer, 20, 5)

    """
    TEST THE MODEL OVER the test docs
    """

    # Just test it on the same data to see if it overfits.
    train_data, train_vocab = prepare_data("./aclimdb/train/", sample_size=NUM_SAMPLES, randomize=False)

    # For tracking preds
    preds = []
    golds = []
    for doc, sentiment in train_data:
        doc_in = prepare_test_sequence(doc, word2id)
        if USE_CUDA:
            doc_in = doc_in.cuda()
        pred = model(doc_in)
        # Get the predicted  class, from the log_softmax distribution
        pred_label = pred.data.max(1)[1][0]
        preds.append(pred_label)
        golds.append(sentiment2id[sentiment])

        print(doc)
        print("TRUE SENTIMENT ID: %i" % sentiment2id[sentiment])
        print("PREDICTED SENTIMENT SCORE: %i" % pred_label)

    print("Accuracy: %.4f%%" % eval_preds(preds, golds))
