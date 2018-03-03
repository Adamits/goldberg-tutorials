# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle

from data import prepare_data, prepare_sequence, prepare_label, eval_preds

USE_CUDA = torch.cuda.is_available()

dtype = torch.FloatTensor
if USE_CUDA:
        dtype = torch.cuda.FloatTensor

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
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Pytorch’s LSTM expects all of its inputs to be 3D tensors.
        # The semantics of the axes of these tensors is important.
        # The first axis is the sequence itself,
        # the second indexes instances in the mini-batch,
        # and the third indexes elements of the input.
        # (num_layers, minibatch_size, hidden_dim)
        if USE_CUDA:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim).type(dtype)),
                                    autograd.Variable(torch.zeros(1, 1, self.hidden_dim).type(dtype)))
        else:
            return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)), autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, doc):
        # Get the embeddings for each word
        embeds = self.word_embeddings(doc)
        # Run the LSTM
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(doc), 1, -1), self.hidden)
        # Compute score distribution
        sentiment_space = self.hidden2label(lstm_out[-1])
        sentiment_scores = F.log_softmax(sentiment_space)

        return sentiment_scores

if __name__=='__main__':
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 300

    # Prepare data, and store ids
    print("Preparing data...")
    data, vocab = prepare_data("./aclimdb/train/", sample_size=100)

    word2id = {w:i for i,w in enumerate(vocab)}
    sentiment2id = {"neg": 0, "pos": 1}
    id2sentiment = {0: "neg", 1: "pos"}

    # Save the encoding dict for words
    word2id_out = open('./models/word2id.pkl', 'wb')
    pickle.dump(word2id, word2id_out)

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2id), len(sentiment2id))
    loss_function = nn.NLLLoss()

    if USE_CUDA:
        model = model.cuda()
        loss_function = loss_function.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.1)

    #TODO Add batching
    #dataloader = DataLoader(data, batch_size=4,
    #                    shuffle=True, num_workers=4)

    for i, epoch in enumerate(range(50)):
        print("EPOCH %i" % i)
        loss_tracker = []
        for doc, sentiment in data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # No batching, so we'll just hardcode this to 1
            model.batch_size = 1

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Variables of word ids.
            doc_in = prepare_sequence(doc, word2id)
            label = prepare_label(sentiment, sentiment2id)

            if USE_CUDA:
                doc_in = doc_in.cuda()
                label = label.cuda()

            # Step 3. Run our forward pass.
            pred = model(doc_in)

            # Step 4. Pass in the computed scores, and true targets
            # to compute the loss, gradients, and update the parameters by
            # calling optimizer.step().
            loss = loss_function(pred, label)
            loss_tracker.append(loss.data[0])
            loss.backward()
            optimizer.step()

            # Save the model at the end of each epoch, overwriting the last each time..
            torch.save(model, "./models/sentiment_model")

        print("AVERAGE LOSS: %2f" % (sum(loss_tracker)/(len(loss_tracker))))

    """
    TEST THE MODEL OVER the test docs
    """
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
