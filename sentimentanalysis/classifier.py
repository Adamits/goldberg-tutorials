import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import prepare_data

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

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
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        # Get the embeddings for each word
        embeds = self.word_embeddings(sentence)
        # Run the LSTM
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        # Compute score distribution
        sentiment_space = self.hidden2label(lstm_out[-1])
        sentiment_scores = F.log_softmax(sentiment_space, dim=1)

        return sentiment_scores

def prepare_sequence(seq, to_id):
    """
    Get a tensor of words
    """
    ids = [to_id[w] for w in seq]
    tensor = torch.LongTensor(ids)
    return autograd.Variable(tensor)

def prepare_label(label, to_id):
    return autograd.Variable(torch.LongTensor([to_id[label]]))

if __name__=='__main__':
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 100

    # Prepare data, and store ids
    print("Preparing data...")
    data, vocab = prepare_data("./aclimdb/train/")
    word2id = {w:i for i,w in enumerate(vocab)}
    sentiment2id = {"neg": 0, "pos": 1}
    id2sentiment = {0: "neg", 1: "pos"}

    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word2id), len(sentiment2id))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    #TODO Add batching
    #dataloader = DataLoader(data, batch_size=4,
    #                    shuffle=True, num_workers=4)

    for i, epoch in enumerate(range(100)):
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

            # Step 3. Run our forward pass.
            scores = model(doc_in)
            pred = scores

            # Step 4. Pass in the computed scores, and true targets
            # to compute the loss, gradients, and update the parameters by
            # calling optimizer.step().
            loss = loss_function(pred, label)
            loss_tracker.append(loss.data[0])
            loss.backward()
            optimizer.step()

        print("AVERAGE LOSS: %2f" % (sum(loss_tracker)/(len(loss_tracker))))
