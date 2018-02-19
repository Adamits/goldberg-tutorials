"""
SVD
"""
import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix

import pickle

from collections import Counter

WINDOW = 5

def get_window_counts(text, window):
    """
    Returns a dict of {word: {context_word: co-occurance_counts, context_word: co-occurance_counts}}
    """
    # Each word will have a key in this dict, pointing to
    # a list of every instance of every word in its window
    window_dict = {}
    for i, w in enumerate(text):
        # Add a list of -window and +window words from
        # current word to the list at window_dict[w]
        window_dict[w] = window_dict.setdefault(w, []) + text[i-window:window] + text[i:i+window]

    return {k: Counter(v) for k, v in window_dict.items()}


def ppmi_matrix(sentences, window):
    """
    Returns a scipy sparse row matrix of PPMI
    """
    # Apply a count threshhold for PMI so that rare words are not assigned high values.
    COUNTTHRESHOLD = 5
    words = [w for s in sentences for w in s]
    unigram_counts = Counter(words)
    window_counts = get_window_counts(words, window)
    D = float(len(words))

    # Remove words with counts below the threhhold
    # I don't think we need to update D?
    REMOVE = [k for k, v in unigram_counts.items() if v < COUNTTHRESHOLD]
    print("%i of %i words BEING REMOVED because their frequency is less than %i" % (len(REMOVE), len(words), COUNTTHRESHOLD))

    words = [w for w in words if w not in REMOVE]

    # Normalize the counts
    unigram_probs = {k: float(v) / D for k, v in unigram_counts.items() if k not in REMOVE}
    window_probs = {}
    for word, context in window_counts.items():
        if word not in REMOVE:
            window_probs[word] = {k: float(v) / D for k, v in context.items()}

    # unigram_counts keys are unique
    unique_words = list(unigram_counts.keys())

    # Encode words into matrix coordinates
    wtoi = {i:w for i,w in zip(unique_words, range(len(unique_words)))}
    itow = {v:k for k,v in wtoi.items()}

    PPMI_Mat = np.zeros((len(unique_words),len(unique_words)))

    def _calculate_PPMI(w, c):
        """
        Finds the positive pointwise mutual information between two words

        max(log(p(x, y) / p(x) * p(y)), 0)
        """
        return max(np.log(window_probs[w][c] / (unigram_probs[w] * unigram_probs[c])), 0)

    ############################
    # Populate the PPMI matrix #
    ############################
    print("Populating the %i by %i PPMI matrix..." % (len(unique_words), len(unique_words)))
    for w in unigram_probs.keys():
        for c in unigram_probs.keys():
            # only calculate score if they cooccur
            try:
                # Force it to throw an error immediately if no context exists
                window_probs[w][c]
                print(w, c)
                PPMI_Mat[wtoi[w]][wtoi[c]] = _calculate_PPMI(w, c)
                print(PPMI_Mat[wtoi[w]][wtoi[c]])
            except:
                continue

    return (csc_matrix(PPMI_Mat), wtoi, itow)

"""
Code taken from dmeeting5materials

Read in the entire text of training data from the conLL data
"""
UDWF=1
UDPOS=3
data = [[]]
for line in filter(lambda x: not x or x[0] != '#',
                   map(lambda x:x.strip(), open("./conll_ud_data/en-ud-train.conllu"))):
    if line == '':
        data.append([])
    else:
        fields = line.split()
        data[-1].append((fields[UDWF],fields[UDPOS]))

sentences = []
for s in data:
    sent = []
    for wp in s:
        sent.append(wp[0])
    sentences.append(sent)

"""
Compute the PPMI matrix, and the indexing vectors
"""
ppmi_mat, wtoi, itow = ppmi_matrix(sentences, WINDOW)
print(ppmi_mat.nnz)

#Save sparse matrix
sparse.save_npz('ppmi_mat_window_%i.npz' % WINDOW, ppmi_mat)
#Save indexing
with open('wtoi.pkl', 'wb') as f:
    pickle.dump(wtoi, f, pickle.HIGHEST_PROTOCOL)
#Save lookup
with open('itow.pkl', 'wb') as f:
    pickle.dump(itow, f, pickle.HIGHEST_PROTOCOL)
