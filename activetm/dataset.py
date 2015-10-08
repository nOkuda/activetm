from __future__ import division

import numpy as np

import ankura.pipeline

def get_labels(dataset, filename):
    pre_labels = {}
    with open(filename) as ifh:
        for line in ifh:
            data = line.strip().split()
            pre_labels[data[0]] = float(data[1])
    return pre_labels
'''
    labels = []
    for doc_id in range(dataset.num_docs):
        labels.append(pre_labels[dataset.titles[doc_id]])
    return labels
'''

class LabeledDataset(ankura.pipeline.Dataset):
    """Implementation of labeled dataset

    Attributes of the object with the same names as those in
    ankura.pipeline.Dataset have the same behaviors.  The labels are stored in a
    dictionary mapping titles to floats.
    """
    def __init__(self, dataset, labels):
        ankura.pipeline.Dataset.__init__(self, dataset.docwords, dataset.vocab, dataset.titles)
        self.labels = labels

    def _compute_cooccurrences(self):
        ankura.Dataset._compute_cooccurrences(self)
        vanilla_Q = self._cooccurrences
        orig_height, orig_width = vanilla_Q.size
        self._cooccurrences = np.zeros(orig_height, orig_width+2)
        self._cooccurrences[:,:-2] = vanilla_Q
        # summing rows of sparse matrix returns a row matrix; but we want a
        # column matrix
        vocab_counts = self.docwords.sum(axis=1).T
        # multiply word counts per document with corresponding regressand
        regressands = np.zeros(len(self.titles))
        for i, t in enumerate(self.titles):
            if t in self.labels:
                regressands[i] = self.labels[t]
            # note that if the label is not present for a given title, the
            # label for that document is assumed to be zero (this is probably a
            # bad assumption)
        reg_sums = self.docwords.multiply(regressands)
        # divide rows by vocabulary counts
        uncumulated = regressands.multiply(1.0/vocab_counts)
        # sum rows to get average regressand value for each vocabulary item
        self._cooccurrences[:,-2] = uncumulated.sum(axis=1)
        # fill in second augmented column with 1 - average
        self._cooccurrences[:,-1] = 1.0 - self._cooccurrences[:,-2]

