import numpy as np

import ankura.pipeline

def get_labels(filename):
    '''Since labels are assumed to be normalized from 0 to 1 in computing the
    augmented Q matrix, this function will scale all labels to fit this range
    '''
    smallest = float('inf')
    largest = float('-inf')
    labels = {}
    with open(filename) as ifh:
        for line in ifh:
            data = line.strip().split()
            val = float(data[1])
            labels[data[0]] = val
            if val < smallest:
                smallest = val
            if val > largest:
                largest = val
    difference = largest - smallest
    if difference < 1e-50:
        # all of the label values were essentially the same, so just assign
        # everything to have the same label
        for d in labels:
            labels[d] = 0.5
    elif abs(difference - 1) > 1e-50:
        for d in labels:
            labels[d] = (labels[d] - smallest) / difference
    # otherwise, the labels were already spanning the range 0 to 1, so no need
    # to change anything
    return labels

class LabeledDataset(ankura.pipeline.Dataset):
    '''Implementation of labeled dataset

    Attributes of the object with the same names as those in
    ankura.pipeline.Dataset have the same behaviors.  The labels are stored in a
    dictionary mapping titles to floats.
    '''
    def __init__(self, dataset, labels):
        ankura.pipeline.Dataset.__init__(self, dataset.docwords, dataset.vocab, dataset.titles)
        self.labels = labels

    def compute_cooccurrences(self):
        ankura.pipeline.Dataset.compute_cooccurrences(self)
        vanilla_Q = self._cooccurrences
        orig_height, orig_width = vanilla_Q.shape
        self._cooccurrences = np.zeros((orig_height, orig_width+2))
        self._cooccurrences[:,:-2] = vanilla_Q
        # summing rows of sparse matrix returns a row matrix; but we want a
        # numpy array
        vocab_counts = np.array(self.docwords.sum(axis=1).T)[0]
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
        uncumulated = np.zeros(reg_sums.shape)
        for i in range(len(vocab_counts)):
            uncumulated[i,:] = reg_sums[i,:] / vocab_counts[i]
        # sum rows to get average regressand value for each vocabulary item
        self._cooccurrences[:,-2] = uncumulated.sum(axis=1)
        # fill in second augmented column with 1 - average
        self._cooccurrences[:,-1] = 1.0 - self._cooccurrences[:,-2]

