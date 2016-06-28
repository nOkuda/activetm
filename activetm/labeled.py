"""LabeledDataset for labeled datasets"""
import numpy as np

import ankura.pipeline

def get_labels(filename):
    """Since labels are assumed to be normalized from 0 to 1 in computing the
    augmented Q matrix, this function will scale all labels to fit this range
    """
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
        for label in labels:
            labels[label] = 0.5
    elif abs(difference - 1) > 1e-50:
        for label in labels:
            labels[label] = (labels[label] - smallest) / difference
    # otherwise, the labels were already spanning the range 0 to 1, so no need
    # to change anything
    return labels

class LabeledDataset(ankura.pipeline.Dataset):
    """Implementation of labeled dataset

    Attributes of the object with the same names as those in
    ankura.pipeline.Dataset have the same behaviors.  The labels are stored in a
    dictionary mapping titles to floats.
    """

    def __init__(self, dataset, labels):
        ankura.pipeline.Dataset.__init__(self, dataset.docwords, dataset.vocab, dataset.titles)
        self.labels = labels
        # precompute vanilla Q beforehand (useful for semi-supervised)
        self._dataset_cooccurrences = ankura.pipeline.Dataset.compute_cooccurrences(self)

    def compute_cooccurrences(self):
        orig_height, orig_width = self._dataset_cooccurrences.shape
        self._cooccurrences = np.zeros((orig_height, orig_width+2))
        self._cooccurrences[:, :-2] = self._dataset_cooccurrences
        # multiply word counts per document with corresponding regressand
        regressands = []
        labeled_docs = []
        for i, title in enumerate(self.titles):
            if title in self.labels:
                regressands.append(self.labels[title])
                labeled_docs.append(i)
        # TODO extract information directly (indexing into matrix is slow)
        labeled_docwords = self.docwords[:, np.array(labeled_docs)]
        reg_sums = labeled_docwords.dot(np.array(regressands))
        # summing rows of sparse matrix returns a row matrix; but we want a
        # numpy array
        vocab_counts = np.array(labeled_docwords.sum(axis=1).T)[0]
        # divide rows by vocabulary counts
        uncumulated = np.zeros(reg_sums.shape)
        for i in range(len(vocab_counts)):
            if vocab_counts[i] > 0:
                uncumulated[i, :] = reg_sums[i, :] / vocab_counts[i]
            # if vocab_counts[i] == 0, reg_sums[i, :] == np.zeros
        # TODO was the above sufficient for making semi-supervised work?
        # sum rows to get average regressand value for each vocabulary item
        self._cooccurrences[:, -2] = uncumulated.sum(axis=1)
        # fill in second augmented column with 1 - average
        self._cooccurrences[:, -1] = 1.0 - self._cooccurrences[:, -2]

