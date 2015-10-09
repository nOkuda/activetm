from __future__ import division

import ankura
import numpy as np
from sklearn.linear_model import Ridge

import abstract
import sampler.ctypesutils as ctypesutils
from .. import labeled

class SupervisedAnchor(abstract.AbstractModel):
    ''''SupervisedAnchor requires the following parameters:
        * numtopics
            the number of topics to look for
        * numtrain
            the number of learners to train
    '''
    def __init__(self, rng, numtopics, numtrain):
        self.rng = rng
        self.numtopics = numtopics
        self.numtrain = numtrain

        # other instance variables initialized in train:
        #   self.corpus_to_train_vocab
        #   self.topicses
        #   self.predictors

    def train(self, dataset, trainingdoc_ids, knownresp,
            continue_training=False):
        # trainingdoc_ids must be a list
        # knownresp must be a list such that its values correspond with trainingdoc_ids
        tmp = ankura.pipeline.Dataset(
                dataset.docwords[:,trainingdoc_ids],
                dataset.vocab,
                [dataset.titles[t] for t in trainingdoc_ids])
        self.corpus_to_train_vocab = [-1] * len(dataset.vocab)
        counter = 0
        for i in range(len(dataset.vocab)):
            if tmp.docwords[i,:].nnz >= 1:
                self.corpus_to_train_vocab[i] = counter
                counter += 1
        filtered = ankura.pipeline.filter_rarewords(tmp, 1)
        labels = {}
        for d, r in zip(trainingdoc_ids, knownresp):
            labels[dataset.titles[d]] = r
        trainingset = labeled.LabeledDataset(filtered, labels)
        self.topicses = []
        self.weightses = []
        for _ in range(self.numtrain):
            anchors = ankura.anchor.gramschmidt_anchors(trainingset,
                    self.numtopics, 0.1 * len(trainingset.titles),
                    1000 if trainingset.vocab_size > 1000 else trainingset.vocab_size)
            topics = ankura.topic.recover_topics(trainingset, anchors,
                    self._get_epsilon(trainingset.num_docs))
            self.topicses.append(topics)
            X = np.zeros((len(trainingset.titles), self.numtopics))
            for d in range(len(trainingset.titles)):
                X[d,:] = ankura.topic.predict_topics(topics,
                        trainingset.doc_tokens(d), rng=self.rng) / \
                        len(trainingset.doc_tokens(d))
            # jitter
            for d in range(X.shape[1]):
                X[d,d] += (self.rng.random()*1e-5) - 5e-6
            weights, residuals, rank, s = np.linalg.lstsq(X, knownresp)
            self.weightses.append(weights)

    def _get_epsilon(self, trainingsize):
        if trainingsize < 1e2:
            return 1e-4
        if trainingsize < 1e3:
            return 1e-5
        if trainingsize < 1e4:
            return 1e-6
        return 1e-7

    def predict(self, doc):
        resultsList = []
        docws = self._convert_vocab_space(doc)
        if len(docws) <= 0:
            return self.rng.random()
        for i in range(self.numtrain):
            X = ankura.topic.predict_topics(self.topicses[i], docws,
                    rng=self.rng) / len(docws)
            guess = np.dot(self.weightses[i], X)
            resultsList.append(guess)
        return np.mean(resultsList)

    def _convert_vocab_space(self, doc):
        result = []
        for token in doc:
            conversion = self.corpus_to_train_vocab[token]
            if conversion >= 0:
                result.append(conversion)
        return result

    def cleanup(self):
        pass

    def get_expected_topic_counts(self, dataset, doc_ids, chain_num, state_num):
        result = []
        for d in doc_ids:
            docws = self._convert_vocab_space(dataset.doc_tokens(d))
            result.append(ankura.topic.predict_topics(self.topicses[chain_num],
                docws, rng=self.rng))
        return result

    def get_topic_distribution(self, topic, chain_num, state_num):
        return np.array(self.topicses[chain_num][:, topic])

