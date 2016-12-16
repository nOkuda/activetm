"""A supervised anchor words implementation for regression"""

import heapq

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.linear_model import Ridge
import numpy as np

import ankura

from . import abstract
from .. import labeled


def get_filtered_for_train(dataset, trainingdoc_ids):
    """Get filtered Dataset

    Return a Dataset with only train_doc_ids in its docwords matrix; vocabulary
    is filtered for word types that no longer exist in Dataset
    """
    tmp = ankura.pipeline.Dataset(dataset.docwords[:, trainingdoc_ids],
                                  dataset.vocab,
                                  [dataset.titles[tid] for tid in trainingdoc_ids])
    corpus_to_train_vocab = [-1] * len(dataset.vocab)
    counter = 0
    for i in range(len(dataset.vocab)):
        # note that this uses the same condition as used in
        # ankura.pipeline.filter_rarewords to count only the words that
        # survive after dropping vocabulary words zeroed out as a result of
        # training set selection
        if tmp.docwords[i, :].nnz >= 1:
            corpus_to_train_vocab[i] = counter
            counter += 1
    filtered = ankura.pipeline.filter_rarewords(tmp, 1)
    return filtered, corpus_to_train_vocab


def get_labels_for_train(dataset, trainingdoc_ids, knownresp):
    """Get labels for train_doc_ids"""
    labels = {}
    for doc, resp in zip(trainingdoc_ids, knownresp):
        labels[dataset.titles[doc]] = resp
    return labels


def buildsupervised(dataset, trainingdoc_ids, knownresp):
    """Construct training set and vocab mapping for supervised learning"""
    filtered, corpus_to_train_vocab = get_filtered_for_train(dataset,
                                                             trainingdoc_ids)
    labels = get_labels_for_train(dataset, trainingdoc_ids, knownresp)
    trainingset = labeled.LabeledDataset(filtered, labels)
    return trainingset, corpus_to_train_vocab, list(range(0, len(trainingdoc_ids)))


def buildsemisupervised(dataset, trainingdoc_ids, knownresp):
    """Construct training set and vocab mapping for semi-supervised learning"""
    labels = get_labels_for_train(dataset, trainingdoc_ids, knownresp)
    trainingset = labeled.LabeledDataset(dataset, labels)
    return trainingset, list(range(0, len(dataset.vocab))), trainingdoc_ids


#pylint:disable=too-many-instance-attributes
class AbstractAnchor(abstract.AbstractModel):
    """Parent class for other supervised anchor words implementations"""

    #pylint:disable=too-many-arguments
    def __init__(self,
                 rng,
                 numtopics,
                 numtrain,
                 regressor,
                 trainingsetbuilder,
                 expgrad_epsilon):
        """AbstractAnchor requires the following parameters:
            * rng :: random.Random
                a random number generator
            * numtopics :: int
                the number of topics to look for
            * numtrain :: int
                the number of models to train (for query by committee)
            * regressor
                factory for sklearn-style learner trained as regression model
            * trainingsetbuilder
                function that constructs the training set and a mapping of vocab
                space from corpus to training set
            * expgrad_epsilon :: float
                epsilon for exponentiated gradient descent
        """
        self.rng = rng
        self.numtopics = numtopics
        self.numtrain = numtrain
        self.regressor = regressor
        self.trainingsetbuilder = trainingsetbuilder
        self.numsamplesperpredictchain = 5
        self.expgrad_epsilon = expgrad_epsilon
        # other instance variables initialized in train:
        #   self.corpus_to_train_vocab
        #   self.vocab_size
        #   self.topicses
        #   self.weightses
        #   self.anchorses

    #pylint:disable=attribute-defined-outside-init,too-many-locals
    def train(self,
              dataset,
              trainingdoc_ids,
              knownresp,
              continue_training=False):
        """training algorithm for supervised anchor words

            * dataset :: activetm.labeled.LabeledDataset
                the complete corpus used for the experiments
            * trainingdoc_ids :: [int]
                the documents in the corpus identified by title order in dataset
            * knownresp :: [float]
                true labels corresponding to documents identified by
                trainingdoc_ids
            * continue_training :: bool
                are we picking up from parameters trained earlier?
        """
        trainingset, self.corpus_to_train_vocab, doc_ids = self.trainingsetbuilder(
            dataset,
            trainingdoc_ids,
            knownresp)
        self.vocab_size = trainingset.vocab_size
        self.topicses = []
        self.weightses = []
        self.predictors = []
        self.anchorses = []
        # print 'Training set size:', trainingset.M.sum()
        for _ in range(self.numtrain):
            pdim = 1000 if trainingset.vocab_size > 1000 else trainingset.vocab_size
            anchors, anchor_indices = \
                ankura.anchor.gramschmidt_anchors(trainingset,
                                                  self.numtopics,
                                                  0.015 * len(trainingset.titles),
                                                  project_dim=pdim,
                                                  return_indices=True)
            '''
            sorted_anchors = np.sort(anchor_indices)
            if (sorted_anchors[1:] == sorted_anchors[:-1]).any():
                print('Duplicate anchor word!')
                for i in range(self.numtopics):
                    print(anchor_indices[i])
            '''
            self.anchorses.append(anchor_indices)
            # print('# Finished gramschmidt')
            topics = ankura.topic.recover_topics(trainingset,
                                                 anchors,
                                                 self.expgrad_epsilon)
            self.topicses.append(topics)
            features = np.zeros((len(trainingdoc_ids), self.numtopics))
            for i, doc in enumerate(doc_ids):
                topic_counts, _ = ankura.topic.predict_topics(topics,
                                                              trainingset.doc_tokens(doc),
                                                              rng=self.rng)
                features[i, :] = topic_counts / len(trainingset.doc_tokens(doc))
            predictor = self.regressor()
            predictor.fit(features, np.array(knownresp))
            self.predictors.append(predictor)

    def predict(self, doc):
        resultslist = []
        docws = self._convert_vocab_space(doc)
        if len(docws) <= 0:
            return self.rng.random()
        for i in range(self.numtrain):
            features = self._predict_topics(i, docws)
            # make sure that sklearn knows this is only one example with
            # multiple features, not multiple examples with one feature each
            guess = self.predictors[i].predict(features.reshape((1, -1)))
            resultslist.append(guess)
        return np.mean(resultslist)

    def _predict_topics(self, pos, docws):
        """Predict topic assignments according to self.topicses[pos]"""
        if len(docws) == 0:
            return np.array([1.0 / self.numtopics] * self.numtopics)
        result = np.zeros(self.numtopics)
        for _ in range(self.numsamplesperpredictchain):
            counts, _ = ankura.topic.predict_topics(
                self.topicses[pos], docws, rng=self.rng)
            result += counts
        result /= (len(docws) * self.numsamplesperpredictchain)
        return result

    def _convert_vocab_space(self, doc):
        """Get vocabulary in training set space"""
        result = []
        for token in doc:
            conversion = self.corpus_to_train_vocab[token]
            if conversion >= 0:
                result.append(conversion)
        return result

    def cleanup(self):
        pass

    def get_topic_compositions(self, dataset, doc_ids):
        """Get topic compositions"""
        result = []
        for doc in doc_ids:
            docws = self._convert_vocab_space(dataset.doc_tokens(doc))
            tmp = []
            for chain_num in range(len(self.topicses)):
                tmp.append(self._predict_topics(chain_num, docws))
            result.append(np.mean(tmp, axis=0))
        return np.array(result)

    def get_topic_distribution(self, topic, chain_num, _):
        """Get top distribution"""
        return np.array(self.topicses[chain_num][:, topic])

    def get_top_topics(self, dataset, doc_ids):
        pqs = self._fill_pqs(dataset, doc_ids)
        result = np.zeros((len(doc_ids), self.vocab_size))
        for i, priorityq in enumerate(pqs):
            (_, highesttopic, i) = heapq.heappop(priorityq)
            result[i, :] = self.get_topic_distribution(highesttopic, i, 0)
        return result

    def _fill_pqs(self, dataset, doc_ids):
        """Get list of priority queues filled with topic counts"""
        pqs = []
        for i in range(len(doc_ids)):
            pqs.append([])
        for i in range(self.numtrain):
            for j, doc in enumerate(doc_ids):
                docws = self._convert_vocab_space(dataset.doc_tokens(doc))
                topiccounts = self._predict_topics(i, docws)
                highest = 0.0
                highesttopic = -1
                for k, val in enumerate(topiccounts):
                    if val > highest:
                        highest = val
                        highesttopic = k
                if highesttopic == -1:
                    highesttopic = self.rng.randint(0, self.numtopics-1)
                    highest = self.rng.random()
                heapq.heappush(pqs[j], (-highest, highesttopic, i))
        return pqs

    def get_uncertainty(self, doc):
        """Get uncertainty"""
        return np.var([self.predict(doc) for _ in range(self.numsamplesperpredictchain)])


class RidgeAnchor(AbstractAnchor):
    """Supervised anchor words implementation using ridge regression"""

    def __init__(self, rng, numtopics, numtrain, expgrad_epsilon):
        super(RidgeAnchor, self).__init__(rng,
                                          numtopics,
                                          numtrain,
                                          Ridge,
                                          buildsupervised,
                                          expgrad_epsilon)


class SemiRidgeAnchor(AbstractAnchor):
    """Semisupervised anchor words implementation using ridge regression"""

    def __init__(self, rng, numtopics, numtrain, expgrad_epsilon):
        super(SemiRidgeAnchor, self).__init__(rng,
                                              numtopics,
                                              numtrain,
                                              Ridge,
                                              buildsemisupervised,
                                              expgrad_epsilon)


def build_gp():
    """Build Gaussian process"""
    return GaussianProcessRegressor(kernel=DotProduct(), normalize_y=True)


class AbstractGPAnchor(AbstractAnchor):
    """Abstract class for anchor words implementation using Gaussian process"""

    #pylint:disable=too-many-arguments
    def __init__(self,
                 rng,
                 numtopics,
                 numtrain,
                 regressor,
                 buildtrainingset,
                 expgrad_epsilon):
        """builds SupervisedAnchor to use Gaussian process"""
        super(AbstractGPAnchor, self).__init__(rng,
                                               numtopics,
                                               numtrain,
                                               build_gp,
                                               buildtrainingset,
                                               expgrad_epsilon)

    def get_uncertainty(self, doc):
        """Get uncertainty"""
        docws = self._convert_vocab_space(doc)
        uncertainties = []
        for i in range(self.numtrain):
            uncertainties.append(
                self.predictors[i](
                    self._predict_topics(i, docws), return_std=True)[1][0])
        return np.mean(uncertainties)


class GPAnchor(AbstractGPAnchor):
    """Supervised anchor words implementation using Gaussian process"""

    def __init__(self, rng, numtopics, numtrain, expgrad_epsilon):
        super(GPAnchor, self).__init__(rng,
                                       numtopics,
                                       numtrain,
                                       build_gp,
                                       buildsupervised,
                                       expgrad_epsilon)


class SemiGPAnchor(AbstractGPAnchor):
    """Semisupervised anchor words implementation using Gaussian process"""

    def __init__(self, rng, numtopics, numtrain, expgrad_epsilon):
        super(SemiGPAnchor, self).__init__(rng,
                                           numtopics,
                                           numtrain,
                                           build_gp,
                                           buildsemisupervised,
                                           expgrad_epsilon)

