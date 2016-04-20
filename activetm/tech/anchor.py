import ankura
import heapq
import numpy as np
from sklearn.linear_model import Ridge

from . import abstract
from .sampler import ctypesutils
from .sampler import slda
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
        self.numsamplesperpredictchain = 5
        # other instance variables initialized in train:
        #   self.corpus_to_train_vocab
        #   self.vocab_size
        #   self.topicses
        #   self.weightses
        #   self.samplers

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
            # note that this uses the same condition as used in
            # ankura.pipeline.filter_rarewords to count only the words that
            # survive after dropping vocabulary words zeroed out as a result of
            # training set selection
            if tmp.docwords[i,:].nnz >= 1:
                self.corpus_to_train_vocab[i] = counter
                counter += 1
        filtered = ankura.pipeline.filter_rarewords(tmp, 1)
        labels = {}
        for d, r in zip(trainingdoc_ids, knownresp):
            labels[dataset.titles[d]] = r
        trainingset = labeled.LabeledDataset(filtered, labels)
        self.vocab_size = trainingset.vocab_size
        self.topicses = []
        self.weightses = []
        self.predictors = []
        # print 'Training set size:', trainingset.M.sum()
        for _ in range(self.numtrain):
            anchors = ankura.anchor.gramschmidt_anchors(trainingset,
                    self.numtopics, 0.1 * len(trainingset.titles),
                    project_dim=1000 if trainingset.vocab_size > 1000 else trainingset.vocab_size)
            topics = ankura.topic.recover_topics(trainingset, anchors,
                    self._get_epsilon(trainingset.num_docs))
            self.topicses.append(topics)
            X = np.zeros((len(trainingset.titles), self.numtopics))
            for d in range(len(trainingset.titles)):
                topic_counts, zs = ankura.topic.predict_topics(
                    topics, trainingset.doc_tokens(d), rng=self.rng)
                X[d,:] = topic_counts / len(trainingset.doc_tokens(d))
            ridge = Ridge()
            ridge.fit(X, np.array(knownresp))
            self.predictors.append(ridge)

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
            X = self._predict_topics(i, docws)
            # make sure that sklearn knows this is only one example with
            # multiple features, not multiple examples with one feature each
            guess = self.predictors[i].predict(X.reshape(1,-1))
            resultsList.append(guess)
        return np.mean(resultsList)

    def _predict_topics(self, pos, docws):
        if len(docws) == 0:
            return np.array([1.0 / self.numtopics] * self.numtopics)
        result = np.zeros(self.numtopics)
        for _ in range(self.numsamplesperpredictchain):
            counts, zs = ankura.topic.predict_topics(
                self.topicses[pos], docws, rng=self.rng)
            result += counts
        result /= (len(docws) * self.numsamplesperpredictchain)
        return result

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
            result.append(self._predict_topics(chain_num, docws))
        return result

    def get_topic_distribution(self, topic, chain_num, state_num):
        return np.array(self.topicses[chain_num][:, topic])

    def get_top_topics(self, dataset, doc_ids):
        pqs = []
        for i in range(len(doc_ids)):
            pqs.append([])
        for i in range(self.numtrain):
            expectedTopicCounts = []
            for j, d in enumerate(doc_ids):
                docws = self._convert_vocab_space(dataset.doc_tokens(d))
                topicCounts = self._predict_topics(i, docws)
                highest = 0.0
                highestTopic = -1
                for k, val in enumerate(topicCounts):
                    if val > highest:
                        highest = val
                        highestTopic = k
                if highestTopic == -1:
                    highestTopic = self.rng.randint(0, self.numtopics-1)
                    highest = self.rng.random()
                heapq.heappush(pqs[j], (-highest, highestTopic, i))
        result = np.zeros((len(doc_ids), self.vocab_size))
        for i, pq in enumerate(pqs):
            (_, highestTopic, i) = heapq.heappop(pq)
            result[i,:] = self.get_topic_distribution(highestTopic, i, 0)
        return result

