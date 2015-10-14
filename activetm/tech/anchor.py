from __future__ import division

import ankura
import ctypes
import heapq
import numpy as np

import abstract
import sampler.ctypesutils as ctypesutils
import sampler.slda as slda
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

        self.alphas = (ctypes.c_double * numtopics)()
        for i in range(numtopics):
            self.alphas[i] = 0.1
        self.hyperbeta = ctypes.c_double(0.01)

        self.numsamplesperpredictchain = 5
        predictschedule = [50] * self.numsamplesperpredictchain
        predictschedule[0] = 500
        self.predictschedarr = ctypesutils.convertFromIntList(predictschedule)

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
        self.samplers = []
        # print 'Training set size:', trainingset.M.sum()
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
                X[d,d] += (self.rng.random()*1e-20) - 5e-21
            weights, residuals, rank, s = np.linalg.lstsq(X, knownresp)
            self.weightses.append(weights)
            '''
            pretopicwordcounts = topics.T
            vocabcounts = trainingset.M.sum(axis=1).A1
            for i in range(trainingset.vocab_size):
                pretopicwordcounts[:,i] *= vocabcounts[i] /\
                        pretopicwordcounts[:,i].sum()
            print pretopicwordcounts
            print pretopicwordcounts.sum()
            topicwordcounts = (ctypes.POINTER(ctypes.c_int) * self.numtopics)()
            for i in range(self.numtopics):
                topicwordcounts[i] = (ctypes.c_int * trainingset.vocab_size)()
                for j in range(trainingset.vocab_size):
                    topicwordcounts[i][j] = ctypes.c_int(
                            int(round(pretopicwordcounts[i][j])))
            numVocabList = [trainingset.vocab_size] * self.numtopics
            print np.array(ctypesutils.convertToListOfLists(topicwordcounts,
                    numVocabList)).sum()
            prepresums = np.array(ctypesutils.convertToListOfLists(topicwordcounts,
                        numVocabList))
            presums = np.sum(prepresums, axis=1)
            topicwordsum = ctypesutils.convertFromIntList(presums)
            # note that I am using self.alpha and self.hyperbeta in the
            # constructor twice because the constructor requires an eta and var;
            # however, since the prediction topic sampling does not use eta nor
            # var, I have merely reused self.alpha and self.hyperbeta to avoid
            # superfluous code.  Note also that other unnecessary pointers are
            # set to null.
            samplerState = slda.SamplerState(ctypes.c_int(self.numtopics),
                    ctypes.c_int(trainingset.vocab_size), self.alphas,
                    self.hyperbeta, self.alphas, self.hyperbeta,
                    None, None,
                    None, topicwordcounts, topicwordsum)
            self.samplers.append(samplerState)
            '''

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
            guess = np.dot(self.weightses[i], X)
            resultsList.append(guess)
        return np.mean(resultsList)

    def _predict_topics(self, pos, docws):
        result = np.zeros(self.numtopics)
        for _ in range(self.numsamplesperpredictchain):
            result += ankura.topic.predict_topics(self.topicses[pos], docws,
                    rng=self.rng)
        result /= (len(docws) * self.numsamplesperpredictchain)
        # return ankura.topic.predict_topics(self.topicses[pos], docws, rng=self.rng) / len(docws)
        '''
        cResults = slda.cPredict(ctypes.c_int(1), ctypes.pointer(self.samplers[pos]),
                ctypes.c_int(len(docws)), ctypesutils.convertFromIntList(docws),
                self.numsamplesperpredictchain, self.predictschedarr)
        result = np.mean(ctypesutils.convertToList(cResults,
            self.numsamplesperpredictchain))
        slda.freeDoubleArray(cResults)
        '''
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
        '''With sampling
        knownwords = (ctypes.POINTER(ctypes.c_int) * len(doc_ids))()
        docSizes = []
        for i, c in enumerate(doc_ids):
            curDoc = self._convert_vocab_space(dataset.doc_tokens(c))
            docSizes.append(len(curDoc))
            knownwords[i] = ctypesutils.convertFromIntList(curDoc)
        expectedTopicCounts = slda.getExpectedTopicCounts(self.samplers[chain_num], 
                ctypes.c_int(len(doc_ids)),
                ctypesutils.convertFromIntList(docSizes),
                knownwords, self.numsamplesperpredictchain,
                self.predictschedarr)
        result = ctypesutils.convertToTlistOfLists(expectedTopicCounts,
                [self.numtopics]*len(doc_ids))
        freeDoubleMatrix(expectedTopicCounts, ctypes.c_int(len(doc_ids)))
        return result
        '''

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
                    highest = rng.random()
                heapq.heappush(pqs[j], (-highest, highestTopic, i))
        result = np.zeros((len(doc_ids), self.vocab_size))
        for i, pq in enumerate(pqs):
            (_, highestTopic, i) = heapq.heappop(pq)
            result[i,:] = self.get_topic_distribution(highestTopic, i, 0)
        return result

