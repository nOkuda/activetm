from __future__ import division
import argparse
import numpy as np
import os
import pickle
import scipy
import sys

from ankura.pipeline import Dataset
from .activetm.labeled import LabeledDataset

'''Magic numbers'''
beta_dist_max = 25
hyperbeta = 0.001
hyperalpha = 0.1

beta_dist_max_minus_1 = beta_dist_max - 1

def parse_args():
    parser = argparse.ArgumentParser(description='generates a pickled labeled '
            'dataset')
    parser.add_argument('seed', type=int, help='random number generator seed')
    parser.add_argument('V', type=int, help='number of token types')
    parser.add_argument('K', type=int, help='number of topics')
    parser.add_argument('D', type=int, help='number of documents')
    parser.add_argument('error', type=float, help='chance of mislabeled '
            'document')
    parser.add_argument('output', help='output filename')
    return parser.parse_args()

def write_args(args):
    args_order = ['seed', 'V', 'K', 'D', 'error', 'output']
    args_dict = vars(args)
    with open(args.output+'.config', 'w') as ofh:
        content = '\n'.join([a+'\t'+str(args_dict[a]) for a in args_order])+'\n'
        ofh.write(content)

def gen_eta(np_rng, K):
    # force alphas and betas to be at least 1
    alphas = (np_rng.rand(K) * beta_dist_max_minus_1) + 1
    betas = (np_rng.rand(K) * beta_dist_max_minus_1) + 1
    return alphas / (alphas + betas)

def gen_phi(np_rng, V, K):
    alpha = np.full(V, hyperbeta)
    return np_rng.dirichlet(alpha, K)

def gen_topic_dist(np_rng, prior):
    return np_rng.dirichlet(prior)

def gen_doc(np_rng, N_d, V, K, topic_dist, phi, eta):
    topic_choices = np_rng.choice(K, size=N_d, p=topic_dist)
    doc_topics = np.bincount(topic_choices, minlength=K)
    words = np.zeros(V, dtype=np.uint64)
    for k in range(K):
        k_count = doc_topics[k]
        if k_count > 0:
            word_choices = np_rng.choice(V, size=k_count, p=phi[k])
            words += np.bincount(word_choices, minlength=V)
    # reshape is stupid but necessary due to scipy's sparse matrix implementation
    return words.reshape((len(words),1)), doc_topics

def gen_label(doc_topics, eta):
    return np.dot(eta, doc_topics / sum(doc_topics))

if __name__ == '__main__':
    '''Generative story according to sLDA'''
    args = parse_args()
    write_args(args)
    np_rng = np.random.RandomState(args.seed)
    eta = gen_eta(np_rng, args.K)
    phi = gen_phi(np_rng, args.V, args.K)

    ''' frus-like
    corpus_doc_size_mean = 383.62065542751679
    corpus_doc_size_var = 179328.38476683316
    # mode will actually be corpus_doc_size_mode+5
    corpus_doc_size_mode = 170
    '''
    corpus_doc_size_mean = 88.360386879730868
    corpus_doc_size_var = 3673.8537209605011
    # mode will actually be corpus_doc_size_mode+5
    corpus_doc_size_mode = 55

    corpus_sigma = np.sqrt(
            np.log(
                (corpus_doc_size_var/corpus_doc_size_mean**2)+1
            )
    )
    doc_sizes = (np_rng.lognormal(np.log(corpus_doc_size_mode)+corpus_sigma**2,
            corpus_sigma, args.D) + 5).astype(int)
    topic_dist_prior = np.array([hyperalpha] * args.K)

    data_matrix = scipy.sparse.dok_matrix((args.V, args.D), dtype=np.uint64)
    titles = []
    labels = {}

    for d in range(args.D):
        N_d = doc_sizes[d]
        topic_dist = gen_topic_dist(np_rng, topic_dist_prior)
        doc_words, doc_topics = gen_doc(np_rng, N_d, args.V, args.K, topic_dist, phi, eta)
        data_matrix[:,d] = doc_words
        if np_rng.rand() < args.error:
            titles.append('e' + str(d))
            labels[titles[-1]] = np_rng.rand()
        else:
            titles.append(str(d))
            labels[titles[-1]] = gen_label(doc_topics, eta)

    unlabeled_dataset = Dataset(data_matrix.tocsc(), [str(a) for a in range(args.V)],
            titles)
    labeled_dataset = LabeledDataset(unlabeled_dataset, labels)
    with open(args.output, 'w') as ofh:
        pickle.dump(labeled_dataset, ofh)

