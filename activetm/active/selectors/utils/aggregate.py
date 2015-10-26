from __future__ import division
import copy
import logging
import numpy as np

def labeled_centroid(distance, represent, dataset, labeled_ids, unlabeled_ids,
        model, rng, num_to_choose):
    if len(unlabeled_ids) <= num_to_choose:
        logging.getLogger(__name__).info("num_to_choose > len(unlabeled_ids); \
                returned copy of unlabeled_ids")
        return copy.deepcopy(unlabeled_ids)
    sums = np.array(represent(dataset, labeled_ids, model, rng))
    sums = sums.sum(axis=0) # sum columns
    labeled_count = len(labeled_ids)
    unlabeled_representations =  represent(dataset, unlabeled_ids, model, rng)
    result = set()
    while len(result) < num_to_choose:
        cur_centroid = sums / labeled_count
        best_score = float('-inf')
        best_i = -1
        for i, doc_id in enumerate(unlabeled_ids):
            if doc_id in result:
                continue
            cur_dist = distance(cur_centroid,
                    unlabeled_representations[i])
            if cur_dist > best_score:
                best_i = i
                best_score = cur_dist
        result.add(unlabeled_ids[best_i])
        sums += unlabeled_representations[best_i]
        labeled_count += 1
    return list(result)

# reward being far from labeled set; penalize being far from unlabeled set
alpha = 0.5
beta = 0.5
def balanced_centroids(distance, represent, dataset, labeled_ids, unlabeled_ids,
        model, rng, num_to_choose):
    if len(unlabeled_ids) <= num_to_choose:
        logging.getLogger(__name__).info("num_to_choose > len(unlabeled_ids);\
                returned copy of unlabeled_ids")
        return copy.deepcopy(unlabeled_ids)
    labeled_sums = np.array(represent(dataset, labeled_ids, model,
            rng)).sum(axis=0)
    labeled_count = len(labeled_ids)
    unlabeled_representations = represent(dataset, unlabeled_ids, model, rng)
    unlabeled_sums = np.array(unlabeled_representations).sum(axis=0)
    unlabeled_count = len(unlabeled_ids)
    result = set()
    while len(result) < num_to_choose:
        labeled_centroid = labeled_sums / labeled_count
        best_score = float('-inf')
        best_i = -1
        for i, doc_id in enumerate(unlabeled_ids):
            if doc_id in result:
                continue
            cur_repr = unlabeled_representations[i]
            unlabeled_centroid = (unlabeled_sums - cur_repr)\
                    / unlabeled_count
            labeled_dist = distance(labeled_centroid, cur_repr)
            unlabeled_dist = distance(unlabeled_centroid, cur_repr)
            score = (alpha * labeled_dist) - (beta * unlabeled_dist)
            if score > best_score:
                best_score = score
                best_i = i
        result.add(unlabeled_ids[best_i])
        best_repr = unlabeled_representations[best_i]
        labeled_sums += best_repr
        labeled_count += 1
        unlabeled_sums -= best_repr
        unlabeled_count -= 1
    return list(result)
