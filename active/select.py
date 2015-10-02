from __future__ import division
import selectors.distance_diversity as distance_diversity
import selectors.balanced as balanced

def reservoir(candidates, rnd, num_to_choose):
    candidates_size = len(candidates)
    if candidates_size < num_to_choose:
        print 'Not enough candidates to perform reservoir sampling!'
        return candidates
    result = candidates[:num_to_choose]
    for i in range(num_to_choose, candidates_size):
        j = rnd.randint(0, i-1)
        if j < num_to_choose:
            result[j] = candidates[i]
    return result

def random(dataset, labeled_ids, candidates, model, rnd, num_to_choose):
    return reservoir(candidates, rnd, num_to_choose)

factory = {
    'random': random,
    'jsd_toptopic_centroid': distance_diversity.jsd_toptopic_centroid,
    'jsd_toptopic_balanced': balanced.jsd_toptopic_balanced
}

