import copy
import logging
from activetm.active.selectors import distance_diversity
from activetm.active.selectors import balanced

def reservoir(candidates, rnd, num_to_choose):
    candidates_size = len(candidates)
    if candidates_size < num_to_choose:
        logger = logging.getLogger(__name__)
        logger.info('Not enough candidates to perform reservoir sampling!')
        return copy.deepcopy(candidates)
    result = copy.deepcopy(candidates[:num_to_choose])
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
    'jsd_toptopic_balanced': balanced.jsd_toptopic_balanced,
    'jsd_comp_centroid': distance_diversity.jsd_comp_centroid
}

