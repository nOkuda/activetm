from __future__ import division

import utils.aggregate as aggregate
import utils.distance as distance
import utils.represent as represent

def jsd_toptopic_balanced(dataset, labeled_ids, unlabeled_ids, model, rng,
        num_to_choose):
    return aggregate.balanced_centroids(distance.js_divergence,
            represent.top_topic, dataset, labeled_ids, unlabeled_ids,
            model, rng, num_to_choose)
