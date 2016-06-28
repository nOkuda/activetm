from activetm.active.selectors.utils import aggregate
from activetm.active.selectors.utils import distance
from activetm.active.selectors.utils import represent

def jsd_toptopic_centroid(dataset, labeled_ids, unlabeled_ids, model, rng,
        num_to_choose):
    return aggregate.labeled_centroid(distance.js_divergence, represent.top_topic, dataset,
            labeled_ids, unlabeled_ids, model, rng, num_to_choose)
