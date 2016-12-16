"""Diversity selection methods"""
from activetm.active.selectors.utils import aggregate
from activetm.active.selectors.utils import distance
from activetm.active.selectors.utils import represent


#pylint:disable-msg=too-many-arguments
def jsd_toptopic_centroid(
        dataset,
        labeled_ids,
        unlabeled_ids,
        model,
        rng,
        num_to_choose):
    """Choose by JSD of top topic between document and labeled set centroid"""
    return aggregate.labeled_centroid(
        distance.js_divergence,
        represent.top_topic,
        dataset,
        labeled_ids,
        unlabeled_ids,
        model,
        rng,
        num_to_choose)


#pylint:disable-msg=too-many-arguments
def jsd_comp_centroid(
        dataset,
        labeled_ids,
        unlabeled_ids,
        model,
        rng,
        num_to_choose):
    """Choose by JSD of topic composition between document and labeled set
    centroid"""
    return aggregate.labeled_centroid(
        distance.js_divergence,
        represent.topic_composition,
        dataset,
        labeled_ids,
        unlabeled_ids,
        model,
        rng,
        num_to_choose)
