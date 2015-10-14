from __future__ import division
import numpy as np

def top_topic(dataset, doc_ids, model, rng):
    return model.get_top_topics(dataset, doc_ids)
