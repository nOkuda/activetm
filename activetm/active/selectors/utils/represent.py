"""Document representations"""


def top_topic(dataset, doc_ids, model, _):
    """Represent document by its most prevalent topic"""
    return model.get_top_topics(dataset, doc_ids)


def topic_composition(dataset, doc_ids, model, _):
    """Represent document by its topic composition"""
    return model.get_topic_compositions(dataset, doc_ids)
