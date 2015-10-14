class AbstractModel:
    def train(self, trainingset, knownlabels, continue_training=False):
        raise NotImplementedError()
    def predict(self, doc):
        raise NotImplementedError()
    def cleanup(self):
        raise NotImplementedError()
    def get_expected_topic_counts(self, dataset, doc_ids, chain_num, state_num):
        raise NotImplementedError()
    def get_topic_distribution(self, topic, chain_num, state_num):
        raise NotImplementedError()
    def get_top_topics(self, dataset, doc_ids):
        raise NotImplementedError()
