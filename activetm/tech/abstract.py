"""Abstract class for models"""
class AbstractModel:
    """Abstract class for models"""

    def train(self, dataset, trainingdoc_ids, knownlabels, continue_training=False):
        """Train the model"""
        raise NotImplementedError()

    def predict(self, doc):
        """Predict label for doc"""
        raise NotImplementedError()

    def cleanup(self):
        """Release resources taken by this model"""
        raise NotImplementedError()

    def get_top_topics(self, dataset, doc_ids):
        """Get top topics"""
        raise NotImplementedError()

    def get_topic_compositions(self, dataset, doc_ids):
        """Get topic compositions"""
        raise NotImplementedError()
