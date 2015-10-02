from __future__ import division
import heapq
import numpy as np

def top_topic(dataset, doc_ids, model, rng):
    pqs = []
    for i in range(len(doc_ids)):
        pqs.append([])
    for i in range(model.numtrainchains):
        for j in range(model.numsamplespertrainchain):
            expectedTopicCounts = model.get_expected_topic_counts(dataset,
                    doc_ids, i, j)
            for d, expected in enumerate(expectedTopicCounts):
                highest = 0.0
                highestTopic = -1
                for (k, val) in enumerate(expected):
                    if val > highest:
                        highest = val
                        highestTopic = k
                if highestTopic == -1:
                    highestTopic = rng.randint(0, model.numtopics-1)
                    highest = rng.random()
                # we want the highest value out first, but heapq pops smallest first
                heapq.heappush(pqs[d], (-highest, highestTopic, i, j))
    result = np.zeros((len(doc_ids), model.wordindex.size()))
    for i, pq in enumerate(pqs):
        (_, highestTopic, i, j) = heapq.heappop(pq)
        result[i, :] = model.get_topic_distribution(highestTopic, i, j)
    return result
