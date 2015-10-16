from __future__ import division
import argparse
import datetime
import numpy as np
import os
import random
import sys
import time

import ankura
from ankura import tokenize

import activetm.active.evaluate as evaluate
import activetm.active.select as select
import activetm.labeled
import activetm.models as models
import activetm.utils as utils

def get_dataset(settings):
    PIPELINE = []
    if settings['corpus'].find('*') >= 0:
        PIPELINE.append((ankura.read_glob, settings['corpus'], tokenize.simple))
    else:
        PIPELINE.append((ankura.read_file, settings['corpus'], tokenize.simple))
    PIPELINE.extend([
            (ankura.filter_stopwords, settings['stopwords']),
            (ankura.filter_rarewords, int(settings['rare'])),
            (ankura.filter_commonwords, int(settings['common'])),
            (ankura.filter_smalldocs, int(settings['smalldoc']))])
    if settings['pregenerate'] == 'YES':
        PIPELINE.append((ankura.pregenerate_doc_tokens))
    return ankura.run_pipeline(PIPELINE)

def partition_data_ids(num_docs, rng, settings):
    TEST_SIZE = int(settings['testsize'])
    START_LABELED = int(settings['startlabeled'])
    shuffled_doc_ids = range(num_docs)
    rng.shuffle(shuffled_doc_ids)
    test_doc_ids = list(shuffled_doc_ids[:TEST_SIZE])
    labeled_doc_ids = list(shuffled_doc_ids[TEST_SIZE:TEST_SIZE+START_LABELED])
    unlabeled_doc_ids = set(shuffled_doc_ids[TEST_SIZE+START_LABELED:])
    return test_doc_ids, labeled_doc_ids, unlabeled_doc_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Job runner for ActiveTM '
            'experiments')
    parser.add_argument('working_dir', help='ActiveTM directory '
            'available to hosts (should be a network path)')
    parser.add_argument('settings', help=\
            '''the path to a file containing settings, as described in \
            README.md in the root ActiveTM directory''')
    parser.add_argument('outputdir', help='directory for output')
    parser.add_argument('label', help='identifying label')
    args = parser.parse_args()

    settings = utils.parse_settings(args.settings)
    trueoutputdir = os.path.join(args.outputdir, settings['group'])
    if not os.path.exists(trueoutputdir):
        try:
            os.makedirs(trueoutputdir)
        except OSError:
            pass

    rng = random.Random(int(settings['seed']))
    model = models.build(rng, settings)

    start = time.time()
    pre_dataset = get_dataset(settings)
    labels = activetm.labeled.get_labels(settings['labels'])
    dataset = activetm.labeled.LabeledDataset(pre_dataset, labels)
    end = time.time()
    import_time = datetime.timedelta(seconds=end-start)

    start = time.time()
    test_doc_ids, labeled_doc_ids, unlabeled_doc_ids =\
            partition_data_ids(dataset.num_docs, rng, settings)
    test_labels = []
    test_words = []
    for t in test_doc_ids:
        test_labels.append(dataset.labels[dataset.titles[t]])
        test_words.append(dataset.doc_tokens(t))
    test_labels_mean = np.mean(test_labels)
    known_labels = []
    for t in labeled_doc_ids:
        known_labels.append(dataset.labels[dataset.titles[t]])

    SELECT_METHOD = select.factory[settings['select']]
    END_LABELED = int(settings['endlabeled'])
    LABEL_INCREMENT = int(settings['increment'])
    CAND_SIZE = int(settings['candsize'])
    results = []
    end = time.time()
    init_time = datetime.timedelta(seconds=end-start)

    start = time.time()
    model.train(dataset, labeled_doc_ids, known_labels)
    metric = evaluate.pR2(model, test_words, test_labels,
            test_labels_mean)
    results.append([len(labeled_doc_ids),
            datetime.timedelta(seconds=time.time()-start).total_seconds(), metric])
    while len(labeled_doc_ids) < END_LABELED and len(unlabeled_doc_ids) > 0:
        candidates = select.reservoir(list(unlabeled_doc_ids), rng,
                CAND_SIZE)
        chosen = SELECT_METHOD(dataset, labeled_doc_ids, candidates, model, rng,
                LABEL_INCREMENT)
        for c in chosen:
            known_labels.append(dataset.labels[dataset.titles[c]])
            labeled_doc_ids.append(c)
            unlabeled_doc_ids.remove(c)
        model.train(dataset, labeled_doc_ids, known_labels, True)
        metric = evaluate.pR2(model, test_words, test_labels,
                test_labels_mean)
        results.append([len(labeled_doc_ids),
                datetime.timedelta(seconds=time.time()-start).total_seconds(), metric])
    model.cleanup()

    output = []
    output.append('# import time: {:s}'.format(str(import_time)))
    output.append('# init time: {:s}'.format(str(init_time)))
    for result in results:
        output.append('\t'.join([str(r) for r in result]))
    output.append('')
    with open(os.path.join(trueoutputdir, args.label), 'w') as ofh:
        ofh.write('\n'.join(output))

