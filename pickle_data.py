from __future__ import division

import argparse
import datetime
import os
import pickle
import time

import ankura.pipeline
from ankura import tokenize

import activetm.labeled
import activetm.utils as utils

def get_dataset(settings):
    PIPELINE = []
    if settings['corpus'].find('*') >= 0:
        PIPELINE.append((ankura.pipeline.read_glob, settings['corpus'], tokenize.simple))
    else:
        PIPELINE.append((ankura.pipeline.read_file, settings['corpus'], tokenize.simple))
    PIPELINE.extend([
            (ankura.pipeline.filter_stopwords, settings['stopwords']),
            (ankura.pipeline.filter_rarewords, int(settings['rare'])),
            (ankura.pipeline.filter_commonwords, int(settings['common'])),
            (ankura.pipeline.filter_smalldocs, int(settings['smalldoc']))])
    if settings['pregenerate'] == 'YES':
        PIPELINE.append((ankura.pipeline.pregenerate_doc_tokens))
    return ankura.pipeline.run_pipeline(PIPELINE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pickler of ActiveTM datasets')
    parser.add_argument('settings', help=\
            '''the path to a file containing settings, as described in \
            README.md in the root ActiveTM directory''')
    parser.add_argument('outputdir', help='directory for output')
    args = parser.parse_args()

    start = time.time()
    settings = utils.parse_settings(args.settings)
    pre_dataset = get_dataset(settings)
    labels = activetm.labeled.get_labels(settings['labels'])
    dataset = activetm.labeled.LabeledDataset(pre_dataset, labels)
    pickle_name = utils.get_pickle_name(args.settings)
    pickle.dump(dataset, open(os.path.join(args.outputdir,
            pickle_name), 'w'))
    end = time.time()
    import_time = datetime.timedelta(seconds=end-start)
    with open(os.path.join(args.outputdir, pickle_name+'_import.time'), 'w') as ofh:
        ofh.write('# import time: {:s}\n'.format(str(import_time)))

