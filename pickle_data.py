"""Build pickle for corpus"""

import argparse
import datetime
import os
import pickle
import re
import time

import ankura.pipeline
from ankura import tokenize

from activetm import labeled
from activetm import utils


def get_dataset(settings):
    """Get dataset"""
    if settings['corpus'].find('*') >= 0:
        sentenceend = re.compile(r'\.([A-Z])')
        frusdelimiters = re.compile(r'\s+|\(\W*|\)\W*')
        def frussplitter(text):
            """Split according to Frus"""
            sentencified = sentenceend.sub(r' \g<1>', text)
            return frusdelimiters.split(sentencified)
        def frustokenizer(text):
            """Tokenize according to Frus"""
            return tokenize.simple(text, splitter=frussplitter)
        dataset = ankura.pipeline.read_glob(settings['corpus'],
                                            tokenizer=frustokenizer)
    else:
        dataset = ankura.pipeline.read_file(settings['corpus'])
    dataset = ankura.pipeline.filter_stopwords(dataset, settings['stopwords'])
    dataset = ankura.pipeline.filter_rarewords(dataset, int(settings['rare']))
    dataset = ankura.pipeline.filter_commonwords(dataset,
                                                 int(settings['common']))
    dataset = ankura.pipeline.filter_smalldocs(dataset,
                                               int(settings['smalldoc']))
    if settings['pregenerate'] == 'YES':
        dataset = ankura.pipeline.pregenerate_doc_tokens(dataset)
    return dataset


def _run():
    parser = argparse.ArgumentParser(description='Pickler of ActiveTM datasets')
    parser.add_argument('settings', help=\
            '''the path to a file containing settings, as described in \
            README.md in the root ActiveTM directory''')
    parser.add_argument('outputdir', help='directory for output')
    args = parser.parse_args()

    start = time.time()
    settings = utils.parse_settings(args.settings)
    pickle_name = utils.get_pickle_name(args.settings)
    if not os.path.exists(os.path.join(args.outputdir, pickle_name)):
        pre_dataset = get_dataset(settings)
        labels = labeled.get_labels(settings['labels'])
        dataset = labeled.LabeledDataset(pre_dataset, labels)
        with open(os.path.join(args.outputdir, pickle_name), 'wb') as ofh:
            pickle.dump(dataset, ofh)
    end = time.time()
    import_time = datetime.timedelta(seconds=end-start)
    with open(os.path.join(args.outputdir, pickle_name+'_import.time'), 'w') as ofh:
        ofh.write('# import time: {:s}\n'.format(str(import_time)))


if __name__ == '__main__':
    _run()

