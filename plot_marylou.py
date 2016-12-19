"""Plots results for experiment on super computer"""

import argparse
import os

import main
from activetm import utils


def parse_arguments():
    """get commandline arguments"""
    parser = argparse.ArgumentParser(prog='Results Plotter',
                                     description='Plots results from '
                                         'supercomputer')
    parser.add_argument('resultspath', help='File path to results')
    parser.add_argument(
        'deltatxt',
        help='Text file listing corpus with desired loss delta for generalized'\
            + ' zero-one loss')
    return parser.parse_args()


def run():
    """plot results from given experiment"""
    args = parse_arguments()
    corpora = [
        'amazon',
        'frus',
        'sotu_broken',
        'yelp'
    ]
    selections = [
        'random',
        'top_topic',
        'topic_comp'
    ]
    for corpus in corpora:
        main.make_plots(
            os.path.join(args.resultspath, corpus),
            selections,
            utils.parse_settings(args.deltatxt))


if __name__ == '__main__':
    run()
