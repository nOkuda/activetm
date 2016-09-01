"""Various utility functions"""
import os

import ankura.util


def count_settings(filename):
    """Counts the number of non-empty lines in the file"""
    count = 0
    with open(filename) as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                count += 1
    return count


@ankura.util.memoize
def parse_settings(filename):
    """Parse settings file

    Once parsed, the settings file is memoized
    """
    settings = {}
    with open(filename) as ifh:
        for line in ifh:
            line = line.strip()
            if line and not line.startswith('#'):
                k, val = line.split()
                settings[k] = val
    return settings


def get_pickle_name(filename):
    """Get pickle name from settings file"""
    settings = parse_settings(filename)
    return settings['pickle']


def get_mae_out_name(outputdir, label, count):
    """Get name for mean absolute error output"""
    directory = os.path.join(outputdir, label+'_mae')
    try:
        os.makedirs(directory)
    except OSError:
        pass
    return os.path.join(directory, str(count))

