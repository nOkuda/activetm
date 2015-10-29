from __future__ import division
import ankura.util

def count_settings(filename):
    count = 0
    with open(filename) as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                count += 1
    return count

@ankura.util.memoize
def parse_settings(filename):
    settings = {}
    with open(filename) as ifh:
        for line in ifh:
            line = line.strip()
            if line and not line.startswith('#'):
                k, v = line.split()
                settings[k] = v
    return settings

def get_pickle_name(filename):
    settings = parse_settings(filename)
    return settings['pickle']

