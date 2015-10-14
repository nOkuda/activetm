from __future__ import division

def count_settings(filename):
    count = 0
    with open(filename) as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                count += 1
    return count

def parse_settings(filename):
    settings = {}
    with open(filename) as ifh:
        for line in ifh:
            line = line.strip()
            if line and not line.startswith('#'):
                k, v = line.split()
                settings[k] = v
    return settings

