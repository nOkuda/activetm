"""Functions for computing various distances

All input is expected to be numpy arrays
"""
#pylint:disable=invalid-name
import math

import numpy as np
import scipy.stats


def js_divergence(u, v):
    """Get JS(u || v)"""
    m = 0.5 * (u + v)
    return 0.5 * (kl_divergence(u, m) + kl_divergence(v, m))


def kl_divergence(u, v):
    """Get KL(u || v)"""
    return scipy.stats.entropy(u, v)


def l1(u, v):
    """Get L1-distance between u and v"""
    return np.abs(u - v).sum()


def l2(u, v):
    """Get L2-distance between u and v"""
    diff = u - v
    return math.sqrt(np.dot(diff, diff))

