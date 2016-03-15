import math

def js_divergence(u, v):
    m = (u + v) / 2.0
    return (kl_divergence(u, m) / 2.0) + (kl_divergence(v, m) / 2.0)

epsilon = 1e-10
def kl_divergence(u, v):
    result = 0.0
    for (p, q) in zip(u, v):
        if abs(p) < epsilon:
            # here, we define 0*log(0) = 0
            continue
        result += p * (math.log(p) - math.log(q))
    return result

def l1(u, v):
    result = 0.0
    for (p, q) in zip(u, v):
        result += math.abs(p - q)
    return result

def l2(u, v):
    result = 0.0
    for (p, q) in zip(u, v):
        diff = p - q
        result += diff * diff
    return math.sqrt(result)

