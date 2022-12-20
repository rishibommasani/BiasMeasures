from scipy import special, stats, spatial
import numpy as np
import itertools

#################################################################################################
# Basic Operations


def average(data):
    if len(data) == 0:
        return 0
    return sum(data) / len(data)


#################################################################################################

#################################################################################################
# Normalization Functions


def sum_normalization(scores):
    Z = sum(scores)
    if Z == 0.0:
        N = len(scores)
        return [1 / N for _ in scores]
    else:
        return [score / Z for score in scores]


softmax = special.softmax
#################################################################################################

#################################################################################################
# Distance Functions


def L1_distance(x, y):
    return np.linalg.norm((x - y), ord=1)


def binary_difference_distance(x, y):
    assert len(x) == 2
    return x[0] - x[1]


def L2_distance(x, y):
    return np.linalg.norm((x - y), ord=2)


def Linfinity_distance(x, y):
    return np.linalg.norm((x - y), ord=np.inf)


def KL_divergence(x, y):
    return sum(special.kl_div(x, y))


JS_divergence = spatial.distance.jensenshannon
#################################################################################################

#################################################################################################
# Similarity Functions


def cosine(x, y):
    if np.linalg.norm(x, ord=2) <= 10**(-8) or np.linalg.norm(y, ord=2) <= 10**(-8):
        return 0
    else:
        return 1 - spatial.distance.cosine(x, y)


def inner_product(x, y):
    return np.dot(x, y)


#################################################################################################

#################################################################################################
# Probability Distributions


def get_uniform_distribution(length):
    distribution = np.ones(length) / length
    total = sum(distribution)
    if (total != 1):
        distribution[-1] += 1 - total
    return distribution


#################################################################################################
