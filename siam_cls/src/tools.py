import itertools
from collections import defaultdict


def get_index_combinations(n, every_n):
    indices = list(range(n))
    indices = indices[0::every_n]
    combinations = list(itertools.combinations(indices, 2))
    return combinations


def find_combs(pairs):
    mapp = defaultdict(list)
    for x, y in pairs:
        mapp[x].append(y)
    return [(x, *y) for x, y in mapp.items()]


def get_pairs(combinations):
    pairs_inds = []
    for combs in combinations:
        for ind in combs[1:]:
            pairs_inds.append([combs[0],ind])
    return pairs_inds 