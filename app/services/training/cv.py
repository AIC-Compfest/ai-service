from __future__ import annotations

import random
from collections import defaultdict


def make_group_folds(groups, n_splits: int = 5, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    unique = sorted(set(groups))
    if len(unique) < 8:
        n_splits = len(unique)
    n_splits = max(2, min(n_splits, len(unique)))
    shuffled = unique[:]
    random.Random(seed).shuffle(shuffled)
    assignments = {group: index % n_splits for index, group in enumerate(shuffled)}
    result = []
    for fold in range(n_splits):
        valid = [index for index, group in enumerate(groups) if assignments[group] == fold]
        train = [index for index, group in enumerate(groups) if assignments[group] != fold]
        result.append((train, valid))
    return result