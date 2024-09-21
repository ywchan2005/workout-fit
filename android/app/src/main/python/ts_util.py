import numpy as np
from dtaidistance.subsequence.dtw import subsequence_alignment

def align(query, series, n, expand_factor=20):
    query = np.array(query).reshape(-1, 1)
    query = __smoothen(query).reshape(-1)
    series = __expand(series, expand_factor)
    ends = []
    for i in range(n):
        _, end = __align(query[:len(query) - n + i + 1], series)
        ends.append(end / expand_factor)
    return ends

def __align(query, series):
    alignment = subsequence_alignment(query, series)
    best_match = alignment.best_match()
    start, end = best_match.segment
    return [start, end]

def __expand(series, factor=20):
    return [*[x for p1, p2 in zip(series[:-1], series[1:]) for x in np.linspace(p1, p2, factor + 1)[:-1]], series[-1]]

def __smoothen(xs):
    xs = np.array(xs)
    upper, lower = __lb_envelope(xs, radius=1)
    smooth_xs = (upper * .8 + lower * .2)[:, 0]
    return smooth_xs

def __lb_envelope(time_series, radius):
    sz, d = time_series.shape
    envelope_up = np.empty((sz, d))
    envelope_down = np.empty((sz, d))

    for i in range(sz):
        min_idx = i - radius
        max_idx = i + radius + 1
        if min_idx < 0:
            min_idx = 0
        if max_idx > sz:
            max_idx = sz
        for di in range(d):
            envelope_down[i, di] = np.min(time_series[min_idx:max_idx, di])
            envelope_up[i, di] = np.max(time_series[min_idx:max_idx, di])

    return envelope_down, envelope_up

if 'main' in __name__:
    print(align([3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3))