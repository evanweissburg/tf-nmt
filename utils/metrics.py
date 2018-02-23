import numpy as np
import random
from collections import Counter


def percent_infer_accuracy(preds, targets):
    correct = 0
    total = 0
    for i in range(len(preds)):
        pred = preds[i]
        target = targets[i]
        for j in range(len(pred)):
            if target[j] == 1:  # /s is the second (index 1) vocab word in secondary_vocab.txt
                break
            total += 1
            if pred[j] == target[j]:
                correct += 1
    return correct/total


def update_standard_deviation(old_total, old_squaresum, n, point):
    total = old_total + point
    squaresum = old_squaresum + (point ** 2)
    stdev = np.sqrt((squaresum / n) - ((total / n) ** 2))
    return total, squaresum, stdev


def print_common_mistake(preds, src, tgts=None):
    mistakes = []
    mistake_freq = []
    for i in range(len(preds)):
        for j in range(len(preds[i])):
            if tgts[i][j] != src[i][j]:
                if not src[i][j] in mistakes:
                    mistakes.append(src[i][j])
                    mistake_freq.append(1)
                else:
                    mistake_freq[mistakes.index(src[i][j])] += 1
    max = 0
    first = 0
    second = 0
    third = 0
    for k in range(len(mistakes)):
        if mistake_freq[k] > max:
            max = mistake_freq[k]
            third = second
            second = first
            first = mistakes[k]
    print('Most common sources of error: {}, {}, {}'.format(first, second, third))


def find_uniques(strings, max_len, sampling_len):
    random.seed(0)

    def bit_sampling(string, sample_indices):
            return ''.join([string[i] if i < len(string) else ' ' for i in sample_indices])

    indices = random.sample(range(max_len), sampling_len)
    hashes = [bit_sampling(string, indices) for string in strings]

    counter = Counter(hashes)
    uniques = list()
    for most_common, count in counter.most_common():
        group_indices = [i for i, x in enumerate(hashes) if x == most_common]
        uniques.append(group_indices[0])
    return uniques


def edit_distance(a, b):
    m = len(a)
    n = len(b)
    edit_dist = [[0 for x in range(n+1)] for x in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                edit_dist[i][j] = j
            elif j == 0:
                edit_dist[i][j] = i
            elif a[i-1] == b[j-1]:
                edit_dist[i][j] = edit_dist[i-1][j-1]
            else:
                edit_dist[i][j] = 1 + min(edit_dist[i][j-1],
                                          edit_dist[i-1][j],
                                          edit_dist[i-1][j-1])

    return edit_dist[m][n] / ((m+n)/2)
