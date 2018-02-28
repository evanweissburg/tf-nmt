import numpy as np
import random
from collections import Counter


def q8_infer_accuracy(preds, targets):
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


def q3_infer_accuracy(preds, targets):
    helix = [5, 8, 9]
    strand = [6, 7]
    loops = [2, 3, 4]

    def replace_q3(seq):
        for i, char in enumerate(seq):
            if char in helix:
                seq[i] = 2
            elif char in strand:
                seq[i] = 3
            elif char in loops:
                seq[i] = 4

    correct = 0
    total = 0
    for i in range(len(preds)):
        pred = preds[i]
        target = targets[i]

        replace_q3(pred)
        replace_q3(target)

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


def print_confusion(preds, tgts):
    confusion = [[0 for x in range(10)] for y in range(10)]     # initialize confusion array
    for i in range(len(preds)):                                 # index over proteins
        for j in range(len(preds[i])):                          # index over ss elements
            confusion[tgts[i][j]][preds[i][j]] += 1             # increment relevant matrix element

    for i in range(len(confusion)):                             # index over all target ss elements
        total = sum(confusion[i])                               # sum all occurrences of this target
        for j in range(len(confusion[i])):                      # index over predictions
            confusion[i][j] = confusion[i][j] / total           # calculate percentages by dividing by the sum


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
