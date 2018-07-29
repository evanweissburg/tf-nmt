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
    helix = [3, 8]
    sheet = [5, 7]
    loops = [2, 4, 6, 9]

    def replace_q3(seq):
        for i, char in enumerate(seq):
            if char in helix:
                seq[i] = 2
            elif char in sheet:
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


def stitch(radius, frags):
    candidates = list()
    for _ in frags:
        candidates.append(list())
    for i, frag in enumerate(frags):
        if i < radius:
            for j in range(i + radius):
                candidates[j].append(frag[j])
        elif i >= len(frags) - radius:
            for j in range(radius + len(frags) - i):
                candidates[i + j - radius].append(frag[j])
        else:
            for j in range(radius * 2):
                candidates[i + j - radius].append(frag[j])

    stitched = list()
    for candidate in candidates:
        candidate = [1 if n == -1 else n for n in candidate]
        stitched.append(np.argmax(np.bincount(candidate)))
    return stitched


def do_stitching(src, tgts, ids, radius, test_frags, test_frag_num):
    new_src = list()
    new_tgts = list()
    new_ids = list()

    i = 0
    while i < len(ids):
        print('i is {}'.format(i))
        j = 0
        k = test_frag_num + i
        while k > 0:
            k -= test_frags[j]
            j += 1
        if k == 0 and len(ids) - i >= test_frags[j]: # Start of protein and all needed frags are present
            new_src.append(stitch(radius, src[i:i+test_frags[j]]))
            new_tgts.append(stitch(radius, tgts[i:i+test_frags[j]]))
            new_ids.append(stitch(radius, ids[i:i+test_frags[j]]))
            i += test_frags[j]
        else:
            i += 1
    test_frag_num += i
    return new_src, new_tgts, new_ids, test_frag_num