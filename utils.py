import csv
import os
import shutil
import urllib.request
import gzip
from difflib import SequenceMatcher
import numpy as np

int_to_dssp_letter = {'0': ' ', '1': 'H', '2': 'B', '3': 'E', '4': 'G', '5': 'I', '6': 'T', '7': 'S'}
dssp_letter_to_int = inv_map = {v: k for k, v in int_to_dssp_letter.items()}


def print_example(preds, src, tgts=None, max_prints=None):
    count = min(max_prints, len(preds)) if max_prints else len(preds)
    for i in range(count):
        frmt = '{:>3}'*len(preds[i])
        print('>>> START PROTEIN <<<')
        if tgts is not None:
            print('Target     :' + frmt.format(*tgts[i]))
        print('Prediction :' + frmt.format(*preds[i]))
        print('Source     :' + frmt.format(*src[i]))


def get_inference_input():
    user_in = input('Enter a protein (FASTA only): ')
    src = ''
    for ch in user_in:
        src = src + ch + ','
    src = src[:-1]
    return src


def clear_previous_run(hparams):
    print('Clearing previous ckpt and log files.')
    shutil.rmtree(hparams.model_dir, ignore_errors=True)
    os.mkdir(hparams.model_dir)


def download_raw_data(data_dir):
    urllib.request.urlretrieve('https://cdn.rcsb.org/etl/kabschSander/ss.txt.gz', filename='ss.txt.gz')

    with gzip.open('ss.txt.gz', 'rb') as inF:
        with open(data_dir+'ss.txt', 'wb+') as outF:
            outF.write(inF.read())

    os.remove('ss.txt.gz')


def make_primary_secondary(data_dir, max_size, max_len, max_weight, delta_weight, min_weight):
    with open(os.path.join(data_dir, 'ss.txt')) as file:
        sequences = []
        l_index = 0
        for line in file:
            if line.find('sequence') is not -1:
                sequences.append([])
                sequences[-1].append(line[:-1])   # Get rid of line breaks
                sequences[-1].append('')
                l_index = 1
            elif line.find('secstr') is not -1:
                sequences[-1].append(line[:-1])
                sequences[-1].append('')
                l_index = 3
            else:
                sequences[-1][l_index] += line[:-1]

        prot_labels = []
        primary = []
        secondary = []
        for i, protein in enumerate(sequences):
            if max_size and i == max_size:
                break
            if max_len and len(protein[3]) > max_len:
                continue
            prot_labels.append(protein[0][1:7])
            primary.append(protein[1])
            secondary.append(protein[3])

    with open(data_dir+'primary.csv', 'w+', newline='') as file:
        writer = csv.writer(file)
        for sequence in primary:
            writer.writerow(sequence)

    with open(data_dir+'secondary.csv', 'w+', newline='') as file:
        writer = csv.writer(file)
        for sequence in secondary:
            writer.writerow(sequence)

    with open(data_dir+'weights.csv', 'w+', newline='') as file:
        writer = csv.writer(file)
        for sequence in secondary:
            weights = list()
            weights.append(1.0)
            for i in range(1, len(sequence)):
                if sequence[i] != sequence[i-1]:
                    weights.append(max_weight)
                else:
                    weights.append(max(weights[-1]-delta_weight, min_weight))
            writer.writerow(weights)


def make_vocab_files(data_dir, src_eos, tgt_sos, tgt_eos):
    primary = list()
    with open(data_dir+'primary.csv', 'r+') as file:
        reader = csv.reader(file)
        for row in reader:
            for char in row:
                if char not in primary:
                    primary.append(char)

    with open(data_dir+'primary_vocab.txt', 'w+', newline='') as file:
        file.write(src_eos + '\n')
        for char in primary:
            file.write(char + '\n')

    secondary = list()
    with open(data_dir+'secondary.csv', 'r+') as file:
        reader = csv.reader(file)
        for row in reader:
            for char in row:
                if char not in secondary:
                    secondary.append(char)

    with open(data_dir+'secondary_vocab.txt', 'w+', newline='') as file:
        file.write(tgt_sos + '\n')
        file.write(tgt_eos + '\n')
        for char in secondary:
            file.write(char + '\n')


def split_dataset(data_dir, test_split_rate):
    primary = list()
    with open(data_dir+'primary.csv', 'r+') as file:
        reader = csv.reader(file)
        for row in reader:
            primary.append(row)

    with open(data_dir+'primary_train.csv', 'w+', newline='') as train:
        with open(data_dir+'primary_test.csv', 'w+', newline='') as test:
            train_w = csv.writer(train)
            test_w = csv.writer(test)
            for i, seq in enumerate(primary):
                train_w.writerow(seq) if i % test_split_rate != 0 else test_w.writerow(seq)

    secondary = list()
    with open(data_dir+'secondary.csv', 'r+') as file:
        reader = csv.reader(file)
        for row in reader:
            secondary.append(row)

    with open(data_dir+'secondary_train.csv', 'w+', newline='') as train:
        with open(data_dir+'secondary_test.csv', 'w+', newline='') as test:
            train_w = csv.writer(train)
            test_w = csv.writer(test)
            for i, seq in enumerate(secondary):
                train_w.writerow(seq) if i % test_split_rate != 0 else test_w.writerow(seq)

    weights = list()
    with open(data_dir+'weights.csv', 'r+') as file:
        reader = csv.reader(file)
        for row in reader:
            weights.append(row)

    with open(data_dir+'weights_train.csv', 'w+', newline='') as train:
        with open(data_dir+'weights_test.csv', 'w+', newline='') as test:
            train_w = csv.writer(train)
            test_w = csv.writer(test)
            for i, seq in enumerate(weights):
                train_w.writerow(seq) if i % test_split_rate != 0 else test_w.writerow(seq)


def prep_nmt_dataset(hparams):
    print('Clearing previous data directory.')

    shutil.rmtree(hparams.data_dir, ignore_errors=True)
    os.mkdir(hparams.data_dir)

    print('Downloading raw data text file.')

    download_raw_data(data_dir=hparams.data_dir)

    print('Generating base dataset.')

    make_primary_secondary(data_dir=hparams.data_dir, max_size=hparams.dataset_max_size, max_len=hparams.max_len,
                           max_weight=hparams.max_weight, delta_weight=hparams.delta_weight,
                           min_weight=hparams.min_weight)

    print('Generating vocab files.')

    make_vocab_files(data_dir=hparams.data_dir, src_eos=hparams.src_eos,
                     tgt_sos=hparams.tgt_sos, tgt_eos=hparams.tgt_eos)

    print('Splitting base dataset into train/test.')

    split_dataset(data_dir=hparams.data_dir, test_split_rate=hparams.test_split_rate)

    print('Files created successfully.')


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


def lib_percent_infer_accuracy(preds, targets):
    avg_ratio = 0
    for i in range(len(preds)):
        end = len(targets[i])
        for j, num in enumerate(targets[i]):
            if num == 1:
                end = j
                break
        pred = preds[i][:end]
        target = targets[i][:end]
        ratio = SequenceMatcher(None, pred, target).ratio()
        avg_ratio += ratio
    return avg_ratio/len(preds)


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
