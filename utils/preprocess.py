import csv
import os
import shutil
import urllib.request
import gzip
from utils import metrics


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


def make_primary_secondary(data_dir, max_size, max_len, sampling_len, max_weight, delta_weight, min_weight):
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
        if i >= max_size:
            break
        if len(protein[3]) > max_len:
            continue
        prot_labels.append(protein[0][1:7])
        primary.append(protein[1])
        secondary.append(protein[3])

    #unique_pri = metrics.find_uniques(primary, max_len, sampling_len)
    unique_sec = metrics.find_uniques(secondary, max_len, sampling_len)
    #tru_uniques = [i for i in unique_pri if i in unique_sec]
    primary = [primary[i] for i in unique_sec]
    secondary = [secondary[i] for i in unique_sec]

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
                    weights.append(round(max(weights[-1]-delta_weight, min_weight), ndigits=2))
            writer.writerow(weights)

    return len(secondary), len(sequences)


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
    with open(data_dir+'primary.csv', 'r+') as file:
        with open(data_dir+'primary_train.csv', 'w+', newline='') as train:
            with open(data_dir+'primary_test.csv', 'w+', newline='') as test:
                reader = csv.reader(file)
                train_w = csv.writer(train)
                test_w = csv.writer(test)
                for i, seq in enumerate(reader):
                    train_w.writerow(seq) if i % test_split_rate != 0 else test_w.writerow(seq)

    with open(data_dir+'secondary.csv', 'r+') as file:
        with open(data_dir+'secondary_train.csv', 'w+', newline='') as train:
            with open(data_dir+'secondary_test.csv', 'w+', newline='') as test:
                reader = csv.reader(file)
                train_w = csv.writer(train)
                test_w = csv.writer(test)
                for i, seq in enumerate(reader):
                    train_w.writerow(seq) if i % test_split_rate != 0 else test_w.writerow(seq)

    with open(data_dir+'weights.csv', 'r+') as file:
        with open(data_dir+'weights_train.csv', 'w+', newline='') as train:
            with open(data_dir+'weights_test.csv', 'w+', newline='') as test:
                reader = csv.reader(file)
                train_w = csv.writer(train)
                test_w = csv.writer(test)
                for i, seq in enumerate(reader):
                    train_w.writerow(seq) if i % test_split_rate != 0 else test_w.writerow(seq)


def prep_nmt_dataset(hparams):
    print('Clearing previous data directory.')

    shutil.rmtree(hparams.data_dir, ignore_errors=True)
    os.mkdir(hparams.data_dir)

    print('Downloading raw data text file.')

    download_raw_data(data_dir=hparams.data_dir)

    print('Generating base dataset (filtered by total size, length, and similarity).')

    num_prots, num_raw = make_primary_secondary(data_dir=hparams.data_dir, max_size=hparams.dataset_max_size,
                                                max_len=hparams.max_len, sampling_len=hparams.sampling_len,
                                                max_weight=hparams.max_weight, delta_weight=hparams.delta_weight,
                                                min_weight=hparams.min_weight)

    print('Using {} out of {} total proteins. Generating vocab files.'.format(num_prots, num_raw))

    make_vocab_files(data_dir=hparams.data_dir, src_eos=hparams.src_eos,
                     tgt_sos=hparams.tgt_sos, tgt_eos=hparams.tgt_eos)

    print('Splitting base dataset into train/test.')

    split_dataset(data_dir=hparams.data_dir, test_split_rate=hparams.test_split_rate)
    num_train = int(num_prots * (hparams.test_split_rate-1) // hparams.test_split_rate + num_prots % hparams.test_split_rate)
    num_test = num_prots // hparams.test_split_rate

    print('{} train and {} test proteins allocated. Files created successfully.'.format(num_train, num_test))
