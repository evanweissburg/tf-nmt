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
    #urllib.request.urlretrieve('http://dunbrack.fccc.edu/Guoli/culledpdb_hh/cullpdb_pc30_res3.0_R1.0_d180719_chains17280.gz', filename='cull.txt.gz')

    with gzip.open('ss.txt.gz', 'rb') as inF:
        with open(data_dir+'ss.txt', 'wb+') as outF:
            outF.write(inF.read())

    #with gzip.open('cull.txt.gz', 'rb') as inF:
        #with open(data_dir+'cull.txt', 'wb+') as outF:
            #outF.write(inF.read())

    os.remove('ss.txt.gz')
    #os.remove('cull.txt.gz')


class SeqNotFound(Exception):
    pass


def make_primary_secondary(data_dir, max_size, max_len, max_weight, delta_weight, min_weight):
    input("pause")
    with open(os.path.join(data_dir, 'cull.txt')) as cull_file:
        with open(os.path.join(data_dir, 'ss.txt')) as file:
            sequences = []
            cull = cull_file.readlines()
            line = file.readline()
            del cull[0]
            for cur_cull in cull:
                try:
                    while not(line.find('sequence') is not -1 and line.find(cur_cull[:4]) is not -1):
                        line = file.readline()
                        if line.find('sequence') is not -1 and line[1:5] > cur_cull[:4]:
                            raise SeqNotFound()
                    sequences.append([])
                    sequences[-1].append(line[:-1])   # Get rid of line breaks
                    sequences[-1].append('')
                    l_index = 1
                    line = file.readline()
                    while line.find('sequence') == -1:
                        if line.find('secstr') is not -1:
                            sequences[-1].append(line[:-1])
                            sequences[-1].append('')
                            l_index = 3
                        else:
                            sequences[-1][l_index] += line[:-1]
                        line = file.readline()
                except SeqNotFound:
                    continue

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

    return len(primary), len(secondary)


def split_dataset(data_dir, test_split_rate, validate_split_rate):
    def do_split(reader, train_w, test_w, validate_w):
        num_train = 0
        num_test = 0
        num_validate = 0
        for i, seq in enumerate(reader):
            i %= 100
            if i < 100 - validate_split_rate - test_split_rate:
                train_w.writerow(seq)
                num_train += 1
            elif i < 100 - validate_split_rate:
                test_w.writerow(seq)
                num_test += 1
            else:
                validate_w.writerow(seq)
                num_validate += 1
        return num_train, num_test, num_validate

    with open(data_dir+'primary.csv', 'r+') as file:
        with open(data_dir+'train/primary_train.csv', 'w+', newline='') as train:
            with open(data_dir+'test/primary_test.csv', 'w+', newline='') as test:
                with open(data_dir+'validate/primary_validate.csv', 'w+', newline='') as validate:
                    do_split(csv.reader(file), csv.writer(train), csv.writer(test), csv.writer(validate))

    with open(data_dir+'secondary.csv', 'r+') as file:
        with open(data_dir+'train/secondary_train.csv', 'w+', newline='') as train:
            with open(data_dir+'test/secondary_test.csv', 'w+', newline='') as test:
                with open(data_dir+'validate/secondary_validate.csv', 'w+', newline='') as validate:
                    do_split(csv.reader(file), csv.writer(train), csv.writer(test), csv.writer(validate))

    with open(data_dir+'weights.csv', 'r+') as file:
        with open(data_dir+'train/weights_train.csv', 'w+', newline='') as train:
            with open(data_dir+'test/weights_test.csv', 'w+', newline='') as test:
                with open(data_dir+'validate/weights_validate.csv', 'w+', newline='') as validate:
                    return do_split(csv.reader(file), csv.writer(train), csv.writer(test), csv.writer(validate))


def fragment_datasets(data_dir, fragment_radius, fragment_jump):
    def fragment_file(dataset):
        with open(data_dir+dataset+'/primary_'+dataset+'.csv', 'r+') as primary:
            with open(data_dir+dataset+'/secondary_'+dataset+'.csv', 'r+') as secondary:
                with open(data_dir+dataset+'/weights_'+dataset+'.csv', 'r+') as weights:
                    with open(data_dir+dataset+'/primary_'+dataset+'_frag.csv', 'w+') as primary_frag:
                        with open(data_dir+dataset+'/secondary_'+dataset+'_frag.csv', 'w+') as secondary_frag:
                            with open(data_dir+dataset+'/weights_'+dataset+'_frag.csv', 'w+') as weights_frag:
                                with open(data_dir+dataset+'/'+dataset+'_frag.csv', 'w+') as frag_lookup:
                                    primary_r = csv.reader(primary)
                                    secondary_r = csv.reader(secondary)
                                    weights_r = csv.reader(weights)
                                    primary_frag_w = csv.writer(primary_frag)
                                    secondary_frag_w = csv.writer(secondary_frag)
                                    weights_frag_w = csv.writer(weights_frag)
                                    frag_lookup_w = csv.writer(frag_lookup)
                                    frag_count = 0
                                    for prim in primary_r:
                                        sec = next(secondary_r)
                                        wei = next(weights_r)
                                        num_frags = len(prim)//fragment_jump
                                        for j in range(num_frags):
                                            start = max(0, j-fragment_radius)
                                            end = min(j+fragment_radius+1, len(prim))
                                            primary_frag_w.writerow(prim[start:end])
                                            secondary_frag_w.writerow(sec[start:end])
                                            weights_frag_w.writerow(wei[start:end])
                                        frag_lookup_w.writerow([num_frags])
                                        frag_count += num_frags
        return frag_count

    num_train_frags = fragment_file('train')
    num_test_frags = fragment_file('test')
    num_validate_frags = fragment_file('validate')
    return num_train_frags, num_test_frags, num_validate_frags


def prep_nmt_dataset(hparams):
    print('Clearing previous data directory.')

    shutil.rmtree(hparams.data_dir, ignore_errors=True)
    os.mkdir(hparams.data_dir)
    os.mkdir(hparams.data_dir+'train/')
    os.mkdir(hparams.data_dir+'test/')
    os.mkdir(hparams.data_dir+'validate/')

    print('Downloading raw data text file.')

    download_raw_data(data_dir=hparams.data_dir)

    print('Generating base dataset (filtered by total size, length, and similarity).')

    num_prots, num_raw = make_primary_secondary(data_dir=hparams.data_dir, max_size=hparams.dataset_max_size,
                                                max_len=hparams.max_len,
                                                max_weight=hparams.max_weight, delta_weight=hparams.delta_weight,
                                                min_weight=hparams.min_weight)

    print('Using {} out of {} total proteins. Generating vocab files.'.format(num_prots, num_raw))

    primary_vocab, secondary_vocab = make_vocab_files(data_dir=hparams.data_dir, src_eos=hparams.src_eos, tgt_sos=hparams.tgt_sos, tgt_eos=hparams.tgt_eos)

    print('Found a primary vocabulary of {} and a secondary vocabulary of {}. Splitting base dataset into train/test/validate.'.format(primary_vocab, secondary_vocab))

    num_train, num_test, num_validate = split_dataset(data_dir=hparams.data_dir, test_split_rate=hparams.test_split_rate, validate_split_rate=hparams.validate_split_rate)

    print('Created train ({}), test ({}), and validate ({}) datasets. Fragmenting datasets.'.format(num_train, num_test, num_validate))

    num_train, num_test, num_validate = fragment_datasets(data_dir=hparams.data_dir, fragment_radius=hparams.fragment_radius, fragment_jump=hparams.fragment_jump)

    print('Fragmented train ({}), test ({}), and validate ({}) datasets.'.format(num_train, num_test, num_validate))

    print('Dataset generation complete.')
