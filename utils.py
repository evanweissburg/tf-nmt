import csv
import os
import numpy as np
import shutil

int_to_dssp_letter = {'0': ' ', '1': 'H', '2': 'B', '3': 'E', '4': 'G', '5': 'I', '6': 'T', '7': 'S'}
dssp_letter_to_int = inv_map = {v: k for k, v in int_to_dssp_letter.items()}


def print_prots(preds, src, tgts=None, max_prints=None):
    count = min(max_prints, len(preds)) if max_prints else len(preds)
    for i in range(count):
        frmt = '{:>3}'*len(preds[i])
        print('>>> START PROTEIN <<<')
        print('Target     :' + frmt.format(*tgts[i]))
        print('Prediction :' + frmt.format(*preds[i]))
        print('Source     :' + frmt.format(*np.insert(src[i], list(src[i]).index(0), [-1]) if src[i][-1] == 0 else np.append(src[i], [-1])))


def fasta_to_integers(protein: str, shift=0):
    return list(map(lambda x: ord(x) - 65 + shift, protein))


def integer_to_fasta(integer: int, shift=0):
    return chr(integer + 65 - shift)


def dssp_to_integers(protein: str, shift=0):
    return list(map(lambda x: int(dssp_letter_to_int[x]) + shift, protein))


def integer_to_dssp(integer: int, shift=0):
    return int_to_dssp_letter[str(integer - shift)]


def download_raw_data(data_dir):
    print('Downloading raw dataset from internet...')

    import urllib.request
    urllib.request.urlretrieve('https://cdn.rcsb.org/etl/kabschSander/ss.txt.gz', filename='ss.txt.gz')

    import gzip
    with gzip.open('ss.txt.gz', 'rb') as inF:
        with open(data_dir+'ss.txt', 'wb+') as outF:
            outF.write(inF.read())

    import os
    os.remove('ss.txt.gz')

    print('Download complete.')


def clear_previous_runs(model_dir, data_dir):
    shutil.rmtree(model_dir, ignore_errors=True)
    shutil.rmtree(data_dir, ignore_errors=True)
    os.mkdir(model_dir)
    os.mkdir(data_dir)


def make_dataset(max_len, max_size, data_dir, max_weight, delta_weight, min_weight):
    if not os.path.isfile(data_dir+'ss.txt'):
        download_raw_data(data_dir)

    print('Generating dataset...')

    file = open(data_dir+'ss.txt', 'r')
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
            sequences[-1][l_index] = sequences[-1][l_index] + (line[:-1])

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
            writer.writerow(fasta_to_integers(sequence, shift=1))

    with open(data_dir+'secondary.csv', 'w+', newline='') as file:
        writer = csv.writer(file)
        for sequence in secondary:
            writer.writerow(dssp_to_integers(sequence, shift=3))

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

    print('Data preparation complete! %s proteins prepared.' % len(primary))


def get_data_stats():
    with open('primary.csv', 'r+') as file:
        reader = csv.reader(file, delimiter=',')
        maxi = 0
        for row in reader:
            maxi = max(maxi, len(row))
            print(len(row))
    return maxi
