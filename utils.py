import csv
import os

int_to_dssp_letter = {'0': ' ', '1': 'H', '2': 'B', '3': 'E', '4': 'G', '5': 'I', '6': 'T', '7': 'S'}
dssp_letter_to_int = inv_map = {v: k for k, v in int_to_dssp_letter.items()}


def fasta_to_integers(protein: str, shift=0):
    return list(map(lambda x: ord(x) - 65 + shift, protein))


def integer_to_fasta(integer: int, shift=0):
    return chr(integer + 65 - shift)


def dssp_to_integers(protein: str, shift=0):
    return list(map(lambda x: int(dssp_letter_to_int[x]) + shift, protein))


def integer_to_dssp(integer: int, shift=0):
    return int_to_dssp_letter[str(integer - shift)]


def download_raw_data():
    print('ss.txt not found, downloading raw dataset from internet...')

    import urllib.request
    urllib.request.urlretrieve('https://cdn.rcsb.org/etl/kabschSander/ss.txt.gz', filename='ss.txt.gz')

    import gzip
    with gzip.open('ss.txt.gz', 'rb') as inF:
        with open('ss.txt', 'wb') as outF:
            outF.write(inF.read())

    import os
    os.remove('ss.txt.gz')

    print('Download complete.')


def integerize_raw_data():
    print('Preprocessed primary and secondary csv files not found, generating...')

    if not os.path.isfile('ss.txt'):
        download_raw_data()

    file = open('ss.txt', 'r')
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
    for protein in sequences:
        prot_labels.append(protein[0][1:7])
        primary.append(protein[1])
        secondary.append(protein[3])

    with open('primary.csv', 'w+') as file:
        writer = csv.writer(file)
        for sequence in primary:
            writer.writerow(fasta_to_integers(sequence, shift=0))

    with open('secondary.csv', 'w+') as file:
        writer = csv.writer(file)
        for sequence in secondary:
            writer.writerow(dssp_to_integers(sequence, shift=0))

    print('Data preparation complete! %s proteins prepared.' % len(primary))


def read_integerized_input():
    print('Beginning read from primary/secondary csv files...')

    if not (os.path.isfile('primary.csv') and os.path.isfile('secondary.csv')):
        integerize_raw_data()

    primary = []
    secondary = []
    with open('primary.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            primary.append(row)
    with open('secondary.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            secondary.append(row)

    print('Primary/secondary data read from file successfully.')
    return primary, secondary


def get_batch(primary, secondary):
    encoder_inputs = None
    decoder_inputs = None
    decoder_outputs = None
    return encoder_inputs, decoder_inputs, decoder_outputs
