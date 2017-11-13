import csv

int_to_dssp_letter = {'0': ' ', '1': 'H', '2': 'B', '3': 'E', '4': 'G', '5': 'I', '6': 'T', '7': 'S'}
dssp_letter_to_int = inv_map = {v: k for k, v in int_to_dssp_letter.items()}


def read_input(filename):
    file = open(filename, 'r')
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

    return prot_labels, primary, secondary


def fasta_to_integers(protein: str, shift):
    return list(map(lambda x: ord(x) - 65 + shift, protein))


def integers_to_fasta(protein: str, shift):
    return ''.join(map(lambda x: chr(x + 65 - shift), protein))


def dssp_to_integers(protein: str, shift):
    return list(map(lambda x: int(dssp_letter_to_int[x]) + shift, protein))


def integers_to_dssp(protein: str, shift):
    return ''.join(map(lambda x: int_to_dssp_letter[str(x - shift)], protein))


print('This should only be run once to translate FASTA/DSSP data to integers!')

print('Loading from source...')
prot_labels, primary, secondary = read_input('ss.txt')

print('Beginning FASTA to integer translation...')
with open('primary.csv', 'w+') as file:
    writer = csv.writer(file)
    for sequence in primary:
        writer.writerow(fasta_to_integers(sequence, shift=0))

print('Beginning DSSP to integer translation...')
with open('secondary.csv', 'w+') as file:
    writer = csv.writer(file)
    for sequence in secondary:
        writer.writerow(dssp_to_integers(sequence, shift=0))

print('Data preparation complete! %s proteins prepared.' % len(primary))
