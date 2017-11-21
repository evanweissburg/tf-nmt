import csv
import utils


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


print('This should only be run once to translate FASTA/DSSP data to integers!')

print('Loading from source...')
prot_labels, primary, secondary = read_input('ss.txt')

print('Beginning FASTA to integer translation...')
with open('primary.csv', 'w+') as file:
    writer = csv.writer(file)
    for sequence in primary:
        writer.writerow(utils.fasta_to_integers(sequence, shift=0))

print('Beginning DSSP to integer translation...')
with open('secondary.csv', 'w+') as file:
    writer = csv.writer(file)
    for sequence in secondary:
        writer.writerow(utils.dssp_to_integers(sequence, shift=0))

print('Data preparation complete! %s proteins prepared.' % len(primary))
