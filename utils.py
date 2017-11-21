import csv

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


def read_integerized_input():
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

    return primary, secondary


