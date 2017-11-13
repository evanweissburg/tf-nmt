import csv

number_to_ss_letter = {'0': ' ', '1': 'H', '2': 'B', '3': 'E', '4': 'G', '5': 'I', '6': 'T', '7': 'S'}
ss_letter_to_number = inv_map = {v: k for k, v in number_to_ss_letter.items()}


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
