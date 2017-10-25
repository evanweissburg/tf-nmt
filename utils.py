import numpy as np
import csv


def read_input(filename):
    with open(filename, 'r') as csvfile:
        sequences = list()
        scan = csv.reader(csvfile, delimiter=' ')
        for observation in scan:
            sequences.append(list(map(lambda x: int(x), observation[0].split(','))))
    return sequences
