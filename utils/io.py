import random
import string

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
    user_in = input('Enter a protein (FASTA only - "r" for random): ')
    src = ''
    if user_in is not 'r':
        for ch in user_in:
            src = src + ch + ','
    else:
        rand = ''.join(random.choices(string.ascii_uppercase, k=30)).replace('J', '')
        for ch in rand:
            src = src + ch + ','
    return src[:-1]     # remove trailing comma




