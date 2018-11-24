import string
import re
from unicodedata import normalize
from collections import Counter

# Acknowledgments:
# this code was adapted from
# https://machinelearningmastery.com/prepare-french-english-dataset-machine-translation/

valid_punctuation = '.?!'

""" 1. Cleaning """
# load doc into memory
def load_doc(filename):
    text = []
    with open(filename, mode='rt', encoding='utf-8') as f:
        text = f.read()
    return text

# True if char is in line and isn't at the end of the line
def char_in_middle_of_line(line, char):
    char_idx = line.find(char)
    return char_idx >= 0 and char_idx != len(line) - 1

# clean a list of lines
def clean_lines(lines):
    cleaned = list()
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # table = str.maketrans('', '', string.punctuation)
    for line in lines:
        line = normalize('NFD', line).encode('ascii', 'ignore')
        line = line.decode('UTF-8')
        line = line.split()
        line = [word.lower() for word in line]
        # line = [word.translate(table) for word in line]
        line = [re_print.sub('', w) for w in line]
        # line = [word for word in line if word.isalpha()]
        cleaned.append(' '.join(line))
    return cleaned

print("Starting cleaning")

doc = load_doc('europarl-v7.fr-en.en')
en_lines = doc.strip().split('\n')
en_lines = clean_lines(en_lines)

doc = load_doc('europarl-v7.fr-en.fr')
fr_lines = doc.strip().split('\n')
fr_lines = clean_lines(fr_lines)

""" 2. Vocab reduction """
# create a frequency table for all words
def to_vocab(lines):
    vocab = Counter()
    for line in lines:
        tokens = line.split()
        vocab.update(tokens)
    return vocab

# remove all words with a frequency below a threshold
def trim_vocab(vocab, min_occurance):
    tokens = [k for k,c in vocab.items() if c >= min_occurance]
    return set(tokens)

# mark all OOV with "unk" for all lines
def update_dataset(lines, vocab):
    new_lines = list()
    for line in lines:
        new_tokens = list()
        for token in line.split():
            if token in vocab:
                new_tokens.append(token)
            else:
                new_tokens.append('unk')
        new_line = ' '.join(new_tokens)
        new_lines.append(new_line)

    return new_lines

print("Starting reduction")

en_filename = 'clean-en.txt'
en_vocab = to_vocab(en_lines)
en_vocab = trim_vocab(en_vocab, 5)
en_lines = update_dataset(en_lines, en_vocab)

fr_filename = 'clean-fr.txt'
fr_vocab = to_vocab(fr_lines)
fr_vocab = trim_vocab(fr_vocab, 5)
fr_lines = update_dataset(fr_lines, fr_vocab)

"""
ctr = 0
for idx, item in enumerate(fr_lines):
    if not item and en_lines[idx]:
        print("idx: ", idx)
        print("en: ", en_lines[idx])
        print("fr: ", fr_lines[idx])
"""

with open(en_filename, 'w') as en_f:
    with open(fr_filename, 'w') as fr_f:
        for idx in range(min(len(fr_lines), len(en_lines))):
            en_line = en_lines[idx]
            fr_line = fr_lines[idx]

            if not en_line or not fr_line or len(en_line) < 3 or len(fr_line) < 3:
                continue
        
            # ignore all training pairs that have more than one period in them
            should_continue = False
            for char in valid_punctuation:
                if char_in_middle_of_line(en_line, char) or \
                    char_in_middle_of_line(fr_line, char):
                        should_continue = True
                        break
            if should_continue:
                continue
            
            # the line has gotta end in some type of punctuation
            if en_line[-1] not in valid_punctuation or \
                    fr_line[-1] not in valid_punctuation:
                        continue

            # the parser is dumb and doesn't handle sentences that end with a
            # single alphabet character
            if en_line[-2].isalpha() and en_line[-3] == ' ' \
                    or fr_line[-2].isalpha() and fr_line[-3] == ' ':
                continue

            en_f.write("%s\n" % en_line)
            fr_f.write("%s\n" % fr_line)
