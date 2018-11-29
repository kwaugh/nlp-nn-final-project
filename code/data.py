from utils import *
import random
import torch


# Wrapper class for an example.
# x = the natural language as one string
# x_tok = tokenized NL, a list of strings
# x_indexed = indexed tokens, a list of ints
# y = the logical form
# y_tok = tokenized logical form, a list of strings
# y_indexed = indexed logical form
class Example(object):
    def __init__(self, x, x_tok, x_indexed, y, y_tok, y_indexed):
        self.x = x
        self.x_tok = x_tok
        self.x_indexed = x_indexed
        self.y = y
        self.y_tok = y_tok
        self.y_indexed = y_indexed

    def __repr__(self):
        return " ".join(self.x_tok) + " => " + " ".join(self.y_tok) + \
                "\n   indexed as: " + repr(self.x_indexed) + " => " + \
                repr(self.y_indexed)

    def __str__(self):
        return self.__repr__()


# Wrapper for a Derivation consisting of an Example object, a score/probability
# associated with that example, and the tokenized prediction.
class Derivation(object):
    def __init__(self, example, p, y_toks):
        self.example = example
        self.p = p
        self.y_toks = y_toks

    def __str__(self):
        return "%s (%s)" % (self.y_toks, self.p)

    def __repr__(self):
        return self.__str__()


PAD_SYMBOL = "<PAD>"
UNK_SYMBOL = "<UNK>"
SOS_SYMBOL = "<SOS>"
EOS_SYMBOL = "<EOS>"


# Reads the training, dev, and test data from the corresponding files.
def load_datasets(
        train_path_input, train_path_output,
        dev_path_input, dev_path_output,
        test_path_input, test_path_output):
    train_raw = load_dataset(train_path_input, train_path_output)
    dev_raw = load_dataset(dev_path_input, dev_path_output)
    test_raw = load_dataset(test_path_input, test_path_output)
    return train_raw, dev_raw, test_raw


# Reads a dataset in from the given file
def load_dataset(input_filename, output_filename):
    dataset = []
    num_pos = 0
    with open(input_filename) as input:
        with open(output_filename) as output:
            input_lines = input.readlines()
            output_lines = output.readlines()
            for i in range(len(input_lines)):
                x = input_lines[i].strip()
                y = output_lines[i].strip()
                dataset.append((x, y))
    print("%i / %i pos exs" % (num_pos, len(dataset)))
    return dataset


# Whitespace tokenization
def tokenize(x):
    return x.split()


def index(x_tok, indexer):
    return [indexer.index_of(xi) if indexer.index_of(xi) >= 0 \
            else indexer.index_of(UNK_SYMBOL) for xi in x_tok]


def index_data(data, input_indexer, output_indexer, example_len_limit):
    data_indexed = []
    for (x, y) in data:
        x_tok = tokenize(x)
        y_tok = tokenize(y)[0:example_len_limit]
        data_indexed.append(Example(
            x,
            x_tok,
            index(x_tok, input_indexer),
            y,
            y_tok,
            index(y_tok, output_indexer) + \
                    [output_indexer.get_index(EOS_SYMBOL)]))
    return data_indexed


# Indexes train and test datasets where all words occurring less than or equal
# to unk_threshold times are replaced by UNK tokens.
def index_datasets(
        train_data, dev_data, test_data, example_len_limit, unk_threshold=0.0):
    input_word_counts = Counter()
    # Count words and build the indexers
    for (x, y) in train_data:
        for word in tokenize(x):
            input_word_counts.increment_count(word, 1.0)
    input_indexer = Indexer()
    output_indexer = Indexer()
    # Reserve 0 for the pad symbol for convenience
    input_indexer.get_index(PAD_SYMBOL)
    input_indexer.get_index(UNK_SYMBOL)
    output_indexer.get_index(PAD_SYMBOL)
    output_indexer.get_index(SOS_SYMBOL)
    output_indexer.get_index(EOS_SYMBOL)
    # Index all input words above the UNK threshold
    for word in input_word_counts.keys():
        if input_word_counts.get_count(word) > unk_threshold + 0.5:
            input_indexer.get_index(word)
    # Index all output tokens in train
    for (x, y) in train_data:
        for y_tok in tokenize(y):
            output_indexer.get_index(y_tok)
    # Index things
    train_data_indexed = index_data(
            train_data, input_indexer, output_indexer, example_len_limit)
    dev_data_indexed = index_data(
            dev_data, input_indexer, output_indexer, example_len_limit)
    test_data_indexed = index_data(
            test_data, input_indexer, output_indexer, example_len_limit)
    return (train_data_indexed,
            dev_data_indexed,
            test_data_indexed,
            input_indexer,
            output_indexer)
