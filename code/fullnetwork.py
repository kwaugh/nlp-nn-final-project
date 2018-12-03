import argparse
import random
import numpy as np
import time
import torch
import functools
import os
import subprocess
import pickle
import data
import models
import utils
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch import optim
from main import Seq2SeqSemanticParser

import sys
sys.path.insert(0, './data/')
from parse_to_sentence import parse_to_sentence

def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument(
            '--test_path_input',
            type=str,
            default='data/english_test.txt',
            help='path to blind test data input')
    parser.add_argument(
            '--test_path_output',
            type=str,
            default='data/french_test.txt',
            help='path to blind test data output')
    parser.add_argument(
            '--test_output_path',
            type=str,
            default='geo_test_output.tsv',
            help='path to write blind test results')
    parser.add_argument(
            '--load_model',
            type=str,
            default=None,
            help='seq2seq model to load from')
    # 65 is all you need for English->French
    # 206 is required for EnglishParse->FrenchParse
    parser.add_argument(
            '--decoder_len_limit',
            type=int,
            default=65,
            help='output length limit of the decoder')
    args = parser.parse_args()
    return args

# translates A -> A parse -> B parse -> B
class FullNetwork(object):
    def __init__(self, parse_seq2seq):
        self.parse_seq2seq = parse_seq2seq

        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.parse_seq2seq.device = device
        self.parse_seq2seq.input_emb = self.parse_seq2seq.input_emb.to(device)
        self.parse_seq2seq.output_emb = self.parse_seq2seq.output_emb.to(device)
        self.parse_seq2seq.encoder = self.parse_seq2seq.encoder.to(device)
        self.parse_seq2seq.decoder = self.parse_seq2seq.decoder.to(device)
        """

    def parse(self, A):
        cwd = os.getcwd()
        os.chdir("../stanford-corenlp")

        in_name = "in.txt"
        out_name = "out.txt"
        with open(in_name, 'w') as f:
            for a in A:
                f.write(a.x)
                f.write("\n")
            
        parser_command = ("java -cp .:stanford-parser.jar:stanford-postagger-3.5.0.jar:stanford-english-corenlp-2018-10-05-models.jar "
                "ShiftReduceDemo -model edu/stanford/nlp/models/srparser/englishSR.ser.gz -tagger english-bidirectional-distsim.tagger "
                "-file {} > {}").format(in_name, out_name)
        subprocess.call(parser_command, shell=True)

        parses = []
        with open(out_name, 'r') as f:
            for line in f.readlines():
                parses.append(line)

        os.chdir(cwd)

        # TODO: output doesn't matter (?)
        return data.index_data(zip(parses, parses), parse_seq2seq.input_indexer, parse_seq2seq.output_indexer, args.decoder_len_limit)

    def unparse(self, test_derivs):
        sentences = []
        for d in test_derivs:
            parse = " ".join(d[0].y_toks)
            sentence = parse_to_sentence(parse)
            sentences.append(sentence)

        return sentences

    def decode(self, A):                                  # [Example]
        # print("A: ", A[:5])
        A_parse = self.parse(A)                           # [Example]
        # print("A_parse: ", A_parse[:5])
        B_parse = self.parse_seq2seq.decode(A_parse)      # [Derivation]
        # print("B_parse: ", B_parse[:5])
        B = self.unparse(B_parse)                         # [string]

        # print("B: ", B[:5])

        derivs = []
        for a, b in zip(A, B):
            derivs.append([data.Derivation(a, 1.0, b.split())])
            
        return derivs

running_bleus = []

# Evaluates decoder against the data in test_data (could be dev data or test
# data). Prints some output every example_freq examples. Writes predictions to
# outfile if defined. Evaluation requires executing the model's predictions
# against the knowledge base. We pick the highest-scoring derivation for each
# example with a valid denotation (if you've provided more than one).
def evaluate(test_data, decoder, example_freq=50, print_output=True, outfile=None):
    pred_derivations = decoder.decode(test_data)

    bleus = []
    for i, ex in enumerate(test_data):
        hypotheses = [pred_derivations[i][0].y_toks]
        reference = pred_derivations[i][0].example.y_tok
        smoothing_func = SmoothingFunction()
        bleu = sentence_bleu(
                hypotheses,
                reference,
                smoothing_function=smoothing_func.method1,
                auto_reweigh=True)
        if i % example_freq == 0:
            print('Example %d' % i)
            print('  x      = "%s"' % ex.x)
            print('  y_tok  = "%s"' % ex.y_tok)
            print('  y_pred = "%s"' % pred_derivations[i][0].y_toks)

            print("hypothesis: {}".format(hypotheses[0]))
            print("reference: {}".format(reference))
            print("bleu: {}".format(bleu))

        bleus.append(bleu)

    if print_output:
        running_bleus.append(sum(bleus)/len(bleus))
        print("average bleus: {}".format(running_bleus))
    # Writes to the output file if needed
    """
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(
                    selected_derivs[i].y_toks) + "\n")
        out.close()
    """

def render_ratio(numer, denom):
    return "%i / %i = %.3f" % (numer, denom, float(numer)/denom)


if __name__ == '__main__':
    args = _parse_args()
    print(args)

    test_data = data.load_dataset(args.test_path_input, args.test_path_output)

    # index doesn't matter here (?)
    test_data_indexed = data.index_data(test_data, data.Indexer(), data.Indexer(), args.decoder_len_limit)

    with open(args.load_model, 'rb') as f:
        state = pickle.load(f)
        parse_seq2seq = state["model"]

        network = FullNetwork(parse_seq2seq)

        print("=======EVALUATION=======")
        evaluate(
                test_data_indexed,
                network,
                print_output=False,
                example_freq=1,
                outfile="geo_test_output.tsv")
