import argparse
import random
import numpy as np
import time
import torch
import functools
import pickle
import data
import models
import utils
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch import optim
from main import Seq2SeqSemanticParser

def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument(
            '--train_path_input',
            type=str,
            default='data/english_train.txt',
            help='path to train data input')
    parser.add_argument(
            '--train_path_output',
            type=str,
            default='data/french_train.txt',
            help='path to train data output')
    parser.add_argument(
            '--dev_path_input',
            type=str,
            default='data/english_dev.txt',
            help='path to dev data input')
    parser.add_argument(
            '--dev_path_output',
            type=str,
            default='data/french_dev.txt',
            help='path to dev data output')
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
            '--save_dir',
            type=str,
            default='model_save/',
            help='directory to save models')
    parser.add_argument(
            '--save_epochs',
            type=int,
            default=10,
            help='how often to save a model')
    parser.add_argument(
            '--load_model',
            type=str,
            default=None,
            help='model to load from')
    parser.add_argument(
            '--evaluate_epochs',
            type=int,
            default=10,
            help='how many epochs to evaluate the model performance')

    # Some common arguments for your convenience
    parser.add_argument(
            '--seed',
            type=int,
            default=0,
            help='RNG seed (default = 0)')
    parser.add_argument(
            '--epochs',
            type=int,
            default=100,
            help='num epochs to train for')
    parser.add_argument(
            '--lr',
            type=float,
            default=.001)
    parser.add_argument(
            '--batch_size',
            type=int,
            default=40,
            help='batch size')
    # 65 is all you need for English->French
    # 206 is required for EnglishParse->FrenchParse
    parser.add_argument(
            '--decoder_len_limit',
            type=int,
            default=65,
            help='output length limit of the decoder')
    parser.add_argument(
            '--input_dim',
            type=int,
            default=100,
            help='input vector dimensionality')
    parser.add_argument(
            '--output_dim',
            type=int,
            default=100,
            help='output vector dimensionality')
    parser.add_argument(
            '--hidden_size',
            type=int,
            default=200,
            help='hidden state dimensionality')

    # Hyperparameters for the encoder -- feel free to play around with these!
    parser.add_argument(
            '--no_bidirectional',
            dest='bidirectional',
            default=True,
            action='store_false',
            help='bidirectional LSTM')
    parser.add_argument(
            '--no_reverse_input',
            dest='reverse_input',
            default=True,
            action='store_false',
            help='disable_input_reversal')
    parser.add_argument(
            '--emb_dropout',
            type=float,
            default=0.2,
            help='input dropout rate')
    parser.add_argument(
            '--rnn_dropout',
            type=float,
            default=0.2,
            help='dropout rate internal to encoder RNN')
    parser.add_argument(
            '--no_attention',
            dest='attention',
            default=True,
            action='store_false',
            help='Use attention in the decoder')
    parser.add_argument(
            '--shuffle',
            dest='shuffle',
            default=False,
            action='store_true',
            help='shuffle the dataset between epochs')
    parser.add_argument(
            '--num_workers',
            type=int,
            default=2,
            help='number of data loading workers')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train, dev, test = data.load_datasets(
            args.train_path_input, args.train_path_output,
            args.dev_path_input, args.dev_path_output,
            args.test_path_input, args.test_path_output)
    train_data_indexed, \
            dev_data_indexed, \
            test_data_indexed, \
            input_indexer, \
            output_indexer = data.index_datasets(
                    train, dev, test, args.decoder_len_limit)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    output_max_len = np.max(
            np.asarray([len(ex.y_indexed) for ex in train_data_indexed]))

    temp = args.load_model.split(".")
    out_file = temp[0] + "_fixed." + temp[1]

    with open(args.load_model, 'rb') as in_f:
        with open(out_file, 'wb') as out_f:
            print("Loaded model: ", args.load_model)
            state = pickle.load(in_f)

            model_enc = state["model_enc"]
            model_dec = state["model_dec"]
            model_input_emb = state["model_input_emb"]
            model_output_emb = state["model_output_emb"]
            enc_optimizer = state["enc_optimizer"]
            dec_optimizer = state["dec_optimizer"]
            epoch = state["epoch"]

            parser = Seq2SeqSemanticParser(
                    model_enc,
                    model_dec,
                    model_input_emb,
                    model_output_emb,
                    output_max_len,
                    input_indexer,
                    output_indexer,
                    args.decoder_len_limit,
                    device)

            new_state = {
                "model": parser,
                "enc_optimizer": enc_optimizer,
                "dec_optimizer": dec_optimizer,
                "epoch": epoch,
            }
            print("Saved model to : ", out_file)
            pickle.dump(new_state, out_f)
