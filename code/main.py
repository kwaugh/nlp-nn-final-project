import argparse
import random
import numpy as np
import time
import torch
import functools
import pickle
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch import optim
from models import *
from data import *
from utils import *

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
            '--domain',
            type=str,
            default='geo',
            help='domain (geo for geoquery)')
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
    # 65 is all you need for GeoQuery
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
    args = parser.parse_args()
    return args


class Seq2SeqSemanticParser(object):
    def __init__(self, encoder, decoder, input_emb, output_emb, max_output_len,
            input_indexer, output_indexer, device):
        self.encoder = encoder
        self.decoder = decoder
        self.input_emb = input_emb
        self.output_emb = output_emb
        self.max_output_len = max_output_len
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.SOS_INX = output_indexer.get_index(SOS_SYMBOL)
        self.EOS_INX = output_indexer.get_index(EOS_SYMBOL)
        self.device = device

    def decode(self, test_data):
        self.encoder.eval()
        self.decoder.eval()
        self.input_emb.eval()
        self.output_emb.eval()

        device = self.device
        test_derivs = []
        input_max_len = np.max(
                np.asarray([len(ex.x_indexed) for ex in test_data]))
        inputs = torch.from_numpy(make_padded_input_tensor(
            test_data, self.input_indexer, input_max_len, args.reverse_input))\
                    .to(device)
        for i in range(len(inputs)):
            test_ex = inputs[i].unsqueeze(0)
            # ENCODER
            zeros = np.where(test_ex.squeeze() == 0)[0]
            inp_len = len(test_ex) if len(zeros) == 0 else zeros[0]
            e_output, e_context, e_final_state = encode_input_for_decoder(
                    test_ex,
                    torch.as_tensor(inp_len, dtype=torch.long).unsqueeze(0),
                    self.input_emb,
                    self.encoder)
            # DECODER
            d_input = self.output_emb.forward(
                    torch.as_tensor([self.SOS_INX]).to(device).unsqueeze(0))
            d_hidden = e_final_state

            tokens = []
            for j in range(65):
                d_output, d_hidden = self.decoder.forward(
                        d_input, d_hidden, e_output)
                d_output = d_output.squeeze().argmax(dim=0)
                if d_output == self.EOS_INX:
                    break
                d_input = self.output_emb.forward(
                        d_output).unsqueeze(0).unsqueeze(0)
                tokens.append(self.output_indexer.get_object(d_output.item()))

            test_derivs.append([Derivation(test_data[i], 1.0, tokens)])

        return test_derivs


# Takes the given Examples and their input indexer and turns them into a numpy
# array by padding them out to max_len.  Optionally reverses them.
def make_padded_input_tensor(exs, input_indexer, max_len, reverse_input):
    #print("exs: {}".format(exs))
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) \
                    else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) \
                else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

# Analogous to make_padded_input_tensor, but without the option to reverse input
def make_padded_output_tensor(exs, output_indexer, max_len):
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) \
            else output_indexer.index_of(PAD_SYMBOL) \
            for i in range(0, max_len)] for ex in exs])


# Runs the encoder (input embedding layer and encoder as two separate modules)
# on a tensor of inputs x_tensor with inp_lens_tensor lengths.
# x_tensor: batch size x sent len tensor of input token indices
# inp_lens: batch size length vector containing the length of each sentence in
# the batch
# model_input_emb: EmbeddingLayer
# model_enc: RNNEncoder
# Returns the encoder outputs (per word), the encoder context mask (matrix of
# 1s and 0s reflecting

# E.g., calling this with x_tensor (0 is pad token):
# [[12, 25, 0, 0],
#  [1, 2, 3, 0],
#  [2, 0, 0, 0]]
# inp_lens = [2, 3, 1]
# will return outputs with the following shape:
# enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1,
# 1, 0], [1, 0, 0, 0]],
# enc_final_states = 3 x dim
def encode_input_for_decoder(
        x_tensor, inp_lens_tensor, model_input_emb, model_enc):
    # print("x_tensor.shape: {}".format(x_tensor.shape))
    # input_emb.shape: batch_size x sentence_len x embedding_size
    input_emb = model_input_emb.forward(x_tensor)
    (enc_output_each_word, enc_context_mask, enc_final_states) = \
            model_enc.forward(input_emb, inp_lens_tensor)
    enc_final_states_reshaped = (
            enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


def train_model_encdec(
        train_data, test_data, input_indexer, output_indexer, args):
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    output_max_len = np.max(
            np.asarray([len(ex.y_indexed) for ex in train_data]))

    print("Train length {}".format(input_max_len))
    print("Train output length: {}".format(
        np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))))
    '''
    print("Train matrix: {}; shape = {}".format(
        all_train_input_data, all_train_input_data.shape))
    print("Output matrix: {}; shape = {}".format(
        all_train_output_data, all_train_output_data.shape))
    '''
    # Create model
    epoch = 0
    if args.load_model:
        with open(args.load_model, 'rb') as f:
            print("Loaded model: ", args.load_model)
            state = pickle.load(f)

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
                    device)
            evaluate(test_data, parser)
    else:
        model_input_emb = EmbeddingLayer(
                args.input_dim, len(input_indexer), args.emb_dropout).to(device)
        model_enc = RNNEncoder(
                args.input_dim,
                args.hidden_size,
                args.rnn_dropout,
                args.bidirectional).to(device)
        model_output_emb = EmbeddingLayer(
                args.output_dim,
                len(output_indexer),
                args.emb_dropout).to(device)
        model_dec = RNNDecoder(
                args.output_dim,
                args.hidden_size,
                args.rnn_dropout,
                len(output_indexer),
                device,
                args.attention).to(device)

        enc_optimizer = optim.Adam(
                list(model_input_emb.parameters()) + list(model_enc.parameters()),
                lr=args.lr)
        dec_optimizer = optim.Adam(
                list(model_output_emb.parameters()) + list(model_dec.parameters()),
                lr=args.lr)

    model_input_emb.train()
    model_output_emb.train()
    model_enc.train()
    model_dec.train()
    # Loop over epochs, loop over examples, given some indexed words, call
    # encode_input_for_decoder, then call your decoder, accumulate losses,
    # update parameters

    # Setup
    loss_func = nn.NLLLoss()

    SOS_INX = output_indexer.get_index(SOS_SYMBOL)
    EOS_INX = output_indexer.get_index(EOS_SYMBOL)

    num_training_examples = len(train_data)
    # num_training_examples = 4

    while epoch < args.epochs:
        # TODO: input shuffling between epochs
        print("epoch: {}".format(epoch))
        total_epoch_loss = 0
        if epoch % args.save_epochs == 0 and epoch != 0:
            state = {
                "model_enc": model_enc,
                "model_dec": model_dec,
                "model_input_emb": model_input_emb,
                "model_output_emb": model_output_emb,
                "epoch": epoch,
                "enc_optimizer": enc_optimizer,
                "dec_optimizer": dec_optimizer
            }

            save_file = args.save_dir + str(epoch) + '.pkl'
            with open(save_file, 'wb') as f:
                pickle.dump(state, f)
            print("Saved model checkpoint to " + save_file)

        if epoch % args.evaluate_epochs == 0 and epoch != 0:
            parser = Seq2SeqSemanticParser(
                    model_enc,
                    model_dec,
                    model_input_emb,
                    model_output_emb,
                    output_max_len,
                    input_indexer,
                    output_indexer,
                    device)
            evaluate(test_data, parser)
            model_input_emb.train()
            model_output_emb.train()
            model_enc.train()
            model_dec.train()

        for i in range(0, num_training_examples, args.batch_size):
            loss = 0
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            batch_size = min(args.batch_size, num_training_examples-i-1)
            if batch_size == 0:
                break

            train_input_data = torch.from_numpy(make_padded_input_tensor(
                  train_data[i:i+batch_size], input_indexer, input_max_len, args.reverse_input)).to(
                  device)
            test_input_data = torch.from_numpy(make_padded_input_tensor(
                  test_data[i:i+batch_size], input_indexer, input_max_len, args.reverse_input)).to(
                  device)

            train_output_data = torch.from_numpy(make_padded_output_tensor(
                  train_data[i:i+batch_size], output_indexer, output_max_len)).to(device)
            test_output_data = torch.from_numpy(make_padded_output_tensor(
                  test_data[i:i+batch_size], output_indexer, output_max_len)).to(device)

            """
            print("train_input_data size: {}".format(train_input_data.size()))
            print("test_input_data size: {}".format(test_input_data.size()))
            print("train_output_data size: {}".format(train_output_data.size()))
            print("test_output_data size: {}".format(test_output_data.size()))
            """

            # ENCODER
            input_lens = torch.as_tensor(list(map(
                lambda x: len(x.x_indexed),
                train_data[i:i+batch_size])), dtype=torch.long).to(device)
            e_output, e_context, e_final_state = encode_input_for_decoder(
                    train_input_data,
                    input_lens,
                    model_input_emb,
                    model_enc)
            # DECODER
            d_input = model_output_emb.forward(torch.as_tensor(
                [SOS_INX] * batch_size).to(device).unsqueeze(0))
            d_hidden = e_final_state

            is_done = [False] * batch_size
            for j in range(len(train_output_data[0])):
                if functools.reduce(lambda x, y: x+int(not y), is_done, 0) == 0:
                    break
                d_output, d_hidden = model_dec.forward(
                        d_input, d_hidden, e_output)
                d_input = model_output_emb.forward(
                        train_output_data[:batch_size,j]).unsqueeze(0)
                d_output = d_output.view(-1, len(output_indexer))
                for k in range(batch_size):
                    if not is_done[k]:
                        loss = loss + loss_func(
                                d_output[k].unsqueeze(0),
                                train_output_data[k,j].unsqueeze(0))
                        is_done[k] = bool(
                                train_output_data[k,j] == EOS_INX)

            loss.backward()
            total_epoch_loss += loss
            enc_optimizer.step()
            dec_optimizer.step()
        #print("epoch loss: {}".format(int(total_epoch_loss)))
        print("average sample loss: {}".format(
            total_epoch_loss / num_training_examples))
        
        epoch += 1

    parser = Seq2SeqSemanticParser(
            model_enc,
            model_dec,
            model_input_emb,
            model_output_emb,
            output_max_len,
            input_indexer,
            output_indexer,
            device)
    evaluate(test_data, parser)
    return parser


# Evaluates decoder against the data in test_data (could be dev data or test
# data). Prints some output every example_freq examples. Writes predictions to
# outfile if defined. Evaluation requires executing the model's predictions
# against the knowledge base. We pick the highest-scoring derivation for each
# example with a valid denotation (if you've provided more than one).
def evaluate(
        test_data, decoder, example_freq=50, print_output=True, outfile=None):
    pred_derivations = decoder.decode(test_data)

    bleus = []
    for i, ex in enumerate(test_data):
        if i % example_freq == 0:
            print('Example %d' % i)
            print('  x      = "%s"' % ex.x)
            print('  y_tok  = "%s"' % ex.y_tok)
            print('  y_pred = "%s"' % pred_derivations[i][0].y_toks)

        hypotheses = [pred_derivations[i][0].y_toks]
        reference = pred_derivations[i][0].example.y_tok
        smoothing_func = SmoothingFunction()
        bleu = sentence_bleu(
                hypotheses,
                reference,
                smoothing_function=smoothing_func.method1,
                auto_reweigh=True)
        print("hypothesis: {}".format(hypotheses[0]))
        print("reference: {}".format(reference))
        print("bleu: {}".format(bleu))

        bleus.append(bleu)

    if print_output:
        print("average bleu: {}".format(sum(bleus)/len(bleus)))
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data

    train, dev, test = load_datasets(
            args.train_path_input, args.train_path_output,
            args.dev_path_input, args.dev_path_output,
            args.test_path_input, args.test_path_output,
            domain=args.domain)
    train_data_indexed, \
            dev_data_indexed, \
            test_data_indexed, \
            input_indexer, \
            output_indexer = index_datasets(
                    train, dev, test, args.decoder_len_limit)
    print("{} train exs, {} dev exs, {} input types, {} output types".format(
        len(train_data_indexed),
        len(dev_data_indexed),
        len(input_indexer),
        len(output_indexer)))
    print("{} train exs, {} dev exs, {} input types, {} output types".format(
        len(train_data_indexed),
        len(dev_data_indexed),
        len(input_indexer), len(output_indexer)))
    print("Input indexer: {}".format(input_indexer))
    print("Input indexer: {}".format(input_indexer))
    print("Output indexer: {}".format(output_indexer))
    print("Here are some examples post tokenization and indexing:")
    for i in range(0, min(len(train_data_indexed), 10)):
        print(train_data_indexed[i])
    decoder = train_model_encdec(
            train_data_indexed,
            dev_data_indexed,
            input_indexer,
            output_indexer,
            args)
    print("=======FINAL EVALUATION ON BLIND TEST=======")
    evaluate(
            test_data_indexed,
            decoder,
            print_output=False,
            outfile="geo_test_output.tsv")
