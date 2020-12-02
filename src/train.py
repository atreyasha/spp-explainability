#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import monotonic
from collections import OrderedDict
from torch import LongTensor
from torch.nn import NLLLoss
from torch.nn.functional import log_softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from .utils.parser_utils import argparse_formatter
from .utils.data_utils import (vocab_from_text, read_labels, read_docs,
                               read_embeddings)
from .utils.model_utils import (shuffled_chunked_sorted, chunked_sorted,
                                to_cuda, Batch, ProbSemiring,
                                enable_gradient_clipping,
                                LogSpaceMaxTimesSemiring, MaxPlusSemiring,
                                fixed_var)
from .soft_patterns_pp import SoftPatternClassifier
from .arg_parser import soft_patterns_pp_arg_parser, training_arg_parser
import numpy as np
import argparse
import torch
import os


def train_batch(model,
                batch,
                num_classes,
                gold_output,
                optimizer,
                loss_function,
                gpu=False,
                debug=0,
                dropout=None):
    """Train on one doc. """
    optimizer.zero_grad()
    time0 = monotonic()
    loss = compute_loss(model, batch, num_classes, gold_output, loss_function,
                        gpu, debug, dropout)
    time1 = monotonic()
    loss.backward()
    time2 = monotonic()

    optimizer.step()
    if debug:
        time3 = monotonic()
        print(
            "Time in loss: {}, time in backward: {}, time in step: {}".format(
                round(time1 - time0, 3), round(time2 - time1, 3),
                round(time3 - time2, 3)))
    return loss.data


def compute_loss(model,
                 batch,
                 num_classes,
                 gold_output,
                 loss_function,
                 gpu,
                 debug=0,
                 dropout=None):
    time1 = monotonic()
    output = model.forward(batch, debug, dropout)

    if debug:
        time2 = monotonic()
        print("Forward total in loss: {}".format(round(time2 - time1, 3)))

    return loss_function(
        log_softmax(output, dim=1).view(batch.size(), num_classes),
        to_cuda(gpu)(fixed_var(LongTensor(gold_output))))


def evaluate_accuracy(model, data, batch_size, gpu, debug=0):
    n = float(len(data))
    correct = 0
    num_1s = 0
    for batch in chunked_sorted(data, batch_size):
        batch_obj = Batch([x for x, y in batch], model.embeddings,
                          to_cuda(gpu))
        gold = [y for x, y in batch]
        predicted = model.predict(batch_obj, debug)
        num_1s += predicted.count(1)
        correct += sum(1 for pred, gold in zip(predicted, gold)
                       if pred == gold)

    print("num predicted 1s:", num_1s)
    print("num gold 1s:     ", sum(gold == 1 for _, gold in data))

    return correct / n


def train(train_data,
          dev_data,
          model,
          num_classes,
          model_save_dir,
          num_iterations,
          model_file_prefix,
          learning_rate,
          batch_size,
          run_scheduler=False,
          gpu=False,
          clip=None,
          max_len=-1,
          debug=0,
          dropout=0,
          word_dropout=0,
          patience=1000):
    """ Train a model on all the given docs """

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_function = NLLLoss(weight=None, reduction="sum")

    enable_gradient_clipping(model, clip)

    if dropout:
        dropout = torch.nn.Dropout(dropout)
    else:
        dropout = None

    debug_print = int(100 / batch_size) + 1

    writer = None

    if model_save_dir is not None:
        writer = SummaryWriter(os.path.join(model_save_dir, "logs"))

    if run_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', 0.1, 10, True)

    best_dev_loss = 100000000
    best_dev_loss_index = -1
    best_dev_acc = -1
    start_time = monotonic()

    for it in range(num_iterations):
        np.random.shuffle(train_data)

        loss = 0.0
        i = 0
        for batch in shuffled_chunked_sorted(train_data, batch_size):
            batch_obj = Batch([x[0] for x in batch], model.embeddings,
                              to_cuda(gpu), word_dropout, max_len)
            gold = [x[1] for x in batch]
            loss += torch.sum(
                train_batch(model, batch_obj, num_classes, gold, optimizer,
                            loss_function, gpu, debug, dropout))

            if i % debug_print == (debug_print - 1):
                print(".", end="", flush=True)
            i += 1

        if writer is not None:
            for name, param in model.named_parameters():
                writer.add_scalar("parameter_mean/" + name, param.data.mean(),
                                  it)
                writer.add_scalar("parameter_std/" + name, param.data.std(),
                                  it)
                if param.grad is not None:
                    writer.add_scalar("gradient_mean/" + name,
                                      param.grad.data.mean(), it)
                    writer.add_scalar("gradient_std/" + name,
                                      param.grad.data.std(), it)

            writer.add_scalar("loss/loss_train", loss, it)

        dev_loss = 0.0
        i = 0
        for batch in chunked_sorted(dev_data, batch_size):
            batch_obj = Batch([x[0] for x in batch], model.embeddings,
                              to_cuda(gpu))
            gold = [x[1] for x in batch]
            dev_loss += torch.sum(
                compute_loss(model, batch_obj, num_classes, gold,
                             loss_function, gpu, debug).data)

            if i % debug_print == (debug_print - 1):
                print(".", end="", flush=True)

            i += 1

        if writer is not None:
            writer.add_scalar("loss/loss_dev", dev_loss, it)
        print("\n")

        finish_iter_time = monotonic()
        train_acc = evaluate_accuracy(model, train_data[:1000], batch_size,
                                      gpu)
        dev_acc = evaluate_accuracy(model, dev_data, batch_size, gpu)

        print(
            "iteration: {:>7,} train time: {:>9,.3f}m, eval time: {:>9,.3f}m "
            "train loss: {:>12,.3f} train_acc: {:>8,.3f}% "
            "dev loss: {:>12,.3f} dev_acc: {:>8,.3f}%".format(
                it, (finish_iter_time - start_time) / 60,
                (monotonic() - finish_iter_time) / 60, loss / len(train_data),
                train_acc * 100, dev_loss / len(dev_data), dev_acc * 100))

        if dev_loss < best_dev_loss:
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                print("New best acc!")
            print("New best dev!")
            best_dev_loss = dev_loss
            best_dev_loss_index = 0
            if model_save_dir is not None:
                model_save_file = os.path.join(
                    model_save_dir, "{}_{}.pth".format(model_file_prefix, it))
                print("saving model to", model_save_file)
                torch.save(model.state_dict(), model_save_file)
        else:
            best_dev_loss_index += 1
            if best_dev_loss_index == patience:
                print("Reached", patience,
                      "iterations without improving dev loss. Breaking")
                break

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            print("New best acc!")
            if model_save_dir is not None:
                model_save_file = os.path.join(
                    model_save_dir, "{}_{}.pth".format(model_file_prefix, it))
                print("saving model to", model_save_file)
                torch.save(model.state_dict(), model_save_file)

        if run_scheduler:
            scheduler.step(dev_loss)

    return model


def main(args):
    print(args)

    pattern_specs = OrderedDict(
        sorted(
            ([int(y) for y in x.split("-")] for x in args.patterns.split("_")),
            key=lambda t: t[0]))

    pre_computed_patterns = None

    if args.pre_computed_patterns is not None:
        pre_computed_patterns = read_patterns(args.pre_computed_patterns,
                                              pattern_specs)
        pattern_specs = OrderedDict(
            sorted(pattern_specs.items(), key=lambda t: t[0]))

    n = args.num_train_instances
    mlp_hidden_dim = args.mlp_hidden_dim
    num_mlp_layers = args.num_mlp_layers

    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    dev_vocab = vocab_from_text(args.vd)
    print("Dev vocab size:", len(dev_vocab))
    train_vocab = vocab_from_text(args.td)
    print("Train vocab size:", len(train_vocab))
    dev_vocab |= train_vocab

    vocab, embeddings, word_dim = \
        read_embeddings(args.embedding_file, dev_vocab)

    num_padding_tokens = max(list(pattern_specs.keys())) - 1

    dev_input, _ = read_docs(args.vd,
                             vocab,
                             num_padding_tokens=num_padding_tokens)
    dev_labels = read_labels(args.vl)
    dev_data = list(zip(dev_input, dev_labels))

    np.random.shuffle(dev_data)
    num_iterations = args.num_iterations

    train_input, _ = read_docs(args.td,
                               vocab,
                               num_padding_tokens=num_padding_tokens)
    train_labels = read_labels(args.tl)

    print("training instances:", len(train_input))

    num_classes = len(set(train_labels))

    # truncate data (to debug faster)
    train_data = list(zip(train_input, train_labels))
    np.random.shuffle(train_data)

    print("num_classes:", num_classes)

    if n is not None:
        train_data = train_data[:n]
        dev_data = dev_data[:n]

    # NOTE: temporary workaround to remove RNN for simplicity
    # if args.use_rnn:
    #     rnn = Rnn(word_dim, args.hidden_dim, cell_type=LSTM, gpu=args.gpu)
    # else:
    #     rnn = None
    rnn = None

    semiring = \
        MaxPlusSemiring if args.maxplus else (
            LogSpaceMaxTimesSemiring if args.maxtimes else ProbSemiring
        )

    model = SoftPatternClassifier(pattern_specs, mlp_hidden_dim,
                                  num_mlp_layers, num_classes, embeddings,
                                  vocab, semiring, args.bias_scale_param,
                                  args.gpu, rnn, pre_computed_patterns,
                                  args.no_sl, args.shared_sl, args.no_eps,
                                  args.eps_scale, args.self_loop_scale)

    if args.gpu:
        model.to_cuda(model)

    model_file_prefix = 'model'
    # Loading model
    if args.input_model is not None:
        state_dict = torch.load(args.input_model)
        model.load_state_dict(state_dict)
        model_file_prefix = 'model_retrained'

    model_save_dir = args.model_save_dir

    if model_save_dir is not None:
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

    print("Training with", model_file_prefix)

    train(train_data, dev_data, model, num_classes, model_save_dir,
          num_iterations, model_file_prefix, args.learning_rate,
          args.batch_size, args.scheduler, args.gpu, args.clip,
          args.max_doc_len, args.debug, args.dropout, args.word_dropout,
          args.patience)


def read_patterns(ifile, pattern_specs):
    with open(ifile, encoding='utf-8') as ifh:
        pre_computed_patterns = [
            l.rstrip().split() for l in ifh if len(l.rstrip())
        ]

    for p in pre_computed_patterns:
        l = len(p) + 1

        if l not in pattern_specs:
            pattern_specs[l] = 1
        else:
            pattern_specs[l] += 1

    return pre_computed_patterns


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse_formatter,
        parents=[training_arg_parser(),
                 soft_patterns_pp_arg_parser()])
    args = parser.parse_args()
    main(args)
