#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
from typing import List, Union, MutableMapping, Tuple, cast
from torch import LongTensor
from torch.nn import NLLLoss, Module
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
                                LogSpaceMaxTimesSemiring, MaxPlusSemiring)
from .soft_patterns_pp import SoftPatternClassifier
from .arg_parser import soft_patterns_pp_arg_parser, training_arg_parser
import numpy as np
import argparse
import torch
import os


def train_batch(model: Module,
                batch: Batch,
                num_classes: int,
                gold_output: List[int],
                optimizer: torch.optim.Optimizer,
                loss_function: torch.nn.modules.loss._Loss,
                gpu: bool = False,
                dropout: Union[torch.nn.Module, None] = None) -> torch.Tensor:
    # set optimizer gradients to zero
    optimizer.zero_grad()

    # compute model loss
    loss = compute_loss(model, batch, num_classes, gold_output, loss_function,
                        gpu, dropout)

    # compute loss gradients for all parameters
    loss.backward()

    # perform a single optimization step
    optimizer.step()

    # detach loss from computational graph and return tensor data
    return loss.detach()


def compute_loss(model: Module,
                 batch: Batch,
                 num_classes: int,
                 gold_output: List[int],
                 loss_function: torch.nn.modules.loss._Loss,
                 gpu: bool,
                 dropout: Union[torch.nn.Module, None] = None) -> torch.Tensor:
    # compute model outputs given batch
    output = model.forward(batch, dropout)

    # return loss over output and gold
    return loss_function(
        log_softmax(output, dim=1).view(batch.size(), num_classes),
        to_cuda(gpu)(LongTensor(gold_output)))


def evaluate_accuracy(model: Module, data: List[Tuple[List[int], int]],
                      batch_size: int, gpu: bool) -> float:
    # instantiate local variables
    number_data_points = float(len(data))
    correct = 0
    num_1s = 0

    # chunk data into sorted batches and iterate
    for batch in chunked_sorted(data, batch_size):
        # create batch and send to GPU if present
        batch_obj = Batch([x for x, y in batch], model.embeddings,
                          to_cuda(gpu))

        # parse gold output
        gold = [y for x, y in batch]

        # predict output using model
        predicted = model.predict(batch_obj)

        # find number of predicted class 1's
        # NOTE: legacy technique for binary classifier
        # TODO: replace or reject, as well as print statement below
        num_1s += predicted.count(1)

        # find number of correctly predicted data points
        correct += sum(1 for pred, gold in zip(predicted, gold)
                       if pred == gold)

    # print information on predicted 1's
    print("num predicted 1s:", num_1s)
    print("num gold 1s:     ", sum(gold == 1 for _, gold in data))

    # return raw accuracy float
    # TODO: replace this workflow with more robust metric such as F1 score
    return correct / number_data_points


def train(train_data: List[Tuple[List[int], int]],
          dev_data: List[Tuple[List[int], int]],
          model: Module,
          num_classes: int,
          models_directory: str,
          epochs: int,
          model_file_prefix: str,
          learning_rate: float,
          batch_size: int,
          run_scheduler: bool = False,
          gpu: bool = False,
          clip_threshold: Union[float, None] = None,
          max_doc_len: int = -1,
          dropout: Union[torch.nn.Module, float, None] = 0,
          word_dropout: float = 0,
          patience: int = 30) -> None:
    # instantiate Adam optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # instantiate negative log-likelihood loss
    # NOTE: reduce by summing over batch instead of default 'mean'
    loss_function = NLLLoss(weight=None, reduction="sum")

    # enable gradient clipping in-place if provided
    enable_gradient_clipping(model, clip_threshold)

    # initialize dropout if provided
    if dropout:
        dropout = torch.nn.Dropout(dropout)
    else:
        dropout = None

    # initialize paramter which adjusts "." creation for progress loop
    debug_print = int(100 / batch_size) + 1

    # initialize tensorboard writer if provided
    writer = None
    if models_directory is not None:
        writer = SummaryWriter(os.path.join(models_directory, "logs"))

    # initialize learning rate scheduler if provided
    if run_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'min', 0.1, 10, True)

    # initialize floats for re-use
    best_dev_loss = 100000000.
    best_dev_loss_index = -1
    best_dev_acc = -1.

    # loop over epochs
    for epoch in range(epochs):
        # shuffle training data
        np.random.shuffle(train_data)

        # initialize training loss and run count
        loss = 0.0
        i = 0

        # loop over shuffled train batches
        for batch in shuffled_chunked_sorted(train_data, batch_size):
            # create batch object
            batch_obj = Batch([x[0] for x in batch], model.embeddings,
                              to_cuda(gpu), word_dropout, max_doc_len)
            # parse out gold labels
            gold = [x[1] for x in batch]
            # find aggregate loss across samples in batch
            loss += torch.sum(
                train_batch(model, batch_obj, num_classes, gold, optimizer,
                            loss_function, gpu, dropout))
            # print dots for progress
            if i % debug_print == (debug_print - 1):
                print(".", end="", flush=True)
            # increment batch counter
            i += 1

        # add parameter data to tensorboard if provided
        if writer is not None:
            # add named parameter data
            for name, param in model.named_parameters():
                writer.add_scalar("parameter_mean/" + name, param.data.mean(),
                                  epoch)
                writer.add_scalar("parameter_std/" + name, param.data.std(),
                                  epoch)
                if param.grad is not None:
                    writer.add_scalar("gradient_mean/" + name,
                                      param.grad.data.mean(), epoch)
                    writer.add_scalar("gradient_std/" + name,
                                      param.grad.data.std(), epoch)
            # add loss data
            writer.add_scalar("loss/loss_train", loss, epoch)

        # initialize dev loss and run count
        dev_loss = 0.0
        i = 0

        # loop over static dev set
        for batch in chunked_sorted(dev_data, batch_size):
            # create batch object
            batch_obj = Batch([x[0] for x in batch], model.embeddings,
                              to_cuda(gpu))
            # parse out gold labels
            gold = [x[1] for x in batch]
            # find aggregate loss across dev samples in batch
            dev_loss += torch.sum(
                compute_loss(model, batch_obj, num_classes, gold,
                             loss_function, gpu).data)
            # print dots for progress
            if i % debug_print == (debug_print - 1):
                print(".", end="", flush=True)
            # increment batch counter
            i += 1

        # add dev loss data to tensorboard
        if writer is not None:
            writer.add_scalar("loss/loss_dev", dev_loss, epoch)

        # add newline for stdout progress loop
        print("\n")

        # evaluate training and dev accuracies
        # TODO: do not limit training data accuracy here
        train_acc = evaluate_accuracy(model, train_data[:1000], batch_size,
                                      gpu)
        dev_acc = evaluate_accuracy(model, dev_data, batch_size, gpu)

        # print out report of current iteration
        print("iteration: {:>7,} train loss: {:>12,.3f} train_acc: {:>8,.3f}% "
              "dev loss: {:>12,.3f} dev_acc: {:>8,.3f}%".format(
                  epoch, loss / len(train_data), train_acc * 100,
                  dev_loss / len(dev_data), dev_acc * 100))

        # check for loss improvement and save model if there is reduction
        # optionally increment patience counter or stop training
        # NOTE: loss values are summed over all data (not mean)
        if dev_loss < best_dev_loss:
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                print("New best acc!")
            print("New best dev!")
            best_dev_loss = dev_loss
            best_dev_loss_index = 0
            if models_directory is not None:
                model_save_file = os.path.join(
                    models_directory,
                    "{}_{}.pth".format(model_file_prefix, epoch))
                print("saving model to", model_save_file)
                torch.save(model.state_dict(), model_save_file)
        else:
            best_dev_loss_index += 1
            if best_dev_loss_index == patience:
                print("Reached", patience,
                      "iterations without improving dev loss. Breaking")
                break

        # check for improvement in dev best accuracy
        # TODO: likely remove this block since it is not necessary
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            print("New best acc!")
            if models_directory is not None:
                model_save_file = os.path.join(
                    models_directory,
                    "{}_{}.pth".format(model_file_prefix, epoch))
                print("saving model to", model_save_file)
                torch.save(model.state_dict(), model_save_file)

        # apply learning rate scheduler after epoch
        if run_scheduler:
            scheduler.step(dev_loss)


def main(args: argparse.Namespace) -> None:
    # print namespace arguments
    print(args)

    # read important arguments and define as local variables
    num_train_instances = args.num_train_instances
    mlp_hidden_dim = args.mlp_hidden_dim
    mlp_num_layers = args.mlp_num_layers
    models_directory = args.models_directory
    epochs = args.epochs

    # set default temporary value for pre_computed_patterns
    pre_computed_patterns = None

    # convert pattern_specs string in OrderedDict
    pattern_specs: MutableMapping[int, int] = OrderedDict(
        sorted(
            (
                [int(y) for y in x.split("-")]  # type: ignore
                for x in args.patterns.split("_")),
            key=lambda t: t[0]))

    # read pre_computed_patterns if it exists, format pattern_specs accordingly
    if args.pre_computed_patterns is not None:
        pre_computed_patterns = read_patterns(args.pre_computed_patterns,
                                              pattern_specs)
        pattern_specs = OrderedDict(
            sorted(pattern_specs.items(), key=lambda t: t[0]))

    # set global random seed if specified
    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # read dev and train vocabularies
    dev_vocab = vocab_from_text(args.valid_data)
    print("Dev vocab size:", len(dev_vocab))
    train_vocab = vocab_from_text(args.train_data)
    print("Train vocab size:", len(train_vocab))

    # combine dev and train vocabularies into global object
    vocab = dev_vocab | train_vocab

    # read embeddings file and output intersected vocab
    # embeddings and word-vector dimensionality
    vocab, embeddings, word_dim = read_embeddings(args.embeddings, vocab)

    # set number of padding tokens as one less than the longest pattern length
    # TODO: understand why this is set and if this is necessary
    num_padding_tokens = max(list(pattern_specs.keys())) - 1

    # read development data and shuffle
    dev_input, _ = read_docs(args.valid_data,
                             vocab,
                             num_padding_tokens=num_padding_tokens)
    dev_labels = read_labels(args.valid_labels)
    dev_input = cast(List[List[int]], dev_input)
    dev_data = list(zip(dev_input, dev_labels))
    np.random.shuffle(dev_data)

    # read train data and shuffle
    train_input, _ = read_docs(args.train_data,
                               vocab,
                               num_padding_tokens=num_padding_tokens)
    train_input = cast(List[List[int]], train_input)
    train_labels = read_labels(args.train_labels)
    num_classes = len(set(train_labels))
    train_data = list(zip(train_input, train_labels))
    np.random.shuffle(train_data)

    # print diagnostic information
    print("training instances:", len(train_input))
    print("num_classes:", num_classes)

    # truncate data if necessary
    if num_train_instances is not None:
        train_data = train_data[:num_train_instances]
        dev_data = dev_data[:num_train_instances]

    # define semiring as per argument provided
    semiring = MaxPlusSemiring if args.max_plus_semiring else (
        LogSpaceMaxTimesSemiring if args.max_times_semiring else ProbSemiring)

    # create SoftPatternClassifier
    model = SoftPatternClassifier(pattern_specs, mlp_hidden_dim,
                                  mlp_num_layers, num_classes, embeddings,
                                  vocab, semiring, args.bias_scale, args.gpu,
                                  pre_computed_patterns, args.no_sl,
                                  args.shared_sl, args.no_eps, args.eps_scale,
                                  args.self_loop_scale)

    # send model to GPU if present
    if args.gpu:
        model.to_cuda(model)

    # define model prefix
    # TODO: format this better with timestamping
    model_file_prefix = "model"

    # load model if argument provided
    if args.load_model is not None:
        state_dict = torch.load(args.load_model)
        model.load_state_dict(state_dict)
        # TODO: format this better with timestamping
        model_file_prefix = 'model_retrained'

    # create models_directory if argument provided
    if models_directory is not None:
        if not os.path.exists(models_directory):
            os.makedirs(models_directory)

    # train SoftPatternClassifier
    train(train_data, dev_data, model, num_classes, models_directory, epochs,
          model_file_prefix, args.learning_rate, args.batch_size,
          args.scheduler, args.gpu, args.clip_threshold, args.max_doc_len,
          args.dropout, args.word_dropout, args.patience)


def read_patterns(filename: str,
                  pattern_specs: MutableMapping[int, int]) -> List[List[str]]:
    # read pre_compute_patterns into a list
    with open(filename, encoding='utf-8') as input_file_stream:
        pre_computed_patterns = [
            line.rstrip().split() for line in input_file_stream
            if len(line.rstrip())
        ]

    # update pattern_specs object with patterns metadata
    for pattern in pre_computed_patterns:
        lookup_length = len(pattern) + 1
        if lookup_length not in pattern_specs:
            pattern_specs[lookup_length] = 1
        else:
            pattern_specs[lookup_length] += 1

    # return read object
    return pre_computed_patterns


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse_formatter,
        parents=[training_arg_parser(),
                 soft_patterns_pp_arg_parser()])
    args = parser.parse_args()
    main(args)
