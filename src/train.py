#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
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
from .utils.logging_utils import make_logger
from .soft_patterns_pp import SoftPatternClassifier
from .arg_parser import (soft_patterns_pp_arg_parser, training_arg_parser,
                         logging_arg_parser)
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

    # set model in evaluation mode
    model.eval()

    # chunk data into sorted batches and iterate
    for batch in chunked_sorted(data, batch_size):
        # create batch and parse gold output
        batch, gold = Batch([x for x, y in batch], model.embeddings,
                            to_cuda(gpu)), [y for x, y in batch]

        # predict output using model
        predicted = model.predict(batch)

        # find number of correctly predicted data points
        correct += sum(1 for pred, gold in zip(predicted, gold)
                       if pred == gold)

    # set model back to train mode
    model.train()

    # return raw accuracy float
    # TODO: replace this workflow with more robust metric such as F1 score
    return correct / number_data_points


def train(train_data: List[Tuple[List[int], int]],
          valid_data: List[Tuple[List[int], int]],
          model: Module,
          num_classes: int,
          models_directory: str,
          epochs: int,
          model_file_prefix: str,
          learning_rate: float,
          batch_size: int,
          use_scheduler: bool = False,
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

    # initialize tensorboard writer if provided
    writer = None
    if models_directory is not None:
        writer = SummaryWriter(os.path.join(models_directory, "logs"))

    # initialize learning rate scheduler if provided
    # TODO boolean below does not correspond correctly, add verbosity flag
    if use_scheduler:
        logger.info("Initializing learning rate scheduler")
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=0.1,
                                      patience=10,
                                      verbose=True)

    # initialize floats for re-use
    best_valid_loss = 100000000.
    best_valid_loss_index = -1
    best_valid_acc = -1.

    # loop over epochs
    for epoch in range(epochs):
        # shuffle training data
        np.random.shuffle(train_data)

        # initialize training and valid loss
        loss = 0.0
        valid_loss = 0.0

        # main training loop
        logger.info("Training SoPa++ model")
        with tqdm(shuffled_chunked_sorted(train_data, batch_size),
                  disable=disable_tqdm,
                  unit="batch") as tqdm_batches:
            # loop over shuffled train batches
            for batch in tqdm_batches:
                # create batch object and parse out gold labels
                batch, gold = Batch([x[0] for x in batch], model.embeddings,
                                    to_cuda(gpu), word_dropout,
                                    max_doc_len), [x[1] for x in batch]

                # find aggregate loss across samples in batch
                loss += torch.sum(
                    train_batch(model, batch, num_classes, gold, optimizer,
                                loss_function, gpu, dropout))

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

        # loop over static valid set
        logger.info("Evaluating SoPa++ model on validation set")
        with tqdm(chunked_sorted(valid_data, batch_size),
                  disable=disable_tqdm,
                  unit="batch") as tqdm_batches:
            for batch in tqdm_batches:
                # create batch object and parse out gold labels
                batch, gold = Batch([x[0] for x in batch], model.embeddings,
                                    to_cuda(gpu)), [x[1] for x in batch]

                # find aggregate loss across valid samples in batch
                valid_loss += torch.sum(
                    compute_loss(model, batch, num_classes, gold,
                                 loss_function, gpu).data)

        # add valid loss data to tensorboard
        if writer is not None:
            writer.add_scalar("loss/loss_valid", valid_loss, epoch)

        # evaluate training and valid accuracies
        # TODO: do not limit training data accuracy here
        train_acc = evaluate_accuracy(model, train_data[:1000], batch_size,
                                      gpu)
        valid_acc = evaluate_accuracy(model, valid_data, batch_size, gpu)

        # log out report of current epoch
        logger.info("epoch: {}, train_loss: {:.3f}, train_acc: {:.3f}%, "
                    "valid_loss: {:.3f}, valid_acc: {:.3f}%".format(
                        epoch, loss / len(train_data), train_acc * 100,
                        valid_loss / len(valid_data), valid_acc * 100))

        # check for loss improvement and save model if there is reduction
        # optionally increment patience counter or stop training
        # NOTE: loss values are summed over all data (not mean)
        if valid_loss < best_valid_loss:
            logger.info("New best validation loss")
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                logger.info("New best validation accuracy")
            best_valid_loss = valid_loss
            best_valid_loss_index = 0
            if models_directory is not None:
                model_save_file = os.path.join(
                    models_directory,
                    "{}_{}.pth".format(model_file_prefix, epoch))
                logger.info("Saving checkpoint: %s" % model_save_file)
                torch.save(model.state_dict(), model_save_file)
        else:
            best_valid_loss_index += 1
            if best_valid_loss_index == patience:
                logger.info(
                    "%s patience epochs threshold reached, stopping training" %
                    patience)
                return None

        # apply learning rate scheduler after epoch
        if use_scheduler:
            scheduler.step(valid_loss)

    # log information at the end of training
    logger.info("%s training epochs completed, stopping training" % epochs)


def main(args: argparse.Namespace) -> None:
    # log namespace arguments
    logger.info(args)

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

    # log diagnositc information on input patterns
    logger.info("Patterns: %s" % pattern_specs)

    # set global random seed if specified
    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # read valid and train vocabularies
    train_vocab = vocab_from_text(args.train_data)
    logger.info("Training vocabulary size: %s" % len(train_vocab))
    valid_vocab = vocab_from_text(args.valid_data)
    logger.info("Validation vocabulary size: %s" % len(valid_vocab))

    # combine valid and train vocabularies into global object
    vocab = valid_vocab | train_vocab

    # read embeddings file and output intersected vocab
    # embeddings and word-vector dimensionality
    vocab, embeddings, _ = read_embeddings(args.embeddings, vocab)

    # set number of padding tokens as one less than the longest pattern length
    # TODO: understand why this is set and if this is necessary
    num_padding_tokens = max(list(pattern_specs.keys())) - 1

    # read validation data and shuffle
    valid_input, _ = read_docs(args.valid_data,
                               vocab,
                               num_padding_tokens=num_padding_tokens)
    valid_labels = read_labels(args.valid_labels)
    valid_input = cast(List[List[int]], valid_input)
    valid_data = list(zip(valid_input, valid_labels))
    np.random.shuffle(valid_data)

    # read train data and shuffle
    train_input, _ = read_docs(args.train_data,
                               vocab,
                               num_padding_tokens=num_padding_tokens)
    train_input = cast(List[List[int]], train_input)
    train_labels = read_labels(args.train_labels)
    num_classes = len(set(train_labels))
    train_data = list(zip(train_input, train_labels))
    np.random.shuffle(train_data)

    # log diagnostic information
    logger.info("Training instances: %s" % len(train_input))
    logger.info("Number of classes: %s" % num_classes)

    # truncate data if necessary
    if num_train_instances is not None:
        train_data = train_data[:num_train_instances]
        valid_data = valid_data[:num_train_instances]

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

    # log diagnostic information on parameter count
    logger.info("Total model parameters: %s" %
                sum(parameter.nelement() for parameter in model.parameters()))

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
    train(train_data, valid_data, model, num_classes, models_directory, epochs,
          model_file_prefix, args.learning_rate, args.batch_size,
          args.use_scheduler, args.gpu, args.clip_threshold, args.max_doc_len,
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
    parser = argparse.ArgumentParser(formatter_class=argparse_formatter,
                                     parents=[
                                         training_arg_parser(),
                                         soft_patterns_pp_arg_parser(),
                                         logging_arg_parser()
                                     ])
    args = parser.parse_args()
    logger = make_logger(args.logging_level)
    disable_tqdm = args.disable_tqdm
    main(args)
