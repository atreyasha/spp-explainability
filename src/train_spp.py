#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
from glob import glob
from collections import OrderedDict
from typing import List, Union, MutableMapping, Tuple, cast
from torch import LongTensor
from torch.nn import NLLLoss, Module
from torch.nn.functional import log_softmax
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from .utils.parser_utils import ArgparseFormatter
from .utils.data_utils import (vocab_from_text, read_labels, read_docs,
                               read_embeddings)
from .utils.model_utils import (shuffled_chunked_sorted, chunked_sorted,
                                to_cuda, argmax, timestamp, Batch,
                                enable_gradient_clipping, ProbSemiring,
                                LogSpaceMaxTimesSemiring, MaxPlusSemiring)
from .soft_patterns_pp import SoftPatternClassifier
from .arg_parser import (soft_patterns_pp_arg_parser, training_arg_parser,
                         logging_arg_parser)
from .utils.logging_utils import make_logger
import numpy as np
import argparse
import torch
import json
import os


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


def save_checkpoint(epoch: int, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Union[ReduceLROnPlateau, None],
                    best_valid_loss: float, best_valid_loss_index: int,
                    best_valid_acc: float, path: str) -> None:
    torch.save(
        {
            "epoch":
            epoch,
            "model_state_dict":
            model.state_dict(),
            "optimizer_state_dict":
            optimizer.state_dict(),
            "scheduler_state_dict":
            scheduler.state_dict() if scheduler is not None else None,
            "best_valid_loss":
            best_valid_loss,
            "best_valid_loss_index":
            best_valid_loss_index,
            "best_valid_acc":
            best_valid_acc
        }, path)


def train_batch(model: Module,
                batch: Batch,
                num_classes: int,
                gold_output: List[int],
                optimizer: torch.optim.Optimizer,
                loss_function: torch.nn.modules.loss._Loss,
                gpu_device: Union[torch.device, None] = None) -> torch.Tensor:
    # set optimizer gradients to zero
    optimizer.zero_grad()

    # compute model loss
    loss = compute_loss(model,
                        batch,
                        num_classes,
                        gold_output,
                        loss_function,
                        gpu_device)

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
                 gpu_device: Union[torch.device, None]) -> torch.Tensor:
    # compute model outputs given batch
    output = model.forward(batch)

    # return loss over output and gold
    return loss_function(
        log_softmax(output, dim=1).view(batch.size(), num_classes),
        to_cuda(gpu_device)(LongTensor(gold_output)))


def evaluate_accuracy(model: Module, data: List[Tuple[List[int], int]],
                      batch_size: int, gpu_device: Union[torch.device,
                                                         None]) -> float:
    # instantiate local variables
    number_data_points = float(len(data))
    correct = 0

    # chunk data into sorted batches and iterate
    for batch in chunked_sorted(data, batch_size):
        # create batch and parse gold output
        batch, gold = Batch(  # type: ignore
            [x for x, y in batch],
            model.embeddings,  # type: ignore
            to_cuda(gpu_device)), [y for x, y in batch]

        # get raw output using model
        output = model.forward(batch)  # type: ignore

        # get predicted classes from raw output
        predicted = [int(x) for x in argmax(output)]

        # find number of correctly predicted data points
        correct += sum(1 for pred, gold in zip(predicted, gold)
                       if pred == gold)

    # return raw accuracy float
    return correct / number_data_points


def train(train_data: List[Tuple[List[int], int]],
          valid_data: List[Tuple[List[int], int]],
          model: Module,
          num_classes: int,
          epochs: int,
          model_log_directory: str,
          learning_rate: float,
          batch_size: int,
          use_scheduler: bool = False,
          gpu_device: Union[torch.device, None] = None,
          clip_threshold: Union[float, None] = None,
          max_doc_len: int = -1,
          word_dropout: float = 0,
          patience: int = 30) -> None:
    # instantiate Adam optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # instantiate negative log-likelihood loss
    # NOTE: reduce by summing over batch instead of default 'mean'
    loss_function = NLLLoss(weight=None, reduction="sum")

    # enable gradient clipping in-place if provided
    enable_gradient_clipping(model, clip_threshold)

    # initialize tensorboard writer if provided
    writer = SummaryWriter(os.path.join(model_log_directory, "events"))

    # initialize learning rate scheduler if provided
    # TODO boolean below does not correspond correctly, add verbosity flag
    if use_scheduler:
        LOGGER.info("Initializing learning rate scheduler")
        scheduler: Union[ReduceLROnPlateau, None]
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=0.1,
                                      patience=10,
                                      verbose=True)
    else:
        scheduler = None

    # initialize floats for re-use
    best_valid_loss = 100000000.
    best_valid_loss_index = 0
    best_valid_acc = -1.

    # loop over epochs
    for epoch in range(epochs):
        # set model on train mode
        model.train()

        # shuffle training data
        np.random.shuffle(train_data)

        # initialize training and valid loss, and hook to stop training
        train_loss = 0.
        valid_loss = 0.
        stop_training = False

        # main training loop
        LOGGER.info("Training SoPa++ model")
        with tqdm(shuffled_chunked_sorted(train_data, batch_size),
                  disable=DISABLE_TQDM,
                  unit="batch",
                  desc="Training [Epoch %s/%s]" %
                  (epoch + 1, epochs)) as tqdm_batches:
            # loop over shuffled train batches
            for i, batch in enumerate(tqdm_batches):
                # create batch object and parse out gold labels
                batch, gold = Batch(
                    [x[0] for x in batch],
                    model.embeddings,  # type: ignore
                    to_cuda(gpu_device),
                    word_dropout,
                    max_doc_len), [x[1] for x in batch]

                # find aggregate loss across samples in batch
                train_batch_loss = train_batch(model, batch, num_classes, gold,
                                               optimizer, loss_function,
                                               gpu_device)

                # add batch loss to train_loss
                train_loss += train_batch_loss  # type: ignore

                # update tqdm progress bar
                if (i + 1) % TQDM_UPDATE_FREQ == 0 or (i +
                                                       1) == len(tqdm_batches):
                    tqdm_batches.set_postfix(
                        batch_loss=train_batch_loss.item() / batch.size())

        # set model on eval mode
        model.eval()

        # compute mean train loss over epoch and accuracy
        # NOTE: mean_train_loss contains stochastic noise due to dropout
        LOGGER.info("Evaluating SoPa++ on training set")
        mean_train_loss = train_loss / len(train_data)
        with torch.no_grad():
            train_acc = evaluate_accuracy(model, train_data, batch_size,
                                          gpu_device)

        # add training loss data
        writer.add_scalar("loss/train_loss", mean_train_loss, epoch)
        writer.add_scalar("accuracy/train_accuracy", train_acc, epoch)

        # add named parameter data
        for name, param in model.named_parameters():
            writer.add_scalar("parameter_mean/" + name,
                              param.detach().mean(), epoch)
            writer.add_scalar("parameter_std/" + name,
                              param.detach().std(), epoch)
            if param.grad is not None:
                writer.add_scalar("gradient_mean/" + name,
                                  param.grad.detach().mean(), epoch)
                writer.add_scalar("gradient_std/" + name,
                                  param.grad.detach().std(), epoch)

        # loop over static valid set
        LOGGER.info("Evaluating SoPa++ on validation set")
        with tqdm(chunked_sorted(valid_data, batch_size),
                  disable=DISABLE_TQDM,
                  unit="batch",
                  desc="Validating [Epoch %s/%s]" %
                  (epoch + 1, epochs)) as tqdm_batches:
            # disable autograd since we are inferring here
            with torch.no_grad():
                for i, batch in enumerate(tqdm_batches):
                    # create batch object and parse out gold labels
                    batch, gold = Batch(
                        [x[0] for x in batch],
                        model.embeddings,  # type: ignore
                        to_cuda(gpu_device)), [x[1] for x in batch]

                    # find aggregate loss across valid samples in batch
                    valid_batch_loss = compute_loss(model,
                                                    batch,
                                                    num_classes,
                                                    gold,
                                                    loss_function,
                                                    gpu_device)

                    # add batch loss to valid_loss
                    valid_loss += valid_batch_loss  # type: ignore

                    if (i + 1) % TQDM_UPDATE_FREQ == 0 or (
                            i + 1) == len(tqdm_batches):
                        tqdm_batches.set_postfix(
                            batch_loss=valid_batch_loss.item() / batch.size())

        # compute mean valid loss over epoch and accuracy
        mean_valid_loss = valid_loss / len(valid_data)
        with torch.no_grad():
            valid_acc = evaluate_accuracy(model, valid_data, batch_size,
                                          gpu_device)

        # add valid loss data to tensorboard
        writer.add_scalar("loss/valid_loss", mean_valid_loss, epoch)
        writer.add_scalar("accuracy/valid_accuracy", valid_acc, epoch)

        # log out report of current epoch
        LOGGER.info(
            "epoch: {}/{}, mean_train_loss: {:.3f}, train_acc: {:.3f}%, "
            "mean_valid_loss: {:.3f}, valid_acc: {:.3f}%".format(
                epoch + 1, epochs, mean_train_loss, train_acc * 100,
                mean_valid_loss, valid_acc * 100))

        # apply learning rate scheduler after epoc
        if scheduler is not None:
            scheduler.step(valid_loss)

        # check for loss improvement and save model if there is reduction
        # optionally increment patience counter or stop training
        # NOTE: loss values are summed over all data (not mean)
        if valid_loss < best_valid_loss:
            # log information and update records
            LOGGER.info("New best validation loss")
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                LOGGER.info("New best validation accuracy")

            # update patience related diagnostics
            best_valid_loss = valid_loss
            best_valid_loss_index = 0
            LOGGER.info("Patience counter: %s/%s" %
                        (best_valid_loss_index, patience))

            # find previous best checkpoint(s)
            legacy_checkpoints = glob(
                os.path.join(model_log_directory, "*_best_*.pt"))

            # save new best checkpoint
            model_save_file = os.path.join(
                model_log_directory, "spp_checkpoint_best_{}.pt".format(epoch))
            LOGGER.info("Saving best checkpoint: %s" % model_save_file)
            save_checkpoint(epoch, model, optimizer, scheduler,
                            best_valid_loss, best_valid_loss_index,
                            best_valid_acc, model_save_file)

            # delete previous best checkpoint(s)
            for legacy_checkpoint in legacy_checkpoints:
                os.remove(legacy_checkpoint)
        else:
            # update patience related diagnostics
            best_valid_loss_index += 1
            LOGGER.info("Patience counter: %s/%s" %
                        (best_valid_loss_index, patience))

            # create hook to exit training if patience threshold reached
            # TODO: what if index is already more than continue train patience
            # can overcome by changing this to >= and writing note
            # or otherwise check this before training continues
            if best_valid_loss_index == patience:
                LOGGER.info(
                    "%s patience epochs threshold reached, stopping training" %
                    patience)
                stop_training = True

        # find previous last checkpoint(s)
        legacy_checkpoints = glob(
            os.path.join(model_log_directory, "*_last_*.pt"))

        # save latest checkpoint
        model_save_file = os.path.join(
            model_log_directory, "spp_checkpoint_last_{}.pt".format(epoch))
        LOGGER.info("Saving last checkpoint: %s" % model_save_file)
        save_checkpoint(epoch, model, optimizer, scheduler, best_valid_loss,
                        best_valid_loss_index, best_valid_acc, model_save_file)

        # delete previous last checkpoint(s)
        for legacy_checkpoint in legacy_checkpoints:
            os.remove(legacy_checkpoint)

        # hook to stop training in case patience threshold was reached
        if stop_training:
            return None

    # log information at the end of training
    LOGGER.info("%s training epochs completed, stopping training" % epochs)


def main(args: argparse.Namespace) -> None:
    # log namespace arguments
    LOGGER.info(args)

    # set torch number of threads
    if args.num_threads is None:
        LOGGER.info("Using default number of CPU threads: %s" %
                    torch.get_num_threads())
    else:
        torch.set_num_threads(args.num_threads)
        LOGGER.info("Using specified number of CPU threads: %s" %
                    args.num_threads)

    # specify cpu device
    cpu_device = torch.device("cpu")

    # specify gpu device if relevant
    if args.gpu:
        gpu_device: Union[torch.device, None]
        gpu_device = torch.device(args.gpu_device)
        LOGGER.info("Using GPU device: %s" % args.gpu_device)
    else:
        gpu_device = None

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
    LOGGER.info("Patterns: %s" % pattern_specs)

    # set global random seed if specified
    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # read valid and train vocabularies
    train_vocab = vocab_from_text(args.train_data)
    LOGGER.info("Training vocabulary size: %s" % len(train_vocab))
    valid_vocab = vocab_from_text(args.valid_data)
    LOGGER.info("Validation vocabulary size: %s" % len(valid_vocab))

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

    # truncate data if necessary
    if num_train_instances is not None:
        train_data = train_data[:num_train_instances]
        valid_data = valid_data[:num_train_instances]

    # log diagnostic information
    LOGGER.info("Number of classes: %s" % num_classes)
    LOGGER.info("Training instances: %s" % len(train_data))
    LOGGER.info("Validation instances: %s" % len(valid_data))

    # define semiring as per argument provided and log
    if args.semiring == "MaxPlusSemiring":
        semiring = MaxPlusSemiring
    elif args.semiring == "MaxTimesSemiring":
        semiring = LogSpaceMaxTimesSemiring
    elif args.semiring == "ProbSemiring":
        semiring = ProbSemiring
    LOGGER.info("Semiring: %s" % args.semiring)

    # create SoftPatternClassifier
    model = SoftPatternClassifier(pattern_specs, mlp_hidden_dim,
                                  mlp_num_layers, num_classes, embeddings,
                                  vocab, semiring, args.bias_scale,
                                  pre_computed_patterns, args.no_self_loops,
                                  args.shared_self_loops, args.no_epsilons,
                                  args.epsilon_scale, args.self_loop_scale,
                                  args.dropout)

    # log diagnostic information on parameter count
    LOGGER.info("Total model parameters: %s" %
                sum(parameter.nelement() for parameter in model.parameters()))

    # define model log directory
    if args.load_model is None:
        # create model log directory
        model_log_directory = os.path.join(models_directory,
                                           "spp_single_train_" + timestamp())
        os.makedirs(model_log_directory, exist_ok=True)

        # extract relevant arguments for spp model
        soft_pattern_args = soft_patterns_pp_arg_parser().parse_args("")
        for key in soft_pattern_args.__dict__:
            if key in args.__dict__:
                setattr(soft_pattern_args, key, getattr(args, key))

        # dump soft patterns model arguments for posterity
        with open(
                os.path.join(model_log_directory,
                             "soft_patterns_pp_config.json"),
                "w") as output_file_stream:
            json.dump(soft_pattern_args.__dict__,
                      output_file_stream,
                      ensure_ascii=False)
    else:
        # load model if argument provided
        # TODO: load with cpu map location -> use earlier device for this
        state_dict = torch.load(args.load_model)
        model.load_state_dict(state_dict)
        # TODO: improve this workflow to borrow from input path
        # TODO: perhaps pass this workflow directly into train
        # and use the same directory for continuing training
        model_log_directory = 'model_retrained'

    # send model to GPU if present
    if gpu_device is not None:
        model.to(gpu_device)  # type: ignore

    # train SoftPatternClassifier
    train(train_data, valid_data, model, num_classes, epochs,
          model_log_directory, args.learning_rate, args.batch_size,
          args.use_scheduler, gpu_device, args.clip_threshold,
          args.max_doc_len, args.word_dropout, args.patience)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter,
                                     parents=[
                                         training_arg_parser(),
                                         soft_patterns_pp_arg_parser(),
                                         logging_arg_parser()
                                     ])
    args = parser.parse_args()
    LOGGER = make_logger(args.logging_level)
    DISABLE_TQDM = args.disable_tqdm
    TQDM_UPDATE_FREQ = args.tqdm_update_freq
    main(args)
