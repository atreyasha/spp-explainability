#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
from glob import glob
from math import ceil
from functools import partial
from collections import OrderedDict
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.nn import NLLLoss, Module, Embedding
from torch.nn.functional import log_softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
from typing import List, Union, Tuple, cast, Any, Callable
from .utils.parser_utils import ArgparseFormatter
from .utils.data_utils import (vocab_from_text, read_labels, read_docs,
                               read_embeddings, Vocab, PAD_TOKEN_INDEX)
from .utils.model_utils import (shuffled_chunked_sorted, chunked_sorted,
                                to_cuda, enable_gradient_clipping, timestamp,
                                Batch, Semiring, LogSpaceMaxProductSemiring,
                                MaxSumSemiring)
from .utils.logging_utils import (stdout_root_logger, add_file_handler,
                                  remove_all_file_handlers)
from .arg_parser import (spp_arg_parser, train_arg_parser, logging_arg_parser,
                         tqdm_arg_parser, hardware_arg_parser,
                         grid_train_arg_parser)
from .torch_module_spp import SoftPatternClassifier
import numpy as np
import argparse
import logging
import signal
import torch
import copy
import json
import sys
import os

# get root LOGGER in case script is called by another
LOGGER = logging.getLogger(__name__)

# define exit-codes
FINISHED_EPOCHS = 0
PATIENCE_REACHED = 1
INTERRUPTION = 2


def signal_handler(filename: str, *args):
    save_exit_code(filename, INTERRUPTION)
    sys.exit()


def save_exit_code(filename: str, code: int) -> None:
    with open(filename, "w") as output_file_stream:
        output_file_stream.write("%s\n" % code)


def get_exit_code(filename: str) -> int:
    with open(filename, "r") as input_file_stream:
        exit_code = int(input_file_stream.readline().strip())
    return exit_code


def parse_configs_to_args(args: argparse.Namespace,
                          prefix: str = "",
                          training: bool = True) -> argparse.Namespace:
    # make copy of existing argument namespace
    args = copy.deepcopy(args)

    # check for json configs and add them to list
    json_files = []
    json_files.append(
        os.path.join(args.model_log_directory, prefix + "model_config.json"))
    if training:
        json_files.append(
            os.path.join(args.model_log_directory,
                         prefix + "training_config.json"))

    # raise error if any of them are missing
    for json_file in json_files:
        if not os.path.exists(json_file):
            raise FileNotFoundError("File not found: %s" % json_file)

    # update argument namespace with information from json files
    for json_file in json_files:
        with open(json_file, "r") as input_file_stream:
            args.__dict__.update(json.load(input_file_stream))
    return args


def set_hardware(args: argparse.Namespace) -> Union[torch.device, None]:
    # set torch number of threads
    if args.torch_num_threads is None:
        LOGGER.info("Using default number of CPU threads: %s" %
                    torch.get_num_threads())
    else:
        torch.set_num_threads(args.torch_num_threads)
        LOGGER.info("Using specified number of CPU threads: %s" %
                    args.torch_num_threads)

    # specify gpu device if relevant
    if args.gpu:
        gpu_device: Union[torch.device, None]
        gpu_device = torch.device(args.gpu_device)
        LOGGER.info("Using GPU device: %s" % args.gpu_device)
    else:
        gpu_device = None
        LOGGER.info("Using CPU device")

    # return device
    return gpu_device


def get_grid_config(args: argparse.Namespace) -> dict:
    with open(args.grid_config, "r") as input_file_stream:
        grid_dict = json.load(input_file_stream)
    return grid_dict


def get_grid_args_superset(
        args: argparse.Namespace,
        param_grid_mapping: dict) -> List[argparse.Namespace]:
    args_superset = []
    # ensure param_grid_mapping keys are integers
    param_grid_mapping = {
        int(key): value
        for key, value in param_grid_mapping.items()
    }
    for i in sorted(param_grid_mapping.keys()):
        param_grid_instance = param_grid_mapping[i]
        args_copy = copy.deepcopy(args)
        for key in param_grid_instance:
            if key in args.__dict__:
                setattr(args_copy, key, param_grid_instance[key])
        args_superset.append(args_copy)
    return args_superset


def get_pattern_specs(args: argparse.Namespace) -> 'OrderedDict[int, int]':
    # convert pattern_specs string in OrderedDict
    pattern_specs: 'OrderedDict[int, int]' = OrderedDict(
        sorted(
            (
                [int(y) for y in x.split("-")]  # type: ignore
                for x in args.patterns.split("_")),
            key=lambda t: t[0]))

    # log diagnositc information on input patterns
    LOGGER.info("Patterns: %s" % pattern_specs)

    # return final objects
    return pattern_specs


def set_random_seed(args: argparse.Namespace) -> None:
    # set global random seed if specified
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_vocab(args: argparse.Namespace) -> Vocab:
    # read valid and train vocabularies
    train_vocab = vocab_from_text(args.train_data)
    LOGGER.info("Training vocabulary size: %s" % len(train_vocab))
    valid_vocab = vocab_from_text(args.valid_data)
    LOGGER.info("Validation vocabulary size: %s" % len(valid_vocab))

    # combine valid and train vocabularies into combined object
    vocab_combined = valid_vocab | train_vocab
    LOGGER.info("Combined vocabulary size: %s" % len(vocab_combined))

    # return final Vocab object
    return vocab_combined


def get_embeddings(args: argparse.Namespace,
                   vocab_combined: Vocab) -> Tuple[Vocab, torch.Tensor, int]:
    # read embeddings file and output intersected vocab
    vocab, embeddings, word_dim = read_embeddings(args.embeddings,
                                                  vocab_combined)
    # convert embeddings to torch FloatTensor
    embeddings = np.vstack(embeddings).astype(np.float32)
    embeddings = torch.from_numpy(embeddings)

    # return final tuple
    return vocab, embeddings, word_dim


def get_vocab_diagnostics(vocab: Vocab, vocab_combined: Vocab,
                          word_dim: int) -> None:
    # show output of tokens lost during vocabulary extraction
    missing = [
        token for token in vocab_combined.names if token not in vocab.names
    ]
    LOGGER.info("GloVe embedding dimensions: %s" % word_dim)
    LOGGER.info("GloVe-intersected vocabulary size: %s" % len(vocab))
    LOGGER.info("Number of tokens not found in GloVe vocabulary: %s" %
                len(missing))
    LOGGER.info("Lost tokens: %s" % missing)


def get_train_valid_data(
    args: argparse.Namespace, vocab: Vocab
) -> Tuple[List[List[str]], List[List[str]], List[Tuple[List[int], int]],
           List[Tuple[List[int], int]], int]:
    # read train data
    train_input, train_text = read_docs(args.train_data, vocab)
    LOGGER.info("Sample training text: %s" % train_text[:10])
    train_input = cast(List[List[int]], train_input)
    train_text = cast(List[List[str]], train_text)
    train_labels = read_labels(args.train_labels)
    num_classes = len(set(train_labels))
    train_data = list(zip(train_input, train_labels))

    # read validation data
    valid_input, valid_text = read_docs(args.valid_data, vocab)
    LOGGER.info("Sample validation text: %s" % valid_text[:10])
    valid_input = cast(List[List[int]], valid_input)
    valid_text = cast(List[List[str]], valid_text)
    valid_labels = read_labels(args.valid_labels)
    valid_data = list(zip(valid_input, valid_labels))

    # truncate data if necessary
    if args.num_train_instances is not None:
        train_data = train_data[:args.num_train_instances]
        valid_data = valid_data[:args.num_train_instances]
        train_text = train_text[:args.num_train_instances]
        valid_text = valid_text[:args.num_train_instances]

    # log diagnostic information
    LOGGER.info("Number of classes: %s" % num_classes)
    LOGGER.info("Training instances: %s" % len(train_data))
    LOGGER.info("Validation instances: %s" % len(valid_data))

    # return final tuple object
    return train_text, valid_text, train_data, valid_data, num_classes


def get_semiring(args: argparse.Namespace) -> Semiring:
    # define semiring as per argument provided and log
    if args.semiring == "MaxSumSemiring":
        semiring = MaxSumSemiring
    elif args.semiring == "MaxProductSemiring":
        semiring = LogSpaceMaxProductSemiring
    LOGGER.info("Semiring: %s" % args.semiring)
    return semiring


def dump_configs(args: argparse.Namespace,
                 model_log_directory: str,
                 prefix: str = "") -> None:
    # create dictionaries to fill up
    spp_args_dict = {}
    train_args_dict = {}

    # extract real arguments and fill up model dictionary
    for action in spp_arg_parser()._actions:
        spp_args_dict[action.dest] = getattr(args, action.dest)

    # extract real arguments and fill up training dictionary
    for action in train_arg_parser()._actions:
        train_args_dict[action.dest] = getattr(args, action.dest)

    # dump soft patterns model arguments for posterity
    with open(os.path.join(model_log_directory, prefix + "model_config.json"),
              "w") as output_file_stream:
        json.dump(spp_args_dict, output_file_stream, ensure_ascii=False)

    # dump training arguments for posterity
    with open(
            os.path.join(model_log_directory, prefix + "training_config.json"),
            "w") as output_file_stream:
        json.dump(train_args_dict, output_file_stream, ensure_ascii=False)


def save_checkpoint(epoch: int, update: int, samples_seen: int,
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: Union[ReduceLROnPlateau, None],
                    numpy_epoch_random_state: Tuple, train_loss: float,
                    best_valid_loss: float, best_valid_loss_index: int,
                    best_valid_acc: float, filename: str) -> None:
    torch.save(
        {
            "epoch":
            epoch,
            "update":
            update,
            "samples_seen":
            samples_seen,
            "model_state_dict":
            model.state_dict(),
            "optimizer_state_dict":
            optimizer.state_dict(),
            "scheduler_state_dict":
            scheduler.state_dict() if scheduler is not None else None,
            "train_loss":
            train_loss,
            "best_valid_loss":
            best_valid_loss,
            "best_valid_loss_index":
            best_valid_loss_index,
            "best_valid_acc":
            best_valid_acc,
            "numpy_epoch_random_state":
            numpy_epoch_random_state,
            "numpy_last_random_state":
            np.random.get_state(),
            "torch_last_random_state":
            torch.random.get_rng_state()
        }, filename)


def train_batch(model: Module,
                batch: Batch,
                num_classes: int,
                gold: List[int],
                optimizer: torch.optim.Optimizer,
                loss_function: torch.nn.modules.loss._Loss,
                gpu_device: Union[torch.device, None] = None) -> torch.Tensor:
    # set optimizer gradients to zero
    optimizer.zero_grad()

    # compute model loss
    loss = compute_loss(model, batch, num_classes, gold, loss_function,
                        gpu_device)

    # compute loss gradients for all parameters
    loss.backward()

    # perform a single optimization step
    optimizer.step()

    # detach loss from computational graph and return tensor data
    return loss.detach()


def compute_loss(model: Module, batch: Batch, num_classes: int,
                 gold: List[int], loss_function: torch.nn.modules.loss._Loss,
                 gpu_device: Union[torch.device, None]) -> torch.Tensor:
    # compute model outputs given batch
    output = model.forward(batch)

    # return loss over output and gold
    return loss_function(
        log_softmax(output, dim=1).view(batch.size(), num_classes),
        to_cuda(gpu_device)(torch.LongTensor(gold)))


def evaluate_metric(model: Module,
                    data: List[Tuple[List[int], int]],
                    batch_size: int,
                    gpu_device: Union[torch.device, None],
                    metric: Callable[[List[int], List[int]], Any],
                    max_doc_len: Union[int, None] = None) -> Any:
    # instantiate local storage variable
    predicted = []
    aggregate_gold = []

    # chunk data into sorted batches and iterate
    for batch in chunked_sorted(data, batch_size):
        # create batch and parse gold output
        batch, gold = Batch(  # type: ignore
            [x for x, y in batch],
            model.embeddings,  # type: ignore
            to_cuda(gpu_device),
            0.,
            max_doc_len), [y for x, y in batch]

        # get raw output using model
        output = model.forward(batch)  # type: ignore

        # get predicted classes from raw output
        predicted.extend(torch.argmax(output, 1).tolist())
        aggregate_gold.extend(gold)

    # return output of metric
    return metric(aggregate_gold, predicted)


def train_inner(train_data: List[Tuple[List[int], int]],
                valid_data: List[Tuple[List[int], int]],
                model: Module,
                num_classes: int,
                epochs: int,
                evaluation_period: int,
                only_epoch_eval: bool,
                model_log_directory: str,
                learning_rate: float,
                batch_size: int,
                disable_scheduler: bool = False,
                scheduler_patience: int = 10,
                scheduler_factor: float = 0.1,
                gpu_device: Union[torch.device, None] = None,
                clip_threshold: Union[float, None] = None,
                max_doc_len: Union[int, None] = None,
                word_dropout: float = 0,
                patience: int = 30,
                resume_training: bool = False,
                disable_tqdm: bool = False,
                tqdm_update_period: int = 1) -> None:
    # create signal handlers in case script receives termination signals
    # adapted from: https://stackoverflow.com/a/31709094
    for specific_signal in [
            signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT
    ]:
        signal.signal(
            specific_signal,
            partial(signal_handler,
                    os.path.join(model_log_directory, "exit_code")))

    # initialize general local variables
    updates_per_epoch = ceil(len(train_data) / batch_size)
    patience_reached = False

    # load model checkpoint if training is being resumed
    if resume_training and len(
            glob(os.path.join(model_log_directory, "*last*.pt"))) > 0:
        model_checkpoint = torch.load(glob(
            os.path.join(model_log_directory, "*last*.pt"))[0],
                                      map_location=torch.device("cpu"))
        model.load_state_dict(
            model_checkpoint["model_state_dict"])  # type: ignore
        if (model_checkpoint["update"] +  # type: ignore
                1) == updates_per_epoch:  # type: ignore
            current_epoch: int = model_checkpoint["epoch"] + 1  # type: ignore
            current_update: int = 0
        else:
            current_epoch: int = model_checkpoint["epoch"]  # type: ignore
            current_update: int = model_checkpoint["update"] + 1  # type: ignore
        best_valid_loss: float = model_checkpoint[  # type: ignore
            "best_valid_loss"]  # type: ignore
        best_valid_loss_index: int = model_checkpoint[  # type: ignore
            "best_valid_loss_index"]  # type: ignore
        best_valid_acc: float = model_checkpoint[  # type: ignore
            "best_valid_acc"]  # type: ignore

        # check for edge-case failures
        if current_epoch >= epochs:
            # log information at the end of training
            LOGGER.info("%s training epoch(s) previously completed, exiting" %
                        epochs)
            # save exit-code and final processes
            save_exit_code(os.path.join(model_log_directory, "exit_code"),
                           FINISHED_EPOCHS)
            return None
        elif best_valid_loss_index >= patience:
            LOGGER.info("Patience threshold previously reached, exiting")
            # save exit-code and final processes
            save_exit_code(os.path.join(model_log_directory, "exit_code"),
                           PATIENCE_REACHED)
            return None
    else:
        resume_training = False
        current_epoch = 0
        current_update = 0
        best_valid_loss_index = 0
        best_valid_loss = float("inf")
        best_valid_acc = float("-inf")

    # send model to correct device
    if gpu_device is not None:
        LOGGER.info("Transferring model to GPU device: %s" % gpu_device)
        model.to(gpu_device)

    # instantiate Adam optimizer
    LOGGER.info("Initializing Adam optimizer with LR: %s" % learning_rate)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # load optimizer state dictionary
    if resume_training:
        optimizer.load_state_dict(
            model_checkpoint["optimizer_state_dict"])  # type: ignore

    # instantiate negative log-likelihood loss which is summed over batch
    LOGGER.info("Using NLLLoss with sum reduction")
    loss_function = NLLLoss(weight=None, reduction="sum")

    # enable gradient clipping in-place if provided
    if clip_threshold is not None and clip_threshold > 0:
        LOGGER.info("Enabling gradient clipping with threshold: %s" %
                    clip_threshold)
        enable_gradient_clipping(model, clip_threshold)

    # initialize learning rate scheduler if relevant
    if not disable_scheduler:
        LOGGER.info(("Initializing learning rate scheduler with "
                     "factor=%s and patience=%s") %
                    (scheduler_factor, scheduler_patience))
        scheduler: Union[ReduceLROnPlateau, None]
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=scheduler_factor,
                                      patience=scheduler_patience,
                                      verbose=True)
        if resume_training:
            scheduler.load_state_dict(
                model_checkpoint["scheduler_state_dict"])  # type: ignore
    else:
        scheduler = None

    # initialize tensorboard writer if provided
    LOGGER.info("Initializing tensorboard writer in directory: %s" %
                os.path.join(model_log_directory, "events"))
    writer = SummaryWriter(os.path.join(model_log_directory, "events"))

    # set numpy and torch RNG back to previous states before training
    if resume_training:
        if current_update == 0:
            np.random.set_state(
                model_checkpoint["numpy_last_random_state"])  # type: ignore
        else:
            np.random.set_state(
                model_checkpoint["numpy_epoch_random_state"])  # type: ignore
        torch.random.set_rng_state(
            model_checkpoint["torch_last_random_state"])  # type: ignore

    # loop over epochs
    for epoch in range(current_epoch, epochs):
        # set model on train mode and enable autograd
        model.train()
        torch.autograd.set_grad_enabled(True)

        # initialize loop variables
        if resume_training and epoch == current_epoch and current_update != 0:
            train_loss: Union[float,
                              torch.Tensor] = model_checkpoint[  # type: ignore
                                  "train_loss"]  # type: ignore
            samples_seen: int = model_checkpoint[  # type: ignore
                "samples_seen"]  # type: ignore
        else:
            train_loss = 0.
            samples_seen = 0

        # cache numpy random state for model checkpoint
        numpy_epoch_random_state = np.random.get_state()

        # main training loop
        LOGGER.info("Training SoPa++ model")
        with tqdm(shuffled_chunked_sorted(train_data, batch_size),
                  position=0,
                  mininterval=0.05,
                  disable=disable_tqdm,
                  unit="batch",
                  desc="Training [Epoch %s/%s]" %
                  (epoch + 1, epochs)) as train_tqdm_batches:
            # loop over train batches
            for update, batch in enumerate(train_tqdm_batches):
                # return to previous update and random state, if relevant
                if (resume_training and epoch == current_epoch
                        and current_update != 0):
                    if update < current_update:
                        continue
                    elif update == current_update:
                        np.random.set_state(model_checkpoint[  # type: ignore
                            "numpy_last_random_state"])  # type: ignore

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

                # increment samples seen
                samples_seen += batch.size()

                # update tqdm progress bar
                if (update + 1) % tqdm_update_period == 0 or (
                        update + 1) == len(train_tqdm_batches):
                    train_tqdm_batches.set_postfix(
                        batch_loss=train_batch_loss.item() / batch.size())

                # start evaluation routine
                if (not only_epoch_eval and (update + 1) % evaluation_period
                        == 0) or (update + 1) == len(train_tqdm_batches):
                    # update tqdm batches counter
                    train_tqdm_batches.update()

                    # set valid loss to zero
                    update_number = (epoch * updates_per_epoch) + (update + 1)
                    valid_loss: Union[float, torch.Tensor] = 0.

                    # set model on eval mode and disable autograd
                    model.eval()
                    torch.autograd.set_grad_enabled(False)

                    # compute mean train loss over updates and accuracy
                    # NOTE: mean_train_loss contains stochastic noise
                    LOGGER.info("Evaluating SoPa++ on training set")
                    train_loss = cast(torch.Tensor, train_loss)
                    mean_train_loss = train_loss.item() / samples_seen
                    train_acc = evaluate_metric(model, train_data, batch_size,
                                                gpu_device, accuracy_score,
                                                max_doc_len)

                    # add training loss data
                    writer.add_scalar("loss/train_loss", mean_train_loss,
                                      update_number)
                    writer.add_scalar("accuracy/train_accuracy", train_acc,
                                      update_number)

                    # add named parameter data
                    for name, param in model.named_parameters():
                        writer.add_scalar("parameter_mean/" + name,
                                          param.detach().mean(), update_number)
                        writer.add_scalar("parameter_std/" + name,
                                          param.detach().std(), update_number)
                        if param.grad is not None:
                            writer.add_scalar("gradient_mean/" + name,
                                              param.grad.detach().mean(),
                                              update_number)
                            writer.add_scalar("gradient_std/" + name,
                                              param.grad.detach().std(),
                                              update_number)

                    # loop over static valid set
                    LOGGER.info("Evaluating SoPa++ on validation set")
                    with tqdm(chunked_sorted(valid_data, batch_size),
                              position=0,
                              mininterval=0.05,
                              disable=disable_tqdm,
                              unit="batch",
                              desc="Validating [Epoch %s/%s] [Batch %s/%s]" %
                              (epoch + 1, epochs, update + 1,
                               updates_per_epoch)) as valid_tqdm_batches:
                        for valid_update, batch in enumerate(
                                valid_tqdm_batches):
                            # create batch object and parse out gold labels
                            batch, gold = Batch(
                                [x[0] for x in batch],
                                model.embeddings,  # type: ignore
                                to_cuda(gpu_device),
                                0.,
                                max_doc_len), [x[1] for x in batch]

                            # find aggregate loss across valid samples in batch
                            valid_batch_loss = compute_loss(
                                model, batch, num_classes, gold, loss_function,
                                gpu_device)

                            # add batch loss to valid_loss
                            valid_loss += valid_batch_loss  # type: ignore

                            if (valid_update +
                                    1) % tqdm_update_period == 0 or (
                                        valid_update +
                                        1) == len(valid_tqdm_batches):
                                valid_tqdm_batches.set_postfix(
                                    batch_loss=valid_batch_loss.item() /
                                    batch.size())

                    # compute mean valid loss and accuracy
                    valid_loss = cast(torch.Tensor, valid_loss)
                    mean_valid_loss = valid_loss.item() / len(valid_data)
                    valid_acc = evaluate_metric(model, valid_data, batch_size,
                                                gpu_device, accuracy_score,
                                                max_doc_len)

                    # set model on train mode and enable autograd
                    model.train()
                    torch.autograd.set_grad_enabled(True)

                    # add valid loss data to tensorboard
                    writer.add_scalar("loss/valid_loss", mean_valid_loss,
                                      update_number)
                    writer.add_scalar("accuracy/valid_accuracy", valid_acc,
                                      update_number)

                    # log out report of current evaluation state
                    LOGGER.info("Epoch: {}/{}, Batch: {}/{}".format(
                        epoch + 1, epochs, (update + 1), updates_per_epoch))
                    LOGGER.info("Mean training loss: {:.3f}, "
                                "Training accuracy: {:.3f}%".format(
                                    mean_train_loss, train_acc * 100))
                    LOGGER.info("Mean validation loss: {:.3f}, "
                                "Validation accuracy: {:.3f}%".format(
                                    mean_valid_loss, valid_acc * 100))

                    # apply learning rate scheduler after evaluation
                    if scheduler is not None:
                        scheduler.step(valid_loss)

                    # check for loss improvement and save model if necessary
                    # optionally increment patience counter or stop training
                    # NOTE: loss values are summed over all data (not mean)
                    if valid_loss.item() < best_valid_loss:
                        # log information and update records
                        LOGGER.info("New best validation loss")
                        if valid_acc > best_valid_acc:
                            best_valid_acc = valid_acc
                            LOGGER.info("New best validation accuracy")

                        # update patience related diagnostics
                        best_valid_loss = valid_loss.item()
                        best_valid_loss_index = 0
                        LOGGER.info("Patience counter: %s/%s" %
                                    (best_valid_loss_index, patience))

                        # find previous best checkpoint(s)
                        legacy_checkpoints = glob(
                            os.path.join(model_log_directory, "*_best_*.pt"))

                        # save new best checkpoint
                        model_save_file = os.path.join(
                            model_log_directory,
                            "spp_checkpoint_best_{}_{}.pt".format(
                                epoch, (update + 1)))
                        LOGGER.info("Saving best checkpoint: %s" %
                                    model_save_file)
                        save_checkpoint(epoch, update, samples_seen, model,
                                        optimizer, scheduler,
                                        numpy_epoch_random_state,
                                        train_loss.item(), best_valid_loss,
                                        best_valid_loss_index, best_valid_acc,
                                        model_save_file)

                        # delete previous best checkpoint(s)
                        for legacy_checkpoint in legacy_checkpoints:
                            os.remove(legacy_checkpoint)
                    else:
                        # update patience related diagnostics
                        best_valid_loss_index += 1
                        LOGGER.info("Patience counter: %s/%s" %
                                    (best_valid_loss_index, patience))

                        # create hook to exit training if patience reached
                        if best_valid_loss_index == patience:
                            patience_reached = True

                    # find previous last checkpoint(s)
                    legacy_checkpoints = glob(
                        os.path.join(model_log_directory, "*_last_*.pt"))

                    # save latest checkpoint
                    model_save_file = os.path.join(
                        model_log_directory,
                        "spp_checkpoint_last_{}_{}.pt".format(
                            epoch, (update + 1)))

                    LOGGER.info("Saving last checkpoint: %s" % model_save_file)
                    save_checkpoint(epoch, update, samples_seen, model,
                                    optimizer,
                                    scheduler, numpy_epoch_random_state,
                                    train_loss.item(), best_valid_loss,
                                    best_valid_loss_index, best_valid_acc,
                                    model_save_file)

                    # delete previous last checkpoint(s)
                    for legacy_checkpoint in legacy_checkpoints:
                        os.remove(legacy_checkpoint)

                    # hook to stop training in case patience was reached
                    # if it was reached strictly before last epoch and update
                    if patience_reached:
                        if not (epoch == max(range(epochs)) and
                                (update + 1) == len(train_tqdm_batches)):
                            LOGGER.info("Patience threshold reached, "
                                        "stopping training")
                            # save exit-code and final processes
                            save_exit_code(
                                os.path.join(model_log_directory, "exit_code"),
                                PATIENCE_REACHED)
                            return None

    # log information at the end of training
    LOGGER.info("%s training epoch(s) completed, stopping training" % epochs)

    # save exit-code and final processes
    save_exit_code(os.path.join(model_log_directory, "exit_code"),
                   FINISHED_EPOCHS)


def train_outer(args: argparse.Namespace, resume_training=False) -> None:
    # create model log directory
    os.makedirs(args.model_log_directory, exist_ok=True)

    # execute code while catching any errors
    try:
        # update LOGGER object with file handler
        global LOGGER
        add_file_handler(LOGGER,
                         os.path.join(args.model_log_directory, "session.log"))

        if resume_training:
            try:
                args = parse_configs_to_args(args)
                exit_code_file = os.path.join(args.model_log_directory,
                                              "exit_code")
                if not os.path.exists(exit_code_file):
                    LOGGER.info(
                        "Exit-code file not found, continuing training")
                else:
                    exit_code = get_exit_code(exit_code_file)
                    if exit_code == 0:
                        LOGGER.info(
                            ("Exit-code 0: training epochs have already "
                             "been reached"))
                        return None
                    elif exit_code == 1:
                        LOGGER.info(
                            ("Exit-code 1: patience threshold has already "
                             "been reached"))
                        return None
                    elif exit_code == 2:
                        LOGGER.info(
                            ("Exit-code 2: interruption during previous "
                             "training, continuing training"))
            except FileNotFoundError:
                if args.grid_training:
                    resume_training = False
                else:
                    raise

        # log namespace arguments and model directory
        LOGGER.info(args)
        LOGGER.info("Model log directory: %s" % args.model_log_directory)

        # set gpu and cpu hardware
        gpu_device = set_hardware(args)

        # get relevant patterns
        pattern_specs = get_pattern_specs(args)

        # set initial random seeds
        set_random_seed(args)

        # get input vocab
        vocab_combined = get_vocab(args)

        # get final vocab, embeddings and word_dim
        vocab, embeddings, word_dim = get_embeddings(args, vocab_combined)

        # add word_dim into arguments
        args.word_dim = word_dim

        # show vocabulary diagnostics
        get_vocab_diagnostics(vocab, vocab_combined, word_dim)

        # get embeddings as torch Module
        embeddings = Embedding.from_pretrained(embeddings,
                                               freeze=args.static_embeddings,
                                               padding_idx=PAD_TOKEN_INDEX)

        # get training and validation data
        _, _, train_data, valid_data, num_classes = get_train_valid_data(
            args, vocab)

        # get semiring
        semiring = get_semiring(args)

        # create SoftPatternClassifier
        model = SoftPatternClassifier(
            pattern_specs,
            num_classes,
            embeddings,  # type:ignore
            vocab,
            semiring,
            args.tau_threshold,
            args.no_wildcards,
            args.bias_scale,
            args.wildcard_scale,
            args.dropout)

        # log information about model
        LOGGER.info("Model: %s" % model)

        if not resume_training:
            # print model diagnostics and dump files
            LOGGER.info("Total model parameters: %s" %
                        sum(parameter.nelement()
                            for parameter in model.parameters()))
            dump_configs(args, args.model_log_directory)
            vocab.dump(args.model_log_directory)

        train_inner(train_data, valid_data, model, num_classes, args.epochs,
                    args.evaluation_period, args.only_epoch_eval,
                    args.model_log_directory, args.learning_rate,
                    args.batch_size, args.disable_scheduler,
                    args.scheduler_patience, args.scheduler_factor, gpu_device,
                    args.clip_threshold, args.max_doc_len, args.word_dropout,
                    args.patience, resume_training, args.disable_tqdm,
                    args.tqdm_update_period)
    finally:
        # update LOGGER object to remove file handler
        remove_all_file_handlers(LOGGER)


def main(args: argparse.Namespace) -> None:
    # depending on training type, create appropriate argparse namespaces
    if args.grid_training:
        # redefine models log directory
        args.models_directory = os.path.join(args.models_directory,
                                             "spp_grid_train_" + timestamp())
        os.makedirs(args.models_directory, exist_ok=True)

        # dump current training configs
        dump_configs(args, args.models_directory, "base_")

        # get grid config and add random iterations to it
        grid_dict = get_grid_config(args)

        # add random seed into grid if necessary
        if args.num_random_iterations > 1:
            seed = list(range(0, args.num_random_iterations))
            grid_dict["seed"] = seed

        # dump parameter grid to file for re-use
        param_grid_mapping = {
            i: param_grid_instance
            for i, param_grid_instance in enumerate(ParameterGrid(grid_dict))
        }
        with open(os.path.join(args.models_directory, "grid_config.json"),
                  "w") as output_file_stream:
            json.dump(param_grid_mapping,
                      output_file_stream,
                      ensure_ascii=False)

        # process new args superset
        args_superset = get_grid_args_superset(args, param_grid_mapping)
    else:
        # make trivial superset
        args_superset = [args]

    # loop and train
    for i, args in enumerate(args_superset):
        args.model_log_directory = os.path.join(
            args.models_directory, "spp_single_train_" +
            timestamp() if not args.grid_training else "spp_single_train_" +
            str(i))
        train_outer(args, resume_training=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter,
                                     parents=[
                                         spp_arg_parser(),
                                         train_arg_parser(),
                                         grid_train_arg_parser(),
                                         hardware_arg_parser(),
                                         logging_arg_parser(),
                                         tqdm_arg_parser()
                                     ])
    LOGGER = stdout_root_logger(
        logging_arg_parser().parse_known_args()[0].logging_level)
    main(parser.parse_args())
