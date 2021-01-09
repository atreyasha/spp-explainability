#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
from glob import glob
from functools import partial
from collections import OrderedDict
from torch.optim import Adam
from torch.nn import NLLLoss, Module, Embedding
from torch.nn.functional import log_softmax
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from typing import List, Union, Tuple, cast, Any, Callable
from .utils.parser_utils import ArgparseFormatter
from .utils.data_utils import (vocab_from_text, read_labels, read_docs,
                               read_embeddings, Vocab, PAD_TOKEN_INDEX)
from .utils.model_utils import (shuffled_chunked_sorted, chunked_sorted,
                                to_cuda, enable_gradient_clipping, timestamp,
                                Batch, Semiring, ProbabilitySemiring,
                                LogSpaceMaxProductSemiring, MaxSumSemiring)
from .utils.logging_utils import (stdout_root_logger, add_file_handler,
                                  remove_all_file_handlers)
from .arg_parser import (soft_patterns_pp_arg_parser, training_arg_parser,
                         logging_arg_parser, tqdm_arg_parser,
                         hardware_arg_parser, grid_training_arg_parser)
from .soft_patterns_pp import SoftPatternClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
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
PATIENCE_THRESHOLD_BEFORE_EPOCHS = 1
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
                          model_log_directory: str,
                          prefix: str = "",
                          training: bool = True) -> argparse.Namespace:
    # check for json configs and add them to list
    json_files = []
    json_files.append(
        os.path.join(model_log_directory, prefix + "model_config.json"))
    if training:
        json_files.append(
            os.path.join(model_log_directory, prefix + "training_config.json"))

    # raise error if any of them are missing
    for json_file in json_files:
        if not os.path.exists(json_file):
            raise FileNotFoundError("%s is missing" % json_file)

    # update argument namespace with information from json files
    for json_file in json_files:
        with open(json_file, "r") as input_file_stream:
            args.__dict__.update(json.load(input_file_stream))
    return args


def set_hardware(args: argparse.Namespace) -> Union[torch.device, None]:
    # set torch number of threads
    if args.num_threads is None:
        LOGGER.info("Using default number of CPU threads: %s" %
                    torch.get_num_threads())
    else:
        torch.set_num_threads(args.num_threads)
        LOGGER.info("Using specified number of CPU threads: %s" %
                    args.num_threads)

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


def get_grid_config(filename: str) -> dict:
    with open(filename, "r") as input_file_stream:
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


def get_patterns(
    patterns: str, pre_computed_patterns: Union[str, None]
) -> Tuple['OrderedDict[int, int]', Union[List[List[str]], None]]:
    # set default temporary value for pre_computed_patterns
    pre_computed_patterns = None

    # convert pattern_specs string in OrderedDict
    pattern_specs: 'OrderedDict[int, int]' = OrderedDict(
        sorted(
            (
                [int(y) for y in x.split("-")]  # type: ignore
                for x in patterns.split("_")),
            key=lambda t: t[0]))

    # read pre_computed_patterns if it exists, format pattern_specs accordingly
    if pre_computed_patterns is not None:
        pattern_specs, pre_computed_patterns = read_patterns(
            pre_computed_patterns, pattern_specs)
        pattern_specs = OrderedDict(
            sorted(pattern_specs.items(), key=lambda t: t[0]))

    # log diagnositc information on input patterns
    LOGGER.info("Patterns: %s" % pattern_specs)

    # return final objects
    return pattern_specs, pre_computed_patterns


def set_random_seed(args: argparse.Namespace) -> None:
    # set global random seed if specified
    if args.seed != -1:
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


def get_training_validation_data(
    args: argparse.Namespace, pattern_specs: 'OrderedDict[int, int]',
    vocab: Vocab, num_train_instances: int
) -> Tuple[List[Tuple[List[int], int]], List[Tuple[List[int], int]], int]:
    # read train data
    train_input, train_text = read_docs(args.train_data, vocab)
    LOGGER.info("Sample training text: %s" % train_text[:10])
    train_input = cast(List[List[int]], train_input)
    train_labels = read_labels(args.train_labels)
    num_classes = len(set(train_labels))
    train_data = list(zip(train_input, train_labels))

    # read validation data
    valid_input, valid_text = read_docs(args.valid_data, vocab)
    LOGGER.info("Sample validation text: %s" % valid_text[:10])
    valid_input = cast(List[List[int]], valid_input)
    valid_labels = read_labels(args.valid_labels)
    valid_data = list(zip(valid_input, valid_labels))

    # truncate data if necessary
    if num_train_instances is not None:
        train_data = train_data[:num_train_instances]
        valid_data = valid_data[:num_train_instances]

    # log diagnostic information
    LOGGER.info("Number of classes: %s" % num_classes)
    LOGGER.info("Training instances: %s" % len(train_data))
    LOGGER.info("Validation instances: %s" % len(valid_data))

    # return final tuple object
    return train_data, valid_data, num_classes


def get_semiring(args: argparse.Namespace) -> Semiring:
    # define semiring as per argument provided and log
    if args.semiring == "MaxSumSemiring":
        semiring = MaxSumSemiring
    elif args.semiring == "MaxProductSemiring":
        semiring = LogSpaceMaxProductSemiring
    elif args.semiring == "ProbabilitySemiring":
        semiring = ProbabilitySemiring
    LOGGER.info("Semiring: %s" % args.semiring)
    return semiring


def dump_vocab(vocab: Vocab, model_log_directory: str) -> None:
    # dump vocab indices for re-use
    with open(os.path.join(model_log_directory, "vocab.txt"),
              "w") as output_file_stream:
        dict_list = [(key, value) for key, value in vocab.index.items()]
        dict_list = sorted(dict_list, key=lambda x: x[1])
        for item in dict_list:
            output_file_stream.write("%s\n" % item[0])


def dump_configs(args: argparse.Namespace,
                 model_log_directory: str,
                 prefix: str = "") -> None:
    # create dictionaries to fill up
    soft_patterns_args_dict = {}
    training_args_dict = {}

    # extract real arguments and fill up model dictionary
    for action in soft_patterns_pp_arg_parser()._actions:
        soft_patterns_args_dict[action.dest] = getattr(args, action.dest)

    # extract real arguments and fill up training dictionary
    for action in training_arg_parser()._actions:
        training_args_dict[action.dest] = getattr(args, action.dest)

    # dump soft patterns model arguments for posterity
    with open(os.path.join(model_log_directory, prefix + "model_config.json"),
              "w") as output_file_stream:
        json.dump(soft_patterns_args_dict,
                  output_file_stream,
                  ensure_ascii=False)

    # dump training arguments for posterity
    with open(
            os.path.join(model_log_directory, prefix + "training_config.json"),
            "w") as output_file_stream:
        json.dump(training_args_dict, output_file_stream, ensure_ascii=False)


def read_patterns(
    filename: str, pattern_specs: 'OrderedDict[int, int]'
) -> Tuple['OrderedDict[int, int]', List[List[str]]]:
    # create new pattern_specs variable copy
    pattern_specs = pattern_specs.copy()

    # read pre_computed_patterns into a list
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
    return pattern_specs, pre_computed_patterns


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
            best_valid_acc,
            "numpy_random_state":
            np.random.get_state(),
            "torch_random_state":
            torch.random.get_rng_state()
        }, path)


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


def evaluate_metric(model: Module, data: List[Tuple[List[int], int]],
                    batch_size: int, gpu_device: Union[torch.device, None],
                    metric: Callable[[List[int], List[int]], Any]) -> Any:
    # instantiate local storage variable
    predicted = []
    aggregate_gold = []

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
        predicted.extend(torch.argmax(output, 1).tolist())
        aggregate_gold.extend(gold)

    # return output of metric
    return metric(aggregate_gold, predicted)


def train_inner(train_data: List[Tuple[List[int], int]],
                valid_data: List[Tuple[List[int], int]],
                model: Module,
                num_classes: int,
                epochs: int,
                model_log_directory: str,
                learning_rate: float,
                batch_size: int,
                disable_scheduler: bool = False,
                scheduler_patience: int = 10,
                scheduler_factor: float = 0.1,
                gpu_device: Union[torch.device, None] = None,
                clip_threshold: Union[float, None] = None,
                max_doc_len: int = -1,
                word_dropout: float = 0,
                patience: int = 30,
                resume_training: bool = False,
                disable_tqdm: bool = False,
                tqdm_update_freq: int = 1) -> None:
    # create signal handlers in case script receives termination signals
    # adapted from: https://stackoverflow.com/a/31709094
    for specific_signal in [
            signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT
    ]:
        signal.signal(
            specific_signal,
            partial(signal_handler,
                    os.path.join(model_log_directory, "exit_code")))

    # load model checkpoint if training is being resumed
    if resume_training:
        try:
            model_checkpoint = glob(
                os.path.join(model_log_directory, "*last*.pt"))[0]
        except IndexError:
            raise FileNotFoundError("Model log directory present, but model "
                                    "checkpoint with '.pt' "
                                    "extension missing. Our suggestion is to "
                                    "train the model again "
                                    "from scratch")
        model_checkpoint = torch.load(model_checkpoint,
                                      map_location=torch.device("cpu"))
        model.load_state_dict(
            model_checkpoint["model_state_dict"])  # type: ignore
        current_epoch: int = model_checkpoint["epoch"] + 1  # type: ignore
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
            LOGGER.info(
                "%s patience epoch(s) threshold previously reached, exiting" %
                patience)
            # save exit-code and final processes
            save_exit_code(os.path.join(model_log_directory, "exit_code"),
                           PATIENCE_THRESHOLD_BEFORE_EPOCHS)
            return None
    else:
        current_epoch = 0
        best_valid_loss = 100000000.
        best_valid_loss_index = 0
        best_valid_acc = -1.

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
        np.random.set_state(
            model_checkpoint["numpy_random_state"])  # type: ignore
        torch.random.set_rng_state(
            model_checkpoint["torch_random_state"])  # type: ignore

    # loop over epochs
    for epoch in range(current_epoch, epochs):
        # set model on train mode
        model.train()

        # initialize training and valid loss, and hook to stop training
        train_loss = 0.
        valid_loss = 0.
        stop_training = False

        # main training loop
        LOGGER.info("Training SoPa++ model")
        with tqdm(shuffled_chunked_sorted(train_data, batch_size),
                  disable=disable_tqdm,
                  unit="batch",
                  desc="Training [Epoch %s/%s]" %
                  (epoch + 1, epochs)) as tqdm_batches:
            # loop over train batches
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
                if (i + 1) % tqdm_update_freq == 0 or (i +
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
            train_acc = evaluate_metric(model, train_data, batch_size,
                                        gpu_device, accuracy_score)

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
                  disable=disable_tqdm,
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
                    valid_batch_loss = compute_loss(model, batch, num_classes,
                                                    gold, loss_function,
                                                    gpu_device)

                    # add batch loss to valid_loss
                    valid_loss += valid_batch_loss  # type: ignore

                    if (i + 1) % tqdm_update_freq == 0 or (
                            i + 1) == len(tqdm_batches):
                        tqdm_batches.set_postfix(
                            batch_loss=valid_batch_loss.item() / batch.size())

        # compute mean valid loss over epoch and accuracy
        mean_valid_loss = valid_loss / len(valid_data)
        with torch.no_grad():
            valid_acc = evaluate_metric(model, valid_data, batch_size,
                                        gpu_device, accuracy_score)

        # add valid loss data to tensorboard
        writer.add_scalar("loss/valid_loss", mean_valid_loss, epoch)
        writer.add_scalar("accuracy/valid_accuracy", valid_acc, epoch)

        # log out report of current epoch
        LOGGER.info(
            "epoch: {}/{}, mean_train_loss: {:.3f}, train_acc: {:.3f}%, "
            "mean_valid_loss: {:.3f}, valid_acc: {:.3f}%".format(
                epoch + 1, epochs, mean_train_loss, train_acc * 100,
                mean_valid_loss, valid_acc * 100))

        # apply learning rate scheduler after epoch
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
            if best_valid_loss_index == patience:
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
        # and if threshold was reached strictly before the last epoch
        if stop_training:
            if epoch < max(range(epochs)):
                LOGGER.info(
                    "%s patience epoch(s) threshold reached, stopping training"
                    % patience)
                # save exit-code and final processes
                save_exit_code(os.path.join(model_log_directory, "exit_code"),
                               PATIENCE_THRESHOLD_BEFORE_EPOCHS)
                return None

    # log information at the end of training
    LOGGER.info("%s training epoch(s) completed, stopping training" % epochs)

    # save exit-code and final processes
    save_exit_code(os.path.join(model_log_directory, "exit_code"),
                   FINISHED_EPOCHS)


def train_outer(args: argparse.Namespace, resume_training=False) -> None:
    # create model log directory
    model_log_directory = args.model_log_directory
    os.makedirs(model_log_directory, exist_ok=True)

    # execute code while catching any errors
    try:
        # update LOGGER object with file handler
        global LOGGER
        LOGGER = add_file_handler(
            LOGGER, os.path.join(model_log_directory, "session.log"))

        if resume_training:
            args = parse_configs_to_args(args, model_log_directory)
            exit_code_file = os.path.join(model_log_directory, "exit_code")
            if not os.path.exists(exit_code_file):
                LOGGER.info("Exit-code file not found, continuing training")
            else:
                exit_code = get_exit_code(exit_code_file)
                if exit_code == 0:
                    LOGGER.info(("Exit-code 0: training epochs have already "
                                 "been reached"))
                    return None
                elif exit_code == 1:
                    LOGGER.info(("Exit-code 1: patience epochs have already "
                                 "been reached"))
                    return None
                elif exit_code == 2:
                    LOGGER.info(
                        ("Exit-code 2: interruption during previous training, "
                         "continuing training"))

        # log namespace arguments and model directory
        LOGGER.info(args)
        LOGGER.info("Model log directory: %s" % model_log_directory)

        # set gpu and cpu hardware
        gpu_device = set_hardware(args)

        # read important arguments and define as local variables
        num_train_instances = args.num_train_instances
        epochs = args.epochs

        # get relevant patterns
        pattern_specs, pre_computed_patterns = get_patterns(
            args.patterns, args.pre_computed_patterns)

        # set initial random seeds
        set_random_seed(args)

        if resume_training:
            vocab_file = os.path.join(model_log_directory, "vocab.txt")
            if os.path.exists(vocab_file):
                vocab = Vocab.from_vocab_file(
                    os.path.join(model_log_directory, "vocab.txt"))
            else:
                raise FileNotFoundError("%s is missing" % vocab_file)
            # generate embeddings to fill up correct dimensions
            embeddings = torch.zeros(len(vocab), args.word_dim)
            embeddings = Embedding.from_pretrained(
                embeddings,
                freeze=args.static_embeddings,
                padding_idx=PAD_TOKEN_INDEX)
        else:
            # get input vocab
            vocab_combined = get_vocab(args)
            # get final vocab, embeddings and word_dim
            vocab, embeddings, word_dim = get_embeddings(args, vocab_combined)
            # add word_dim into arguments
            args.word_dim = word_dim
            # show vocabulary diagnostics
            get_vocab_diagnostics(vocab, vocab_combined, word_dim)
            # get embeddings as torch Module
            embeddings = Embedding.from_pretrained(
                embeddings,
                freeze=args.static_embeddings,
                padding_idx=PAD_TOKEN_INDEX)

        # get training and validation data
        train_data, valid_data, num_classes = get_training_validation_data(
            args, pattern_specs, vocab, num_train_instances)

        # get semiring
        semiring = get_semiring(args)

        # create SoftPatternClassifier
        model = SoftPatternClassifier(
            pattern_specs,
            num_classes,
            embeddings,  # type:ignore
            vocab,
            semiring,
            pre_computed_patterns,
            args.shared_self_loops,
            args.no_epsilons,
            args.no_self_loops,
            args.bias_scale,
            args.epsilon_scale,
            args.self_loop_scale,
            args.dropout)

        # log information about model
        LOGGER.info("Model: %s" % model)

        if not resume_training:
            # print model diagnostics and dump files
            LOGGER.info("Total model parameters: %s" %
                        sum(parameter.nelement()
                            for parameter in model.parameters()))
            dump_configs(args, model_log_directory)
            dump_vocab(vocab, model_log_directory)

        train_inner(train_data, valid_data, model, num_classes, epochs,
                    model_log_directory, args.learning_rate, args.batch_size,
                    args.disable_scheduler, args.scheduler_patience,
                    args.scheduler_factor, gpu_device, args.clip_threshold,
                    args.max_doc_len, args.word_dropout, args.patience,
                    resume_training, args.disable_tqdm, args.tqdm_update_freq)
    finally:
        # update LOGGER object to remove file handler
        LOGGER = remove_all_file_handlers(LOGGER)


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
        grid_dict = get_grid_config(args.grid_config)

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
                                         training_arg_parser(),
                                         grid_training_arg_parser(),
                                         hardware_arg_parser(),
                                         soft_patterns_pp_arg_parser(),
                                         logging_arg_parser(),
                                         tqdm_arg_parser()
                                     ])
    args = parser.parse_args()
    LOGGER = stdout_root_logger(args.logging_level)
    main(args)
