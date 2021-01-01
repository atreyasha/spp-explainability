#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from glob import glob
from .utils.data_utils import Vocab, PAD_TOKEN_INDEX
from .utils.parser_utils import ArgparseFormatter
from .utils.logging_utils import (stdout_root_logger, add_file_handler,
                                  remove_all_file_handlers)
from .arg_parser import (logging_arg_parser, tqdm_arg_parser,
                         hardware_arg_parser, resume_training_arg_parser)
from .soft_patterns_pp import SoftPatternClassifier
from .train_spp import (set_hardware, get_patterns, set_random_seed,
                        get_training_validation_data, get_semiring, train)
import argparse
import torch
import json
import os


def parse_configs_to_args(args: argparse.Namespace,
                          model_log_directory: str) -> argparse.Namespace:
    # check for json configs
    json_files = glob(os.path.join(model_log_directory, "*_config.json"))
    assert len(json_files) == 2, ("Number of json configuration files "
                                  "present in %s not equal to two" %
                                  model_log_directory)

    # update argument namespace with information from json files
    for json_file in json_files:
        with open(json_file, "r") as input_file_stream:
            args.__dict__.update(json.load(input_file_stream))
    return args


def get_exit_code(filename: str) -> int:
    with open(filename, "r") as input_file_stream:
        exit_code = int(input_file_stream.readline().strip())
    return exit_code


def main(args: argparse.Namespace) -> None:
    # create model log directory
    model_log_directory = args.model_log_directory

    # check for existence of model_log_directory
    if not os.path.exists(model_log_directory):
        raise FileNotFoundError("%s does not exist" % model_log_directory)

    # update LOGGER object with file handler
    global LOGGER
    LOGGER = add_file_handler(LOGGER,
                              os.path.join(model_log_directory, "session.log"))

    # get exit code of model log directory
    exit_code_file = os.path.join(model_log_directory, "exit_code")
    if not os.path.exists(exit_code_file):
        LOGGER.info("Exit-code file not found, continuing training")
    else:
        exit_code = get_exit_code(exit_code_file)
        if exit_code == 0:
            LOGGER.info(
                "Exit-code 0: training epochs have already been reached")
            return None
        elif exit_code == 1:
            LOGGER.info(
                "Exit-code 1: patience epochs have already been reached")
            return None
        elif exit_code == 2:
            LOGGER.info(("Exit-code 2: interruption during previous training, "
                         "continuing training"))

    # load configurations directly into argument namespace
    args = parse_configs_to_args(args, model_log_directory)

    # log namespace arguments
    LOGGER.info(args)

    # set gpu and cpu hardware
    gpu_device = set_hardware(args)

    # read important arguments and define as local variables
    num_train_instances = args.num_train_instances
    mlp_hidden_dim = args.mlp_hidden_dim
    mlp_num_layers = args.mlp_num_layers
    epochs = args.epochs

    # get relevant patterns
    pattern_specs, pre_computed_patterns = get_patterns(args)

    # set initial random seeds, which will be updated again later
    set_random_seed(args)

    # read vocab from file
    vocab = Vocab.from_vocab_file(
        os.path.join(model_log_directory, "vocab.txt"))

    # generate embeddings to fill up correct dimensions
    embeddings = torch.zeros(len(vocab), args.word_dim)
    embeddings = torch.nn.Embedding.from_pretrained(
        embeddings, freeze=args.static_embeddings,
        padding_idx=PAD_TOKEN_INDEX)

    # get training and validation data
    train_data, valid_data, num_classes = get_training_validation_data(
        args, pattern_specs, vocab, num_train_instances)

    # get semiring
    semiring = get_semiring(args)

    # create SoftPatternClassifier
    model = SoftPatternClassifier(pattern_specs, mlp_hidden_dim,
                                  mlp_num_layers, num_classes, embeddings,
                                  vocab, semiring, args.bias_scale,
                                  pre_computed_patterns, args.no_self_loops,
                                  args.shared_self_loops, args.no_epsilons,
                                  args.epsilon_scale, args.self_loop_scale,
                                  args.dropout)

    # train SoftPatternClassifier
    train(train_data, valid_data, model, num_classes, epochs,
          model_log_directory, args.learning_rate, args.batch_size,
          args.disable_scheduler, args.scheduler_patience,
          args.scheduler_factor, gpu_device, args.clip_threshold,
          args.max_doc_len, args.word_dropout, args.patience, True,
          args.disable_tqdm, args.tqdm_update_freq)

    # update LOGGER object to remove file handler
    LOGGER = remove_all_file_handlers(LOGGER)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter,
                                     parents=[
                                         resume_training_arg_parser(),
                                         hardware_arg_parser(),
                                         logging_arg_parser(),
                                         tqdm_arg_parser()
                                     ])
    args = parser.parse_args()
    LOGGER = stdout_root_logger(args.logging_level)
    main(args)
