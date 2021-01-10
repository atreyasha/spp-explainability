#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .utils.parser_utils import dir_path, file_path, glob_path
from .soft_patterns_pp import (SHARED_SL_PARAM_PER_STATE_PER_PATTERN,
                               SHARED_SL_SINGLE_PARAM)
import argparse


def soft_patterns_pp_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    # add sopa group
    sopa = parser.add_argument_group('optional sopa-architecture arguments')
    # numeric and character-accepting options
    sopa.add_argument(
        "--patterns",
        help=("Pattern lengths and counts with the following syntax: " +
              "PatternLength1-PatternCount1_PatternLength2-PatternCount2_..."),
        default="7-25_6-25_5-25_4-25_3-25_2-25",
        type=str)
    sopa.add_argument("--semiring",
                      help="Specify which semiring to use",
                      default="MaxSumSemiring",
                      choices=[
                          "MaxSumSemiring", "MaxProductSemiring",
                          "ProbabilitySemiring"
                      ],
                      type=str)
    sopa.add_argument("--bias-scale",
                      help="Scale biases by this parameter",
                      type=float)
    sopa.add_argument("--epsilon-scale",
                      help="Scale epsilons by this parameter",
                      type=float)
    sopa.add_argument("--self-loop-scale",
                      help="Scale self-loops by this parameter",
                      type=float)
    sopa.add_argument(
        "--shared-self-loops",
        help=("Option to share main path and self loop parameters. " +
              "0: do not share parameters, " +
              str(SHARED_SL_PARAM_PER_STATE_PER_PATTERN) +
              ": share one parameter per state per pattern, " +
              str(SHARED_SL_SINGLE_PARAM) + ": share one global parameter"),
        default=0,
        choices=[
            0, SHARED_SL_PARAM_PER_STATE_PER_PATTERN, SHARED_SL_SINGLE_PARAM
        ],
        type=int)
    sopa.add_argument("--word-dim", help=argparse.SUPPRESS, type=int)
    # boolean flags
    sopa.add_argument("--static-embeddings",
                      help="Freeze learning of token embeddings",
                      action='store_true')
    sopa.add_argument("--no-self-loops",
                      help="Do not use self loops",
                      action='store_true')
    sopa.add_argument("--no-epsilons",
                      help="Do not use epsilon transitions",
                      action='store_true')
    return parser


def training_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    # add required group
    required = parser.add_argument_group('required training arguments')
    required.add_argument("--embeddings",
                          help="Path to GloVe token embeddings file",
                          required=True,
                          type=file_path)
    required.add_argument("--train-data",
                          help="Path to train data file",
                          required=True,
                          type=file_path)
    required.add_argument("--train-labels",
                          help="Path to train labels file",
                          required=True,
                          type=file_path)
    required.add_argument("--valid-data",
                          help="Path to validation data file",
                          required=True,
                          type=file_path)
    required.add_argument("--valid-labels",
                          help="Path to validation labels file",
                          required=True,
                          type=file_path)
    # add train group for clearer annotations
    train = parser.add_argument_group('optional training arguments')
    # numeric and character-accepting options
    train.add_argument("--models-directory",
                       help="Base directory where all models will be saved",
                       default="./models",
                       type=dir_path)
    train.add_argument("--pre-computed-patterns",
                       help="Path to file containing per-computed patterns",
                       type=file_path)
    train.add_argument("--learning-rate",
                       help="Learning rate for Adam optimizer",
                       default=1e-3,
                       type=float)
    train.add_argument("--word-dropout",
                       help="Word dropout probability",
                       default=0.2,
                       type=float)
    train.add_argument("--clip-threshold",
                       help="Gradient clipping threshold",
                       type=float)
    train.add_argument("--dropout",
                       help="Neuron dropout probability",
                       default=0.2,
                       type=float)
    train.add_argument(
        "--scheduler-factor",
        help="Factor by which the learning rate will be reduced",
        default=0.1,
        type=float)
    train.add_argument("--batch-size",
                       help="Batch size for training",
                       default=256,
                       type=int)
    train.add_argument("--max-doc-len",
                       help=("Maximum document length allowed. "
                             "-1 refers to no length restriction"),
                       default=-1,
                       type=int)
    train.add_argument("--seed",
                       help="Global random seed for numpy and torch",
                       default=42,
                       type=int)
    train.add_argument("--num-train-instances",
                       help="Maximum number of training instances",
                       type=int)
    train.add_argument("--epochs",
                       help="Maximum number of training epochs",
                       default=50,
                       type=int)
    train.add_argument("--patience",
                       help=("Number of epochs with no improvement after "
                             "which training will be stopped"),
                       default=10,
                       type=int)
    train.add_argument("--scheduler-patience",
                       help=("Number of epochs with no improvement after "
                             "which learning rate will be reduced"),
                       default=5,
                       type=int)
    # boolean flags
    train.add_argument("--disable-scheduler",
                       help=("Disable learning rate scheduler which reduces "
                             "learning rate on performance plateau"),
                       action='store_true')
    return parser


def evaluation_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    # add required group
    required = parser.add_argument_group('required evaluation arguments')
    required.add_argument("--eval-data",
                          help="Path to evaluation data file",
                          required=True,
                          type=file_path)
    required.add_argument("--eval-labels",
                          help="Path to evaluation labels file",
                          required=True,
                          type=file_path)
    required.add_argument("--model-checkpoint",
                          help=("Glob path to model checkpoint with '.pt' "
                                "extension. Note that 'model_config.json' "
                                "must be in the same directory level as the "
                                "model checkpoint file"),
                          required=True,
                          type=glob_path)
    # add evaluate group for optional arguments
    evaluate = parser.add_argument_group('optional evaluation arguments')
    evaluate.add_argument("--output-prefix",
                          help="Prefix for output classification report",
                          default="test",
                          type=str)
    evaluate.add_argument("--batch-size",
                          help="Batch size for training",
                          default=256,
                          type=int)
    return parser


def grid_training_arg_parser(
        resume_training: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    grid = parser.add_argument_group('optional grid-training arguments')
    if not resume_training:
        grid.add_argument(
            "--grid-config",
            help="Path to grid configuration file",
            default="./src/resources/flat_grid_light_config.json",
            type=file_path)
        grid.add_argument(
            "--num-random-iterations",
            help="Number of random iteration(s) for each grid instance",
            default=1,
            type=int)
    grid.add_argument("--grid-training",
                      help="Use grid-training instead of single-training",
                      action="store_true")
    return parser


def resume_training_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    required = parser.add_argument_group('required training arguments')
    required.add_argument("--model-log-directory",
                          help=("Base model directory containing model "
                                "data to be resumed for training"),
                          required=True,
                          type=dir_path)
    return parser


def hardware_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    hardware = parser.add_argument_group(
        'optional hardware-acceleration arguments')
    hardware.add_argument("--gpu-device",
                          help=("GPU device specification in case --gpu option"
                                " is used"),
                          default="cuda:0",
                          type=str)
    hardware.add_argument("--num-threads",
                          help=("Set the number of threads used for intraop "
                                "parallelism on CPU"),
                          type=int)
    hardware.add_argument("--gpu",
                          help="Use GPU hardware acceleration",
                          action='store_true')
    return parser


def preprocess_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    preprocess = parser.add_argument_group('optional preprocessing arguments')
    preprocess.add_argument("--data-directory",
                            help="Data directory containing clean input data",
                            default="./data/facebook_multiclass_nlu/",
                            type=dir_path)
    preprocess.add_argument(
        "--truecase",
        help=("Retain true casing when preprocessing data. "
              "Otherwise data will be lowercased by default"),
        action="store_true")
    preprocess.add_argument(
        "--disable-upsampling",
        help=("Disable upsampling on the train and validation "
              "data sets"),
        action="store_true")
    return parser


def logging_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    logging = parser.add_argument_group('optional logging arguments')
    logging.add_argument(
        "--logging-level",
        help="Set logging level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        type=str)
    return parser


def tqdm_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    tqdm = parser.add_argument_group('optional progress-bar arguments')
    tqdm.add_argument(
        "--tqdm-update-period",
        help=("Specify after how many training updates should "
              "the tqdm progress bar be updated with model diagnostics"),
        default=5,
        type=int)
    tqdm.add_argument("--disable-tqdm",
                      help="Disable tqdm progress bars",
                      action='store_true')
    return parser
