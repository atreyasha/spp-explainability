#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from .soft_patterns_pp import (SHARED_SL_PARAM_PER_STATE_PER_PATTERN,
                               SHARED_SL_SINGLE_PARAM)


def soft_patterns_pp_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    # add sopa group
    sopa = parser.add_argument_group('optional sopa-architecture arguments')
    # numeric and character-accepting options
    sopa.add_argument(
        "--patterns",
        help=("Pattern lengths and counts with the following syntax: " +
              "PatternLength1-PatternCount1_PatternLength2-PatternCount2_..."),
        default="5-50_4-50_3-50_2-50",
        type=str)
    sopa.add_argument(
        "--semiring",
        help="Specify which semiring to use",
        default="MaxPlusSemiring",
        choices=["MaxPlusSemiring", "MaxTimesSemiring", "ProbSemiring"],
        type=str)
    sopa.add_argument("--bias-scale",
                      help="Scale biases by this parameter",
                      default=0.1,
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
    sopa.add_argument("--mlp-hidden-dim",
                      help="MLP hidden dimension",
                      default=25,
                      type=int)
    sopa.add_argument("--mlp-num-layers",
                      help="Number of MLP layers",
                      default=2,
                      type=int)
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
    required = parser.add_argument_group('required arguments')
    required.add_argument("--embeddings",
                          help="Path to GloVe token embeddings file",
                          required=True,
                          type=str)
    required.add_argument("--train-data",
                          help="Path to train data file",
                          required=True,
                          type=str)
    required.add_argument("--train-labels",
                          help="Path to train labels file",
                          required=True,
                          type=str)
    required.add_argument("--valid-data",
                          help="Path to validation data file",
                          required=True,
                          type=str)
    required.add_argument("--valid-labels",
                          help="Path to validation labels file",
                          required=True,
                          type=str)
    # add train group for clearer annotations
    train = parser.add_argument_group('optional training arguments')
    # numeric and character-accepting options
    train.add_argument("--gpu-device",
                       help=("GPU device specification in case --gpu option"
                             " is used"),
                       default="cuda:0",
                       type=str)
    train.add_argument("--models-log-directory",
                       help=("Directory where all models and tensorboard logs "
                             "will be saved"),
                       default="./models",
                       type=str)
    train.add_argument("--pre-computed-patterns",
                       help="Path to file containing per-computed patterns",
                       type=str)
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
    train.add_argument("--batch-size",
                       help="Batch size for training",
                       default=64,
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
                       default=200,
                       type=int)
    train.add_argument("--patience",
                       help="Patience parameter for early stopping",
                       default=30,
                       type=int)
    train.add_argument("--num-threads",
                       help=("Set the number of threads used for intraop "
                             "parallelism on CPU"),
                       default=None,
                       type=int)
    # boolean flags
    train.add_argument("--disable-scheduler",
                       help=("Disable learning rate scheduler which reduces "
                             "learning rate on performance plateau"),
                       action='store_true')
    train.add_argument("--gpu",
                       help="Use GPU hardware acceleration",
                       action='store_true')
    return parser


def preprocess_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    # add preprocess group
    preprocess = parser.add_argument_group('optional preprocessing arguments')
    preprocess.add_argument("--data-directory",
                            help="Data directory containing clean input data",
                            default="./data/facebook_multiclass_nlu/",
                            type=str)
    preprocess.add_argument(
        "--truecase",
        help=("Retain true casing when preprocessing data. "
              "Otherwise data will be lowercased by default"),
        action="store_true")
    return parser


def logging_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    # add logging group
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
    # add tqdm group
    tqdm = parser.add_argument_group('optional progress-bar arguments')
    tqdm.add_argument(
        "--tqdm-update-freq",
        help=("Specify after how many training updates should "
              "the tqdm progress bar be updated with model diagnostics"),
        default=1,
        type=int)
    tqdm.add_argument("--disable-tqdm",
                      help="Disable tqdm progress bars",
                      action='store_true')
    return parser
