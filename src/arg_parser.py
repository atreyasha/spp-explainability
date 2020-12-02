#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from .soft_patterns_pp import (SHARED_SL_PARAM_PER_STATE_PER_PATTERN,
                               SHARED_SL_SINGLE_PARAM)


def soft_patterns_pp_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    # add sopa group
    sopa = parser.add_argument_group('optional sopa-architecture arguments')
    # numeric and character-accepting options
    sopa.add_argument(
        "--patterns",
        help=
        "Pattern lengths and numbers: an underscore separated list of length-number pairs",
        default="5-50_4-50_3-50_2-50",
        type=str)
    sopa.add_argument("--bias-scale-param",
                      help="Scale bias term by this parameter",
                      default=0.1,
                      type=float)
    sopa.add_argument("--eps-scale",
                      help="Scale epsilon by this parameter",
                      type=float)
    sopa.add_argument("--self-loop-scale",
                      help="Scale self_loop by this parameter",
                      type=float)
    sopa.add_argument(
        "--shared-sl",
        help=
        "Share main path and self loop parameters, where self loops are discounted by a self_loop_parameter. "
        + str(SHARED_SL_PARAM_PER_STATE_PER_PATTERN) +
        ": one parameter per state per pattern, " +
        str(SHARED_SL_SINGLE_PARAM) + ": a global parameter",
        default=0,
        type=int)
    sopa.add_argument("--mlp-hidden-dim",
                      help="MLP hidden dimension",
                      default=25,
                      type=int)
    sopa.add_argument("--num-mlp-layers",
                      help="Number of MLP layers",
                      default=2,
                      type=int)
    # boolean flags
    sopa.add_argument("--maxplus",
                      help="Use max-plus semiring instead of plus-times",
                      action='store_true')
    sopa.add_argument("--maxtimes",
                      help="Use max-times semiring instead of plus-times",
                      action='store_true')
    sopa.add_argument("--no-sl",
                      help="Don't use self loops",
                      action='store_true')
    sopa.add_argument("--no-eps",
                      help="Don't use epsilon transitions",
                      action='store_true')
    sopa.add_argument("--use-rnn",
                      help="Use an RNN underneath soft-patterns",
                      action="store_true")
    return parser


def training_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    # add required group
    required = parser.add_argument_group('required arguments')
    required.add_argument("--td",
                          help="Train data file",
                          required=True,
                          type=str)
    required.add_argument("--tl",
                          help="Train labels file",
                          required=True,
                          type=str)
    required.add_argument("--vd",
                          help="Validation data file",
                          required=True,
                          type=str)
    required.add_argument("--vl",
                          help="Validation labels file",
                          required=True,
                          type=str)
    required.add_argument("--embedding-file",
                          help="Word embedding file",
                          required=True,
                          type=str)
    # add train group for clearer annotations
    train = parser.add_argument_group('optional training arguments')
    # numeric and character-accepting options
    train.add_argument("--input-model",
                       help="Input model (for testing, not training)",
                       type=str)
    train.add_argument("--model-save-dir",
                       help="where to save the trained model",
                       type=str)
    train.add_argument("--pre-computed-patterns",
                       help="File containing pre-computed patterns",
                       type=str)
    train.add_argument("--learning-rate",
                       help="Adam Learning rate",
                       default=1e-3,
                       type=float)
    train.add_argument("--word-dropout",
                       help="Use word dropout",
                       default=0,
                       type=float)
    train.add_argument("--clip", help="Gradient clipping", type=float)
    train.add_argument("--dropout", help="Use dropout", default=0, type=float)
    train.add_argument("--batch-size", help="Batch size", default=1, type=int)
    train.add_argument(
        "--max-doc-len",
        help=
        "Maximum doc length. For longer documents, spans of length max_doc_len will be randomly "
        "selected each iteration (-1 means no restriction)",
        default=-1,
        type=int)
    train.add_argument("--seed", help="Random seed", default=100, type=int)
    train.add_argument("--num-train-instances",
                       help="Number of training instances",
                       type=int)
    train.add_argument("--num-iterations",
                       help="Number of iterations",
                       default=10,
                       type=int)
    train.add_argument("--patience",
                       help="Patience parameter (for early stopping)",
                       default=30,
                       type=int)
    train.add_argument("--debug", help="Debug", default=0, type=int)
    # boolean flags
    train.add_argument("--scheduler",
                       help="Use reduce learning rate on plateau schedule",
                       action='store_true')
    train.add_argument("--gpu", help="Use GPU", action='store_true')
    return parser


def preprocess_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    # add preprocess group
    preprocess = parser.add_argument_group('optional preprocessing arguments')
    preprocess.add_argument(
        "--data-directory",
        help="Root directory containing facebook multi-class NLU data",
        default="./data/facebook_multiclass_nlu/",
        type=str)
    return parser


def logging_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    # add preprocess group
    logging = parser.add_argument_group('optional logging arguments')
    logging.add_argument(
        "--logging-level",
        help="Set logging level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        type=str)
    return parser
