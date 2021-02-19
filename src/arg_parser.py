#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .utils.parser_utils import dir_path, file_path
import argparse


def spp_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    sopa = parser.add_argument_group('optional sopa-architecture arguments')
    # numeric and character-accepting options
    sopa.add_argument(
        "--patterns",
        help=("Pattern lengths and counts with the following syntax: " +
              "PatternLength1-PatternCount1_PatternLength2-PatternCount2_..."),
        default="6-50_5-50_4-50_3-50",
        type=str)
    sopa.add_argument("--semiring",
                      help="Specify which semiring to use",
                      default="MaxSumSemiring",
                      choices=["MaxSumSemiring", "MaxProductSemiring"],
                      type=str)
    sopa.add_argument("--tau-threshold",
                      help="Specify value of STE binarizer tau threshold",
                      default=0.,
                      type=float)
    sopa.add_argument("--bias-scale",
                      help="Scale biases by this parameter",
                      default=1.,
                      type=float)
    sopa.add_argument("--wildcard-scale",
                      help="Scale wildcard(s) by this parameter",
                      type=float)
    sopa.add_argument("--word-dim", help=argparse.SUPPRESS, type=int)
    # boolean flags
    sopa.add_argument("--no-wildcards",
                      help="Do not use wildcard transitions",
                      action='store_true')
    sopa.add_argument("--static-embeddings",
                      help="Freeze learning of token embeddings",
                      action='store_true')
    return parser


def train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    # add group for required arguments
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
    # add group for optional arguments
    train = parser.add_argument_group('optional training arguments')
    # numeric and character-accepting options
    train.add_argument("--models-directory",
                       help="Base directory where all models will be saved",
                       default="./models",
                       type=dir_path)
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
    train.add_argument("--evaluation-period",
                       help=("Specify after how many training updates should "
                             "model evaluation(s) be conducted. Evaluation "
                             "will always be conducted at the end of epochs"),
                       default=100,
                       type=int)
    train.add_argument("--seed",
                       help="Global random seed for numpy and torch",
                       default=42,
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
    train.add_argument("--max-doc-len",
                       help="Maximum document length allowed",
                       type=int)
    train.add_argument("--num-train-instances",
                       help="Maximum number of training instances",
                       type=int)
    # boolean flags
    train.add_argument("--disable-scheduler",
                       help=("Disable learning rate scheduler which reduces "
                             "learning rate on performance plateau"),
                       action='store_true')
    train.add_argument("--only-epoch-eval",
                       help=("Only evaluate model at the end of epoch, "
                             "instead of evaluation by updates"),
                       action='store_true')
    return parser


def explain_simplify_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    # add group for required arguments
    required = parser.add_argument_group('required explainability arguments')
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
    required.add_argument("--neural-model-checkpoint",
                          help=("Path to neural model checkpoint with '.pt' "
                                "extension"),
                          required=True,
                          nargs="+",
                          type=file_path)
    # add group for optional arguments
    explain = parser.add_argument_group('optional explainability arguments')
    explain.add_argument("--atol",
                         help=("Specify absolute tolerance when comparing "
                               "equivalences between tensors"),
                         default=1e-6,
                         type=float)
    explain.add_argument("--batch-size",
                         help="Batch size for explainability",
                         default=256,
                         type=int)
    explain.add_argument("--num-train-instances",
                         help="Maximum number of training instances",
                         type=int)
    explain.add_argument("--max-doc-len",
                         help="Maximum document length allowed",
                         type=int)
    return parser


def explain_compress_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    # add group for required arguments
    required = parser.add_argument_group('required explainability arguments')
    required.add_argument("--regex-model-checkpoint",
                          help="Path to regex model with '.pt' extension",
                          required=True,
                          nargs="+",
                          type=file_path)
    return parser


def evaluate_arg_parser(compare: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    # add group for required arguments
    required = parser.add_argument_group('required evaluation arguments')
    required.add_argument("--eval-data",
                          help="Path to evaluation data file",
                          required=True,
                          type=file_path)
    required.add_argument("--eval-labels",
                          help="Path to evaluation labels file",
                          required=True,
                          type=file_path)
    if compare:
        required.add_argument("--model-log-directory",
                              help=("Model log directory which contains "
                                    "both the best neural and compressed "
                                    "regex models"),
                              required=True,
                              nargs="+",
                              type=dir_path)
    else:
        required.add_argument("--model-checkpoint",
                              help=("Path to model checkpoint with '.pt' "
                                    "extension"),
                              required=True,
                              nargs="+",
                              type=file_path)
    # add group for optional arguments
    evaluate = parser.add_argument_group('optional evaluation arguments')
    if compare:
        evaluate.add_argument(
            "--atol",
            help=("Specify absolute tolerance when comparing "
                  "equivalences between tensors"),
            default=1e-6,
            type=float)
    evaluate.add_argument("--output-prefix",
                          help="Prefix for output classification report",
                          default="test",
                          type=str)
    evaluate.add_argument("--batch-size",
                          help="Batch size for evaluation",
                          default=256,
                          type=int)
    evaluate.add_argument("--max-doc-len",
                          help="Maximum document length allowed",
                          type=int)
    return parser


def grid_evaluate_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    grid = parser.add_argument_group('optional grid-evaluation arguments')
    grid.add_argument(
        "--evaluation-metric",
        help="Specify which evaluation metric to use for comparison",
        choices=["recall", "precision", "f1-score", "accuracy"],
        default="f1-score",
        type=str)
    grid.add_argument("--evaluation-metric-type",
                      help="Specify which type of evaluation metric to use",
                      choices=["weighted avg", "macro avg"],
                      default="weighted avg",
                      type=str)
    grid.add_argument(
        "--grid-evaluation",
        help="Use grid-evaluation framework to find/summarize best model",
        action="store_true")
    return parser


def grid_train_arg_parser(
        start_training: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    grid = parser.add_argument_group('optional grid-training arguments')
    if start_training:
        grid.add_argument(
            "--grid-config",
            help="Path to grid configuration file",
            default="./src/resources/flat_grid_heavy_config.json",
            type=file_path)
        grid.add_argument(
            "--num-random-iterations",
            help="Number of random iteration(s) for each grid instance",
            default=10,
            type=int)
    grid.add_argument("--grid-training",
                      help="Use grid-training instead of single-training",
                      action="store_true")
    return parser


def train_resume_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    required = parser.add_argument_group('required training arguments')
    required.add_argument("--model-log-directory",
                          help=("Base model directory containing model "
                                "data to be resumed for training"),
                          required=True,
                          type=dir_path)
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


def hardware_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    hardware = parser.add_argument_group(
        'optional hardware-acceleration arguments')
    hardware.add_argument("--gpu-device",
                          help=("GPU device specification in case --gpu option"
                                " is used"),
                          default="cuda:0",
                          type=str)
    hardware.add_argument("--torch-num-threads",
                          help=("Set the number of threads used for CPU "
                                "intraop parallelism with PyTorch"),
                          type=int)
    hardware.add_argument("--gpu",
                          help="Use GPU hardware acceleration",
                          action='store_true')
    return parser


def logging_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    logging = parser.add_argument_group('optional logging arguments')
    logging.add_argument(
        "--logging-level",
        help="Set logging level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
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
