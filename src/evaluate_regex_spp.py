#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
from glob import glob
from torch.nn import Linear
from typing import cast, List, Union
from sklearn.metrics import classification_report
from .utils.model_utils import chunked
from .utils.parser_utils import ArgparseFormatter
from .utils.logging_utils import stdout_root_logger
from .utils.data_utils import read_docs, read_labels, Vocab
from .arg_parser import (logging_arg_parser, hardware_arg_parser,
                         evaluate_arg_parser, tqdm_arg_parser)
from .torch_model_regex_spp import RegexSoftPatternClassifier
from .train_spp import set_hardware
import argparse
import torch
import json
import os


def evaluate_inner(eval_text: List[str],
                   eval_labels: List[int],
                   model_checkpoint: str,
                   model_log_directory: str,
                   batch_size: int,
                   output_prefix: str,
                   gpu_device: Union[torch.device, None] = None,
                   disable_tqdm: bool = False) -> None:
    # load model checkpoint
    model_checkpoint_loaded = torch.load(model_checkpoint,
                                         map_location=torch.device("cpu"))

    # log current stage
    LOGGER.info("Loading and pre-compiling regex model")

    # load linear submodule
    linear = Linear(
        len(model_checkpoint_loaded["activating_regex"]),
        model_checkpoint_loaded["linear_state_dict"]["weight"].size(0))
    linear.load_state_dict(model_checkpoint_loaded["linear_state_dict"])

    # create model and load respective parameters
    model = RegexSoftPatternClassifier(
        model_checkpoint_loaded["pattern_specs"],
        model_checkpoint_loaded["activating_regex"], linear)

    # log model information
    LOGGER.info("Model: %s" % model)

    # send model to correct device
    if gpu_device is not None:
        LOGGER.info("Transferring model to GPU device: %s" % gpu_device)
        model.to(gpu_device)

    # set model on eval mode and disable autograd
    model.eval()
    torch.autograd.set_grad_enabled(False)

    # loop over data in batches
    predicted = []
    for batch in tqdm(chunked(eval_text, batch_size), disable=disable_tqdm):
        predicted.extend(torch.argmax(model.forward(batch), 1).tolist())

    # process classification report
    clf_report = classification_report(eval_labels,
                                       predicted,
                                       output_dict=True)

    # dump json report in model_log_directory
    with open(
            os.path.join(
                model_log_directory,
                os.path.basename(model_checkpoint).replace(".pt", "_") +
                output_prefix + "_classification_report.json"),
            "w") as output_file_stream:
        json.dump(clf_report, output_file_stream)


def evaluate_outer(args: argparse.Namespace) -> None:
    # log namespace arguments and model directory
    LOGGER.info(args)
    LOGGER.info("Model log directory: %s" % args.model_log_directory)

    # set gpu and cpu hardware
    gpu_device = set_hardware(args)

    # load vocab and embeddings
    vocab_file = os.path.join(args.model_log_directory, "vocab.txt")
    if os.path.exists(vocab_file):
        vocab = Vocab.from_vocab_file(
            os.path.join(args.model_log_directory, "vocab.txt"))
    else:
        raise FileNotFoundError("%s is missing" % vocab_file)

    # load evaluation data here
    _, eval_text = read_docs(args.eval_data, vocab)
    eval_text = cast(List[List[str]], eval_text)
    LOGGER.info("Sample evaluation text: %s" % eval_text[:10])
    eval_labels = read_labels(args.eval_labels)

    # apply maximum document length if necessary
    if args.max_doc_len is not None:
        eval_text = [
            eval_text[:args.max_doc_len]  # type: ignore
            for doc in eval_text
        ]

    # convert eval_text into list of string
    eval_text = [" ".join(doc) for doc in eval_text]

    # execute inner evaluation workflow
    evaluate_inner(eval_text, eval_labels, args.model_checkpoint,
                   args.model_log_directory, args.batch_size,
                   args.output_prefix, gpu_device, args.disable_tqdm)


def main(args: argparse.Namespace) -> None:
    # parse glob to get all paths
    model_checkpoint_collection = glob(args.model_checkpoint)

    # loop over all provided models
    for model_checkpoint in model_checkpoint_collection:
        args.model_checkpoint = model_checkpoint
        args.model_log_directory = os.path.dirname(args.model_checkpoint)
        evaluate_outer(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter,
                                     parents=[
                                         evaluate_arg_parser(),
                                         hardware_arg_parser(),
                                         logging_arg_parser(),
                                         tqdm_arg_parser()
                                     ])
    LOGGER = stdout_root_logger(
        logging_arg_parser().parse_known_args()[0].logging_level)
    main(parser.parse_args())
