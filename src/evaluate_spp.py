#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from glob import glob
from functools import partial
from .utils.parser_utils import ArgparseFormatter
from .utils.logging_utils import stdout_root_logger
from .utils.data_utils import Vocab, PAD_TOKEN_INDEX, read_docs, read_labels
from .arg_parser import (logging_arg_parser, hardware_arg_parser,
                         evaluation_arg_parser)
from .train_spp import (parse_configs_to_args, set_hardware, get_pattern_specs,
                        get_semiring, evaluate_metric)
from .spp_model import SoftPatternClassifier
from sklearn.metrics import classification_report
from typing import cast, List, Tuple, Union
from torch.nn import Embedding, Module
import argparse
import torch
import json
import os


def evaluate_inner(eval_data: List[Tuple[List[int], int]],
                   model: Module,
                   model_checkpoint: str,
                   model_log_directory: str,
                   num_classes: int,
                   batch_size: int,
                   output_prefix: str,
                   gpu_device: Union[torch.device, None] = None,
                   max_doc_len: Union[int, None] = None) -> None:
    # load model checkpoint
    model_checkpoint = torch.load(model_checkpoint,
                                  map_location=torch.device("cpu"))
    model.load_state_dict(model_checkpoint["model_state_dict"])  # type: ignore

    # send model to correct device
    if gpu_device is not None:
        LOGGER.info("Transferring model to GPU device: %s" % gpu_device)
        model.to(gpu_device)

    # set model on eval mode and disable autograd
    model.eval()
    torch.autograd.set_grad_enabled(False)

    # compute f1 classification report as python dictionary
    clf_report = evaluate_metric(
        model, eval_data, batch_size, gpu_device,
        partial(classification_report, output_dict=True), max_doc_len)

    # dump json report in model_log_directory
    with open(
            os.path.join(model_log_directory,
                         output_prefix + "_classification_report.json"),
            "w") as output_file_stream:
        json.dump(clf_report, output_file_stream)


def evaluate_outer(args: argparse.Namespace) -> None:
    # create local model_log_directory variable
    model_log_directory = args.model_log_directory

    # log namespace arguments and model directory
    LOGGER.info(args)
    LOGGER.info("Model log directory: %s" % model_log_directory)

    # set gpu and cpu hardware
    gpu_device = set_hardware(args)

    # get relevant patterns
    pattern_specs = get_pattern_specs(args)

    # load vocab and embeddings
    vocab_file = os.path.join(model_log_directory, "vocab.txt")
    if os.path.exists(vocab_file):
        vocab = Vocab.from_vocab_file(
            os.path.join(model_log_directory, "vocab.txt"))
    else:
        raise FileNotFoundError("%s is missing" % vocab_file)
    # generate embeddings to fill up correct dimensions
    embeddings = torch.zeros(len(vocab), args.word_dim)
    embeddings = Embedding.from_pretrained(embeddings,
                                           freeze=args.static_embeddings,
                                           padding_idx=PAD_TOKEN_INDEX)

    # load evaluation data here
    eval_input, eval_text = read_docs(args.eval_data, vocab)
    LOGGER.info("Sample evaluation text: %s" % eval_text[:10])
    eval_input = cast(List[List[int]], eval_input)
    eval_labels = read_labels(args.eval_labels)
    num_classes = len(set(eval_labels))
    eval_data = list(zip(eval_input, eval_labels))

    # get semiring
    semiring = get_semiring(args)

    # create SoftPatternClassifier
    model = SoftPatternClassifier(
        pattern_specs,
        num_classes,
        embeddings,  # type:ignore
        vocab,
        semiring,
        args.shared_self_loops,
        args.no_epsilons,
        args.no_self_loops,
        args.bias_scale,
        args.epsilon_scale,
        args.self_loop_scale,
        0.)

    # log information about model
    LOGGER.info("Model: %s" % model)

    # execute inner evaluation workflow
    evaluate_inner(eval_data, model, args.model_checkpoint,
                   model_log_directory, num_classes, args.batch_size,
                   args.output_prefix, gpu_device, args.max_doc_len)


def main(args: argparse.Namespace) -> None:
    for model_checkpoint in glob(args.model_checkpoint):
        args.model_checkpoint = model_checkpoint
        args.model_log_directory = os.path.dirname(args.model_checkpoint)
        args = parse_configs_to_args(args, training=False)
        evaluate_outer(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter,
                                     parents=[
                                         evaluation_arg_parser(),
                                         hardware_arg_parser(),
                                         logging_arg_parser()
                                     ])
    LOGGER = stdout_root_logger(
        logging_arg_parser().parse_known_args()[0].logging_level)
    main(parser.parse_args())
