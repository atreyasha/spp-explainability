#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from glob import glob
from functools import partial
from sklearn.metrics import classification_report
from typing import cast, List, Tuple, Optional
from torch.nn import Embedding, Module
from .utils.parser_utils import ArgparseFormatter
from .utils.logging_utils import stdout_root_logger
from .utils.data_utils import Vocab, PAD_TOKEN_INDEX, read_docs, read_labels
from .arg_parser import (logging_arg_parser, hardware_arg_parser,
                         evaluate_arg_parser, grid_evaluate_arg_parser)
from .train_spp import (parse_configs_to_args, set_hardware, get_pattern_specs,
                        get_semiring, evaluate_metric)
from .torch_module_spp import SoftPatternClassifier
import argparse
import torch
import json
import os


def evaluate_inner(eval_data: List[Tuple[List[int], int]],
                   model: Module,
                   model_checkpoint: str,
                   model_log_directory: str,
                   batch_size: int,
                   output_prefix: str,
                   gpu_device: Optional[torch.device] = None,
                   max_doc_len: Optional[int] = None) -> dict:
    # load model checkpoint
    model_checkpoint_loaded = torch.load(model_checkpoint,
                                         map_location=torch.device("cpu"))
    model.load_state_dict(
        model_checkpoint_loaded["model_state_dict"])  # type: ignore

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

    # designate filename
    filename = os.path.join(
        model_log_directory,
        os.path.basename(model_checkpoint).replace(".pt", "_") +
        output_prefix + "_classification_report.json")

    # dump json report in model_log_directory
    LOGGER.info("Writing classification report: %s" % filename)
    with open(filename, "w") as output_file_stream:
        json.dump(clf_report, output_file_stream)

    # return classification report
    return clf_report


def evaluate_outer(args: argparse.Namespace) -> dict:
    # log namespace arguments and model directory
    LOGGER.info(args)
    LOGGER.info("Model log directory: %s" % args.model_log_directory)

    # set gpu and cpu hardware
    gpu_device = set_hardware(args)

    # get relevant patterns
    pattern_specs = get_pattern_specs(args)

    # load vocab and embeddings
    vocab_file = os.path.join(args.model_log_directory, "vocab.txt")
    if os.path.exists(vocab_file):
        vocab = Vocab.from_vocab_file(
            os.path.join(args.model_log_directory, "vocab.txt"))
    else:
        raise FileNotFoundError("File not found: %s" % vocab_file)

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
        args.tau_threshold,
        args.no_wildcards,
        args.bias_scale,
        args.wildcard_scale,
        0.)

    # log information about model
    LOGGER.info("Model: %s" % model)

    # execute inner evaluation workflow
    clf_report = evaluate_inner(eval_data, model, args.model_checkpoint,
                                args.model_log_directory, args.batch_size,
                                args.output_prefix, gpu_device,
                                args.max_doc_len)
    return clf_report


def main(args: argparse.Namespace) -> None:
    # collect all checkpoints
    model_checkpoint_collection = glob(args.model_checkpoint)
    evaluation_metric_collection = []

    # infer and assume grid directory if provided
    if args.grid_evaluation:
        model_checkpoint_grid_directories = [
            os.path.dirname(os.path.dirname(model_checkpoint))
            for model_checkpoint in model_checkpoint_collection
        ]
        assert len(set(model_checkpoint_grid_directories)) == 1, (
            "Checkpoints provided cannot be processed with " +
            "--grid-evaluation because it corresponds to more than " +
            "one grid directory")
        model_grid_directory = model_checkpoint_grid_directories[0]

    # loop over all provided models
    for model_checkpoint in model_checkpoint_collection:
        args.model_checkpoint = model_checkpoint
        args.model_log_directory = os.path.dirname(args.model_checkpoint)
        args = parse_configs_to_args(args, training=False)
        clf_report = evaluate_outer(args)
        evaluation_metric_collection.append({model_checkpoint: clf_report})

    if args.grid_evaluation:
        # find best clf report by checking all entries
        # source: https://stackoverflow.com/a/30546905
        best_clf_report = max(
            evaluation_metric_collection,
            key=lambda dictionary: next(iter(dictionary.values()))[
                args.evaluation_metric_type][args.evaluation_metric]
            if args.evaluation_metric != "accuracy" else next(
                iter(dictionary.values()))[args.evaluation_metric])

        # add additional evaluation information
        best_clf_report["evaluation_metric"] = args.evaluation_metric
        if args.evaluation_metric != "accuracy":
            best_clf_report[
                "evaluation_metric_type"] = args.evaluation_metric_type

        # dump json report in grid_directory
        with open(
                os.path.join(
                    model_grid_directory, "spp_best_" + args.output_prefix +
                    "_classification_report.json"), "w") as output_file_stream:
            json.dump(best_clf_report, output_file_stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter,
                                     parents=[
                                         evaluate_arg_parser(),
                                         grid_evaluate_arg_parser(),
                                         hardware_arg_parser(),
                                         logging_arg_parser()
                                     ])
    LOGGER = stdout_root_logger(
        logging_arg_parser().parse_known_args()[0].logging_level)
    main(parser.parse_args())
