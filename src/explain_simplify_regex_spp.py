#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
from collections import OrderedDict
from typing import List, Tuple, Union, Dict, cast
from torch.nn import Embedding, Module
from .utils.parser_utils import ArgparseFormatter
from .utils.logging_utils import stdout_root_logger
from .utils.data_utils import unique, PAD_TOKEN_INDEX, Vocab
from .utils.model_utils import to_cuda, chunked, Batch
from .arg_parser import (explain_simplify_arg_parser, hardware_arg_parser,
                         logging_arg_parser, tqdm_arg_parser)
from .train_spp import (parse_configs_to_args, set_hardware, get_semiring,
                        get_train_valid_data, get_pattern_specs)
from .torch_model_spp import SoftPatternClassifier
import argparse
import torch
import os


def save_regex_model(pattern_specs: 'OrderedDict[int, int]',
                     activating_regex: Dict[int, List[str]],
                     linear_state_dict: 'OrderedDict', filename: str) -> None:
    torch.save(
        {
            "pattern_specs": pattern_specs,
            "activating_regex": activating_regex,
            "linear_state_dict": linear_state_dict
        }, filename)


def explain_inner(explain_data: List[Tuple[List[int], int]],
                  explain_text: List[List[str]],
                  model: Module,
                  model_checkpoint: str,
                  model_log_directory: str,
                  batch_size: int,
                  atol: float,
                  gpu_device: Union[torch.device, None],
                  max_doc_len: Union[int, None] = None,
                  disable_tqdm: bool = False) -> None:
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

    # sort explain_data and explain_text for batch processing
    explain_data, explain_text = map(
        list,
        zip(*sorted(zip(explain_data, explain_text),
                    key=lambda v: len(v[0][0]),
                    reverse=True)))
    explain_data = cast(List[Tuple[List[int], int]], explain_data)

    # start explanation workflow on all explain_data
    LOGGER.info("Retrieving activating spans and back pointers")
    activating_spans_back_pointers = [
        [
            back_pointer if back_pointer.binarized_score else None
            for back_pointer in back_pointers
        ] for explain_batch in tqdm(chunked(explain_data, batch_size),
                                    disable=disable_tqdm)
        for back_pointers in model.forward_with_trace(  # type: ignore
            Batch(
                [doc for doc, _ in explain_batch],
                model.embeddings,  # type: ignore
                to_cuda(gpu_device),
                0.,
                max_doc_len),
            atol)[0]
    ]

    # reprocess back pointers by pattern
    LOGGER.info("Grouping activating spans by patterns")
    activating_spans_back_pointers = {
        pattern_index: sorted(
            [[
                explain_text[explain_data_index],
                activating_spans_back_pointers[explain_data_index]
                [pattern_index]
            ] for explain_data_index in range(len(explain_data))
             if activating_spans_back_pointers[explain_data_index]
             [pattern_index]],
            key=lambda mixed: mixed[1].raw_score,  # type: ignore
            reverse=True)
        for pattern_index in range(model.total_num_patterns)  # type: ignore
    }

    # extract segments of text leading to activations
    LOGGER.info("Converting activating spans to regex")
    activating_regex = {
        pattern_index: [
            back_pointer_with_text[1].get_regex(  # type: ignore
                back_pointer_with_text[0])
            for back_pointer_with_text in back_pointers_with_text
        ]
        for pattern_index, back_pointers_with_text in
        activating_spans_back_pointers.items()
    }

    # make all spans unqiue
    LOGGER.info("Making activating regex unique")
    activating_regex = {
        pattern_index: [
            list(text_tuple_inner) for text_tuple_inner in unique([
                tuple(text_list_inner) for text_list_inner in text_list_outer
            ])
        ]
        for pattern_index, text_list_outer in activating_regex.items()
    }

    # produce sample text for the user to peruse
    LOGGER.info("Sample activating text: %s" % activating_regex[0])

    # convert regex list to regex strings
    LOGGER.info("Converting regex lists to regex strings")
    activating_regex = {
        pattern_index: [
            "\\b" + " ".join(activating_regex_pattern_instance) + "\\b"
            for activating_regex_pattern_instance in activating_regex_pattern
        ]
        for pattern_index, activating_regex_pattern in activating_regex.items()
    }

    # define model filename
    model_filename = os.path.join(
        model_log_directory, "regex_" + os.path.basename(model_checkpoint))

    # save regular expression ensemble
    LOGGER.info("Saving regular expression ensemble to disk: %s" %
                model_filename)
    save_regex_model(
        model.pattern_specs,  # type: ignore
        activating_regex,
        model.linear.state_dict(),  # type: ignore
        model_filename)


def explain_outer(args: argparse.Namespace) -> None:
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
        raise FileNotFoundError("%s is missing" % vocab_file)

    # generate embeddings to fill up correct dimensions
    embeddings = torch.zeros(len(vocab), args.word_dim)
    embeddings = Embedding.from_pretrained(embeddings,
                                           freeze=args.static_embeddings,
                                           padding_idx=PAD_TOKEN_INDEX)

    # get training and validation data
    (train_text, valid_text, train_data, valid_data,
     num_classes) = get_train_valid_data(args, vocab)
    explain_text = train_text + valid_text
    explain_data = train_data + valid_data

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

    # execute inner function here
    explain_inner(explain_data, explain_text, model, args.model_checkpoint,
                  args.model_log_directory, args.batch_size, args.atol,
                  gpu_device, args.max_doc_len, args.disable_tqdm)


def main(args: argparse.Namespace) -> None:
    args.model_log_directory = os.path.dirname(args.model_checkpoint)
    args = parse_configs_to_args(args, training=False)
    explain_outer(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter,
                                     parents=[
                                         explain_simplify_arg_parser(),
                                         hardware_arg_parser(),
                                         logging_arg_parser(),
                                         tqdm_arg_parser()
                                     ])
    LOGGER = stdout_root_logger(
        logging_arg_parser().parse_known_args()[0].logging_level)
    main(parser.parse_args())
