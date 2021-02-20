#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from glob import glob
from tqdm import tqdm
from typing import cast, List, Tuple, Union
from torch.nn import Embedding, Module, Linear
from .utils.model_utils import chunked, to_cuda, Batch
from .utils.parser_utils import ArgparseFormatter
from .utils.logging_utils import stdout_root_logger
from .utils.data_utils import Vocab, PAD_TOKEN_INDEX, read_docs, read_labels
from .arg_parser import (logging_arg_parser, hardware_arg_parser,
                         evaluate_arg_parser, tqdm_arg_parser)
from .train_spp import (parse_configs_to_args, set_hardware, get_pattern_specs,
                        get_semiring)
from .torch_module_spp import SoftPatternClassifier
from .torch_module_regex_spp import RegexSoftPatternClassifier
import argparse
import torch
import json
import os


def compare_inner(eval_data: List[Tuple[List[int], int]],
                  eval_text: List[str],
                  neural_model: Module,
                  neural_model_checkpoint: str,
                  regex_model_checkpoint: str,
                  model_log_directory: str,
                  batch_size: int,
                  atol: float,
                  output_prefix: str,
                  gpu_device: Union[torch.device, None] = None,
                  max_doc_len: Union[int, None] = None,
                  disable_tqdm: bool = False) -> None:
    # load neural model checkpoint
    neural_model_checkpoint_loaded = torch.load(
        neural_model_checkpoint, map_location=torch.device("cpu"))
    neural_model.load_state_dict(
        neural_model_checkpoint_loaded["model_state_dict"])  # type: ignore

    # load regex model checkpoint
    LOGGER.info("Loading and pre-compiling regex model")
    regex_model_checkpoint_loaded = torch.load(
        regex_model_checkpoint, map_location=torch.device("cpu"))

    # load linear submodule
    linear = Linear(
        len(regex_model_checkpoint_loaded["activating_regex"]),
        regex_model_checkpoint_loaded["linear_state_dict"]["weight"].size(0))
    linear.load_state_dict(regex_model_checkpoint_loaded["linear_state_dict"])

    # create model and load respective parameters
    regex_model = RegexSoftPatternClassifier(
        regex_model_checkpoint_loaded["pattern_specs"],
        regex_model_checkpoint_loaded["activating_regex"], linear)

    # log model information
    LOGGER.info("Regex model: %s" % regex_model)

    # send models to correct device
    if gpu_device is not None:
        LOGGER.info("Transferring models to GPU device: %s" % gpu_device)
        neural_model.to(gpu_device)
        regex_model.to(gpu_device)

    # set model on eval mode and disable autograd
    neural_model.eval()
    regex_model.eval()
    torch.autograd.set_grad_enabled(False)

    # create results storage
    results_store = {
        "neural_model": neural_model_checkpoint,
        "regex_model": regex_model_checkpoint,
        "comparisons": {}
    }

    # log current state
    LOGGER.info("Looping over data and text using neural and regex models")

    # loop over evaluation data and text
    for eval_batch in tqdm(chunked(list(zip(eval_data, eval_text)),
                                   batch_size),
                           disable=disable_tqdm):
        # separate data and text for processing
        eval_batch_data, eval_batch_text = map(list, zip(*eval_batch))
        eval_batch_labels = [
            label  # type: ignore
            for _, label in eval_batch_data
        ]

        # proceed with neural model processsing
        neural_forward_trace_output = neural_model.forward_with_trace(  # type: ignore
            Batch(
                [doc for doc, _ in eval_batch_data],  # type: ignore
                neural_model.embeddings,  # type: ignore
                to_cuda(gpu_device),
                0.,
                max_doc_len),
            atol)

        # proceed with regex model processing
        regex_forward_trace_output = regex_model.forward_with_trace(
            eval_batch_text)  # type: ignore

        # loop/process outputs and add them to results storage
        for (eval_sample_text, eval_sample_label, back_pointers,
             neural_linear_output, regex_lookup, regex_linear_output) in zip(
                 *((eval_batch_text, ) + (eval_batch_labels, ) +
                   neural_forward_trace_output + regex_forward_trace_output)):
            # assign current key to update results storage
            if results_store["comparisons"] == {}:
                current_key = 0
            else:
                current_key = max(map(
                    int,
                    results_store["comparisons"].keys())) + 1  # type: ignore

            # create local storage which will be updated in loop
            local_store = results_store["comparisons"][  # type: ignore
                current_key] = {}

            # add text related data
            local_store["text"] = eval_sample_text
            local_store["gold_label"] = eval_sample_label

            # add neural model diagnostics
            neural_local_store = local_store["neural_model"] = {}
            neural_local_store["activating_text"] = [
                back_pointer.get_text(eval_sample_text.split())
                if back_pointer.binarized_score else None
                for back_pointer in back_pointers
            ]
            neural_local_store["binaries"] = [
                int(back_pointer.binarized_score)
                for back_pointer in back_pointers
            ]
            neural_local_store["softmax"] = torch.softmax(
                neural_linear_output, 0).tolist()
            neural_local_store["predicted_label"] = torch.argmax(
                neural_linear_output, 0).item()

            # add regex model diagnostics
            regex_local_store = local_store["regex_model"] = {}
            regex_local_store["activating_text"] = [
                regex_match.group(2) if regex_match is not None else None
                for regex_match in regex_lookup
            ]
            regex_local_store["binaries"] = [
                1 if regex_match else 0 for regex_match in regex_lookup
            ]
            regex_local_store["softmax"] = torch.softmax(
                regex_linear_output, 0).tolist()
            regex_local_store["predicted_label"] = torch.argmax(
                regex_linear_output, 0).item()

            # add inter-model diagnostics
            inter_model_store = local_store[
                "inter_model_distance_metrics"] = {}
            inter_model_store["softmax_difference_norm"] = torch.dist(
                torch.FloatTensor(neural_local_store["softmax"]),
                torch.FloatTensor(regex_local_store["softmax"])).item()
            inter_model_store["binary_misalignment_rate"] = sum([
                neural_binary != regex_binary
                for neural_binary, regex_binary in zip(
                    neural_local_store["binaries"],
                    regex_local_store["binaries"])
            ]) / len(neural_local_store["binaries"])

    # designate filename
    filename = os.path.join(
        model_log_directory, "_".join([
            "compare", output_prefix,
            os.path.basename(neural_model_checkpoint).replace(".pt", ""),
            os.path.basename(regex_model_checkpoint).replace(".pt", "")
        ]) + ".json")

    # dump final dictionary in model_log_directory
    LOGGER.info("Writing output file: %s" % filename)
    with open(filename, "w") as output_file_stream:
        json.dump(results_store, output_file_stream)


def compare_outer(args: argparse.Namespace) -> None:
    # log namespace arguments and model directory
    LOGGER.info(args)

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

    # load evaluation data here
    eval_input, eval_text = read_docs(args.eval_data, vocab)
    LOGGER.info("Sample evaluation text: %s" % eval_text[:10])
    eval_input = cast(List[List[int]], eval_input)
    eval_text = cast(List[List[str]], eval_text)
    eval_labels = read_labels(args.eval_labels)
    num_classes = len(set(eval_labels))
    eval_data = list(zip(eval_input, eval_labels))

    # apply maximum document length if necessary to text
    if args.max_doc_len is not None:
        eval_text = [
            eval_text[:args.max_doc_len]  # type: ignore
            for doc in eval_text
        ]

    # convert eval_text into list of string
    eval_text = [" ".join(doc) for doc in eval_text]

    # sort all data in decreasing order for batch processing
    eval_data, eval_text = map(
        list,
        zip(*sorted(zip(eval_data, eval_text),
                    key=lambda v: len(v[0][0]),
                    reverse=True)))
    eval_data = cast(List[Tuple[List[int], int]], eval_data)
    eval_text = cast(List[str], eval_text)

    # get semiring
    semiring = get_semiring(args)

    # create SoftPatternClassifier
    neural_model = SoftPatternClassifier(
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
    LOGGER.info("Neural model: %s" % neural_model)

    # execute inner comparison workflow
    compare_inner(eval_data, eval_text, neural_model,
                  args.neural_model_checkpoint, args.regex_model_checkpoint,
                  args.model_log_directory, args.batch_size, args.atol,
                  args.output_prefix, gpu_device, args.max_doc_len,
                  args.disable_tqdm)


def main(args: argparse.Namespace) -> None:
    # collect all checkpoints
    model_log_directory_collection = glob(args.model_log_directory)

    # loop over all provided models
    for model_log_directory in model_log_directory_collection:
        # start workflow and update argument namespace
        args.model_log_directory = model_log_directory
        try:
            args.neural_model_checkpoint = glob(
                os.path.join(model_log_directory,
                             "spp_checkpoint_best_*.pt"))[0]
        except IndexError:
            raise FileNotFoundError(
                "Best neural checkpoint is missing in directory: %s" %
                model_log_directory)
        try:
            args.regex_model_checkpoint = glob(
                os.path.join(model_log_directory,
                             "regex_compressed_spp_checkpoint_best_*.pt"))[0]
        except IndexError:
            raise FileNotFoundError(
                "Best regex checkpoint is missing in directory: %s" %
                model_log_directory)
        args = parse_configs_to_args(args, training=False)
        compare_outer(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter,
                                     parents=[
                                         evaluate_arg_parser(compare=True),
                                         hardware_arg_parser(),
                                         logging_arg_parser(),
                                         tqdm_arg_parser()
                                     ])
    LOGGER = stdout_root_logger(
        logging_arg_parser().parse_known_args()[0].logging_level)
    main(parser.parse_args())
