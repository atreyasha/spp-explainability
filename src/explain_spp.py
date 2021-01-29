#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
from typing import List, Tuple, Union
from torch.nn import Embedding, Module
from .utils.parser_utils import ArgparseFormatter
from .utils.logging_utils import stdout_root_logger
from .utils.data_utils import PAD_TOKEN_INDEX, Vocab
from .utils.model_utils import decreasing_length, to_cuda, Batch
from .utils.explain_utils import (cat_nested, zip_lambda_nested,
                                  torch_apply_float_function, BackPointer)
from .arg_parser import (explain_arg_parser, hardware_arg_parser,
                         logging_arg_parser)
from .train_spp import (parse_configs_to_args, set_hardware, get_semiring,
                        get_train_valid_data, get_pattern_specs)
from .spp_model import SoftPatternClassifier
import argparse
import torch
import os


def restart_padding(model: Module, token_index: int,
                    total_num_patterns: int) -> List[List[BackPointer]]:
    return [
        BackPointer(raw_score=state_activation,
                    binarized_score=0.,
                    pattern_index=pattern_index,
                    previous=None,
                    transition=None,
                    start_token_index=token_index,
                    current_token_index=token_index,
                    end_token_index=token_index)
        for pattern_index, state_activation in enumerate(
            model.semiring.one(total_num_patterns))
    ]


def transition_once_with_trace(
        model: Module, token_index: int, total_num_patterns: int,
        wildcard_values: Union[torch.Tensor,
                               None], hiddens: List[List[BackPointer]],
        transition_matrix: torch.Tensor) -> List[List[BackPointer]]:
    # add main transition; consume a token and state
    main_transitions = cat_nested(
        restart_padding(model, token_index, total_num_patterns),
        zip_lambda_nested(
            lambda back_pointer, transition_value:
            BackPointer(raw_score=torch_apply_float_function(
                model.semiring.times, back_pointer.raw_score, transition_value
            ),
                        binarized_score=0.,
                        pattern_index=back_pointer.pattern_index,
                        previous=back_pointer,
                        transition="main_transition",
                        start_token_index=back_pointer.start_token_index,
                        current_token_index=token_index,
                        end_token_index=token_index + 1),
            [hidden[:-1] for hidden in hiddens], transition_matrix))

    if model.no_wildcards:
        return main_transitions
    else:
        # add wildcard transition; consume a generic token and state
        wildcard_transitions = cat_nested(
            restart_padding(model, token_index, total_num_patterns),
            zip_lambda_nested(
                lambda back_pointer, wildcard_value: BackPointer(
                    raw_score=torch_apply_float_function(
                        model.semiring.times, back_pointer.raw_score,
                        wildcard_value),
                    binarized_score=0.,
                    pattern_index=back_pointer.pattern_index,
                    previous=back_pointer,
                    transition="wildcard_transition",
                    start_token_index=back_pointer.start_token_index,
                    current_token_index=token_index,
                    end_token_index=token_index + 1),
                [hidden[:-1] for hidden in hiddens], wildcard_values))

        # return final object
        return zip_lambda_nested(max, main_transitions, wildcard_transitions)


def get_activating_spans(
        model: Module, doc: Tuple[List[int], int],
        max_doc_len: Union[int, None],
        gpu_device: Union[torch.device, None]) -> List[BackPointer]:  # yapf: disable
    # initialize local variables
    batch = Batch([doc[0]], model.embeddings, to_cuda(gpu_device), 0.,
                  max_doc_len)
    scores_history = model.forward(batch, explain=True).squeeze()
    transition_matrices = model.get_transition_matrices(batch)
    total_num_patterns = model.total_num_patterns
    wildcard_values = model.get_wildcard_values()
    end_states = model.end_states.squeeze()
    hiddens = model.semiring.zero(total_num_patterns, model.max_pattern_length)

    # set start state activation to 1 for each pattern
    hiddens[:, 0] = model.semiring.one(total_num_patterns)

    # convert to back-pointers
    hiddens = [[
        BackPointer(raw_score=state_activation,
                    binarized_score=0.,
                    pattern_index=pattern_index,
                    previous=None,
                    transition=None,
                    start_token_index=0,
                    current_token_index=0,
                    end_token_index=0) for state_activation in pattern
    ] for pattern_index, pattern in enumerate(hiddens)]

    # create end-states
    end_state_back_pointers = [
        bp[end_state] for bp, end_state in zip(hiddens, end_states)
    ]

    # iterate over sequence
    for token_index in range(transition_matrices.size(1)):
        transition_matrix = transition_matrices[0, token_index, :, :]
        hiddens = transition_once_with_trace(model, token_index,
                                             total_num_patterns,
                                             wildcard_values, hiddens,
                                             transition_matrix)
        # extract end-states and max with current bests
        end_state_back_pointers = [
            max(best_back_pointer, hidden_back_pointers[end_state])
            for best_back_pointer, hidden_back_pointers, end_state in zip(
                end_state_back_pointers, hiddens, end_states)
        ]

    # check that both explainability routine and model match
    assert torch.equal(
        torch.Tensor([bp.raw_score for bp in end_state_back_pointers]),
        scores_history[0]), ("Explainability routine does not produce "
                             "matching scores with SoPa++ routine")

    for pattern_index in range(total_num_patterns):
        end_state_back_pointers[
            pattern_index].binarized_score = scores_history[1][
                pattern_index].item()

    # return best back pointers
    return end_state_back_pointers


def explain_inner(explain_data: List[Tuple[List[int], int]],
                  explain_text: List[List[str]],
                  model: Module,
                  model_checkpoint: str,
                  model_log_directory: str,
                  batch_size: int,
                  gpu_device: Union[torch.device, None],
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

    # sort explain data and text in decreasing length order
    explain_data, explain_text = zip(
        *decreasing_length(zip(explain_data, explain_text)))

    # start explanation workflow on all explain_data
    activating_spans_back_pointers = [[
        back_pointer if back_pointer.binarized_score else None
        for back_pointer in back_pointers
    ] for back_pointers in [
        get_activating_spans(model, data, max_doc_len, gpu_device)
        for data in tqdm(explain_data)
    ]]

    # iterate over all patterns
    for pattern_index in range(model.total_num_patterns):
        # log current iterating
        LOGGER.info(
            "Pattern %s of length %s:" %
            (pattern_index, model.end_states[pattern_index].item() + 1))

        # log relevant spans
        text_back_pointers = sorted([
            [explain_text[index],
             activating_spans_back_pointers[index][pattern_index]]
            for index in range(len(explain_data))
            if activating_spans_back_pointers[index][pattern_index]
        ], key=lambda mixed: mixed[1].raw_score)
        activating_spans = [back_pointer.display(text)
                            for text, back_pointer in text_back_pointers]
        for text in activating_spans:
            LOGGER.info(text)


def explain_outer(args: argparse.Namespace) -> None:
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
        args.no_wildcards,
        args.bias_scale,
        args.wildcard_scale,
        0.)

    # log information about model
    LOGGER.info("Model: %s" % model)

    # execute inner function here
    explain_inner(explain_data, explain_text, model, args.model_checkpoint,
                  model_log_directory, args.batch_size, gpu_device,
                  args.max_doc_len)


def main(args: argparse.Namespace) -> None:
    args.model_log_directory = os.path.dirname(args.model_checkpoint)
    args = parse_configs_to_args(args, training=False)
    explain_outer(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter,
                                     parents=[
                                         explain_arg_parser(),
                                         hardware_arg_parser(),
                                         logging_arg_parser()
                                     ])
    LOGGER = stdout_root_logger(
        logging_arg_parser().parse_known_args()[0].logging_level)
    main(parser.parse_args())
