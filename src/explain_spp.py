#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple, Any, Generator
from .utils.parser_utils import ArgparseFormatter
from .utils.logging_utils import stdout_root_logger
from .utils.data_utils import PAD_TOKEN_INDEX, Vocab
from .utils.model_utils import decreasing_length, Batch
from .utils.explain_utils import (BackPointer, cat_2d, zip_ap_2d,
                                  get_nearest_neighbors)
from .arg_parser import (explain_arg_parser, hardware_arg_parser,
                         logging_arg_parser)
from .train_spp import (parse_configs_to_args, set_hardware, get_semiring,
                        get_train_valid_data, get_patterns)
from .spp_model import SoftPatternClassifier
from torch.nn import Embedding, Module
import argparse
import torch
import os


def get_top_scoring_sequences(
        model: Module, dev_set: List[Tuple[List[int], int]],
        max_doc_len: int) -> Generator[List[BackPointer], None, None]:
    for doc_idx, doc in enumerate(dev_set):
        yield get_top_scoring_spans_for_doc(model, doc, max_doc_len)


def transition_once_with_trace(
        model: Module, token_idx: int, eps_value: torch.Tensor,
        back_pointers: List[List[Any]], transition_matrix_val: torch.Tensor,
        restart_padding: List[BackPointer]) -> List[List[Any]]:
    def times(a, b):
        # wildly inefficient, oh well
        return model.semiring.times(torch.FloatTensor([a]),
                                    torch.FloatTensor([b]))[0]

    # Epsilon transitions (don't consume a token, move forward one state)
    # We do this before self-loops and single-steps.
    # We only allow one epsilon transition in a row.
    epsilons = cat_2d(
        restart_padding(token_idx),
        zip_ap_2d(
            lambda bp, e: BackPointer(score=times(bp.score, e),
                                      previous=bp,
                                      transition="epsilon-transition",
                                      start_token_idx=bp.start_token_idx,
                                      end_token_idx=token_idx),
            [xs[:-1] for xs in back_pointers],
            eps_value  # doesn't depend on token, just state
        ))

    epsilons = zip_ap_2d(max, back_pointers, epsilons)

    happy_paths = cat_2d(
        restart_padding(token_idx),
        zip_ap_2d(
            lambda bp, t: BackPointer(score=times(bp.score, t),
                                      previous=bp,
                                      transition="happy path",
                                      start_token_idx=bp.start_token_idx,
                                      end_token_idx=token_idx + 1),
            [xs[:-1] for xs in epsilons], transition_matrix_val[:, 1, :-1]))

    # Adding self loops (consume a token, stay in same state)
    self_loops = zip_ap_2d(
        lambda bp, sl: BackPointer(score=times(bp.score, sl),
                                   previous=bp,
                                   transition="self-loop",
                                   start_token_idx=bp.start_token_idx,
                                   end_token_idx=token_idx + 1), epsilons,
        transition_matrix_val[:, 0, :])
    return zip_ap_2d(max, happy_paths, self_loops)


def get_top_scoring_spans_for_doc(model: Module, doc: List[str],
                                  max_doc_len: int) -> List[BackPointer]:
    batch = Batch([doc[0]], model.embeddings, model.to_cuda, 0,
                  max_doc_len)  # single doc
    transition_matrices = model.get_transition_matrices(batch)
    num_patterns = model.total_num_patterns
    end_states = model.end_states.data.view(num_patterns)

    def restart_padding(t):
        return [
            BackPointer(score=x,
                        previous=None,
                        transition=None,
                        start_token_idx=t,
                        end_token_idx=t)
            for x in model.semiring.one(num_patterns)
        ]

    eps_value = model.get_eps_value().data
    hiddens = model.semiring.zero(num_patterns, model.max_pattern_length)
    # set start state activation to 1 for each pattern in each doc
    hiddens[:, 0] = model.semiring.one(num_patterns, 1)
    # convert to back-pointers
    hiddens = \
        [
            [
                BackPointer(
                    score=state_activation,
                    previous=None,
                    transition=None,
                    start_token_idx=0,
                    end_token_idx=0
                )
                for state_activation in pattern
            ]
            for pattern in hiddens
        ]
    # extract end-states
    end_state_back_pointers = [
        bp[end_state] for bp, end_state in zip(hiddens, end_states)
    ]
    for token_idx, transition_matrix in enumerate(transition_matrices):
        transition_matrix = transition_matrix[0, :, :, :].data
        hiddens = transition_once_with_trace(model, token_idx, eps_value,
                                             hiddens, transition_matrix,
                                             restart_padding)
        # extract end-states and max with current bests
        end_state_back_pointers = [
            max(best_bp, hidden_bps[end_state]) for best_bp, hidden_bps,
            end_state in zip(end_state_back_pointers, hiddens, end_states)
        ]
    return end_state_back_pointers


def explain_inner(model: Module,
                  dev_set: List[Tuple[List[int], int]],
                  dev_text: List[List[str]],
                  k_best: int,
                  max_doc_len: int = -1,
                  num_padding_tokens: int = 0) -> None:
    dev_sorted = decreasing_length(zip(dev_set, dev_text))
    dev_labels = [label for _, label in dev_set]
    dev_set = [doc for doc, _ in dev_sorted]
    dev_text = [text for _, text in dev_sorted]
    num_patterns = model.total_num_patterns
    pattern_length = model.max_pattern_length
    back_pointers = list(get_top_scoring_sequences(model, dev_set,
                                                   max_doc_len))

    # TODO: understand what this does in terms of frequent words
    # TODO: look into better practices for accessing model data with detach
    # TODO: consider issues with no loops policy and how it affects this
    # not sure where the exact sorting happenes as per docstring
    nearest_neighbors = \
        get_nearest_neighbors(
            model.diags.data,
            model.to_cuda(torch.FloatTensor(model.embeddings).t())
        ).view(
            num_patterns,
            model.num_diags,
            pattern_length
        )
    diags = model.diags.view(num_patterns, model.num_diags, pattern_length,
                             model.word_dim).data
    biases = model.bias.view(num_patterns, model.num_diags,
                             pattern_length).data
    self_loop_norms = torch.norm(diags[:, 0, :, :], 2, 2)
    self_loop_neighbs = nearest_neighbors[:, 0, :]
    self_loop_biases = biases[:, 0, :]
    fwd_one_norms = torch.norm(diags[:, 1, :, :], 2, 2)
    fwd_one_biases = biases[:, 1, :]
    fwd_one_neighbs = nearest_neighbors[:, 1, :]
    epsilons = model.get_eps_value().data

    for p in range(num_patterns):
        p_len = model.end_states[p].data[0] + 1
        k_best_doc_idxs = \
            sorted(
                range(len(dev_set)),
                key=lambda doc_idx: back_pointers[doc_idx][p].score,
                reverse=True  # high-scores first
            )[:k_best]

        def span_text(doc_idx):
            back_pointer = back_pointers[doc_idx][p]
            return back_pointer.score, back_pointer.display(
                dev_text[doc_idx], '#label={}'.format(dev_labels[doc_idx]),
                num_padding_tokens)

        print("Pattern:", p, "of length", p_len)
        print("Highest scoring spans:")
        for k, d in enumerate(k_best_doc_idxs):
            score, text = span_text(d)
            print("{} {:2.3f}  {}".format(k, score, text.encode('utf-8')))

        def transition_str(norm, neighb, bias):
            return "{:5.2f} * {:<15} + {:5.2f}".format(norm,
                                                       model.vocab[neighb],
                                                       bias)

        print(
            "self-loops: ", ", ".join(
                transition_str(norm, neighb, bias) for norm, neighb, bias in
                zip(self_loop_norms[p, :p_len], self_loop_neighbs[p, :p_len],
                    self_loop_biases[p, :p_len])))
        print(
            "fwd 1s:     ", ", ".join(
                transition_str(norm, neighb, bias)
                for norm, neighb, bias in zip(fwd_one_norms[
                    p, :p_len -
                    1], fwd_one_neighbs[p, :p_len -
                                        1], fwd_one_biases[p, :p_len - 1])))
        print("epsilons:   ",
              ", ".join("{:31.2f}".format(x) for x in epsilons[p, :p_len - 1]))
        print()


def explain_outer(args: argparse.Namespace, model_log_directory: str) -> None:
    # log namespace arguments and model directory
    LOGGER.info(args)
    LOGGER.info("Model log directory: %s" % model_log_directory)

    # set gpu and cpu hardware
    gpu_device = set_hardware(args)

    # get relevant patterns
    pattern_specs, pre_computed_patterns = get_patterns(args.patterns, None)

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
        pre_computed_patterns,
        args.shared_self_loops,
        args.no_epsilons,
        args.no_self_loops,
        args.bias_scale,
        args.epsilon_scale,
        args.self_loop_scale,
        0.)

    # log information about model
    LOGGER.info("Model: %s" % model)

    # execute inner function here
    explain_inner(explain_data, explain_text, model, args.model_checkpoint,
                  model_log_directory, args.k_best, args.batch_size,
                  gpu_device)


def main(args: argparse.Namespace) -> None:
    model_log_directory = os.path.dirname(args.model_checkpoint)
    args = parse_configs_to_args(args, model_log_directory, training=False)
    explain_outer(args, model_log_directory)


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
