#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, Any
from collections import OrderedDict
from torch.nn import Module, Parameter, Linear, Dropout, LayerNorm, init
from .utils.model_utils import Semiring, Batch
from .utils.data_utils import Vocab
import torch


class STEHeavisideFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Any) -> Any:  # type: ignore
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:  # type: ignore
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_input


class STEHeaviside(Module):
    def __init__(self) -> None:
        super(STEHeaviside, self).__init__()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = STEHeavisideFunction.apply(batch)
        return batch


class SoftPatternClassifier(Module):
    def __init__(self,
                 pattern_specs: 'OrderedDict[int, int]',
                 num_classes: int,
                 embeddings: Module,
                 vocab: Vocab,
                 semiring: Semiring,
                 no_wildcards: bool = False,
                 bias_scale: Union[float, None] = None,
                 wildcard_scale: Union[float, None] = None,
                 dropout: float = 0.) -> None:
        # initialize all class properties from torch.nn.Module
        super(SoftPatternClassifier, self).__init__()

        # assign quick class variables
        self.pattern_specs = pattern_specs
        self.max_pattern_length = max(list(pattern_specs.keys()))
        self.total_num_patterns = sum(pattern_specs.values())
        self.linear = Linear(self.total_num_patterns, num_classes)
        self.embeddings = embeddings
        self.vocab = vocab
        self.semiring = semiring
        self.no_wildcards = no_wildcards
        self.dropout = Dropout(dropout)
        self.normalizer = LayerNorm(self.total_num_patterns,
                                    elementwise_affine=False)
        self.binarizer = STEHeaviside()

        # create transition matrix diagonal and bias tensors
        diags_size = (self.total_num_patterns * (self.max_pattern_length - 1))
        diags = torch.Tensor(  # type: ignore
            diags_size, self.embeddings.embedding_dim)
        bias = torch.Tensor(diags_size, 1)

        # initialize diags and bias using glorot initialization
        init.xavier_normal_(diags)
        init.xavier_normal_(bias)

        # convert both diagonal and bias data into learnable parameters
        self.diags = Parameter(diags)
        self.bias = Parameter(bias)

        # assign bias_scale based on conditionals
        if bias_scale is not None:
            self.register_buffer("bias_scale",
                                 torch.FloatTensor([bias_scale]),
                                 persistent=False)
        else:
            self.register_buffer("bias_scale",
                                 torch.FloatTensor([1.]),
                                 persistent=False)

        # assign wildcard-related variables from conditionals
        if not self.no_wildcards:
            # initialize wildcards and store it as a parameter
            wildcards = torch.Tensor(self.total_num_patterns,
                                     self.max_pattern_length - 1)
            init.xavier_normal_(wildcards)
            self.wildcards = Parameter(wildcards)

            # factor by which to scale wildcard parameter
            if wildcard_scale is not None:
                self.register_buffer("wildcard_scale",
                                     self.semiring.from_outer_to_semiring(
                                         torch.FloatTensor([wildcard_scale])),
                                     persistent=False)
            else:
                self.register_buffer("wildcard_scale",
                                     self.semiring.one(1),
                                     persistent=False)

        # register end_states tensor
        self.register_buffer(
            "end_states",
            torch.LongTensor(
                [[end]
                 for pattern_len, num_patterns in self.pattern_specs.items()
                 for end in num_patterns * [pattern_len - 1]]),
            persistent=False)

        # register null scores tensor
        self.register_buffer("scores",
                             self.semiring.zero(self.total_num_patterns),
                             persistent=False)

        # register restart_padding tensor
        self.register_buffer("restart_padding",
                             self.semiring.one(self.total_num_patterns, 1),
                             persistent=False)

        # register hiddens tensor
        self.register_buffer(
            "hiddens",
            torch.cat((self.semiring.one(self.total_num_patterns, 1),
                       self.semiring.zero(self.total_num_patterns,
                                          self.max_pattern_length - 1)),
                      axis=1),
            persistent=False)

    def get_transition_matrices(self, batch: Batch) -> torch.Tensor:
        # initialize local variables
        batch_size = batch.size()
        max_doc_len = batch.max_doc_len

        # compute transition scores which document transition scores
        # into each state given the current token
        transition_matrices = self.semiring.from_outer_to_semiring(
            torch.mm(self.diags, batch.local_embeddings) +
            self.bias_scale * self.bias).t()

        # apply registered dropout
        transition_matrices = self.dropout(transition_matrices)

        # index transition scores for each doc in batch
        transition_matrices = [
            torch.index_select(transition_matrices, 0, doc)
            for doc in batch.docs
        ]

        # reformat transition scores
        transition_matrices = torch.cat(transition_matrices).view(
            batch_size, max_doc_len, self.total_num_patterns,
            self.max_pattern_length - 1)

        # finally return transition matrices for all tokens
        return transition_matrices

    def get_wildcard_matrix(self) -> Union[torch.Tensor, None]:
        return None if self.no_wildcards else self.semiring.times(
            self.wildcard_scale.clone(),  # type: ignore
            self.semiring.from_outer_to_semiring(self.wildcards))

    def forward(self, batch: Batch, explain: bool = False) -> torch.Tensor:
        # start timer and get transition matrices
        transition_matrices = self.get_transition_matrices(batch)

        # assign batch_size
        batch_size = batch.size()

        # clone null scores tensor
        scores = self.scores.expand(  # type: ignore
            batch_size, -1).clone()

        # clone restart_padding tensor to add start state for each word
        restart_padding = self.restart_padding.expand(  # type: ignore
            batch_size, -1, -1).clone()

        # initialize hiddens tensor
        hiddens = self.hiddens.expand(  # type: ignore
            batch_size, -1, -1).clone()

        # enumerate all end pattern states
        end_states = self.end_states.expand(  # type: ignore
            batch_size, self.total_num_patterns, -1).clone()

        # get wildcard_matrix based on previous class settings
        wildcard_matrix = self.get_wildcard_matrix()

        # start loop over all transition matrices
        for token_index in range(transition_matrices.size(1)):
            # extract current transition matrix
            transition_matrix = transition_matrices[:, token_index, :, :]

            # retrieve all hiddens given current state embeddings
            hiddens = self.transition_once(hiddens, transition_matrix,
                                           wildcard_matrix, restart_padding)

            # look at the end state for each pattern, and "add" it into score
            end_state_values = torch.gather(hiddens, 2, end_states).view(
                batch_size, self.total_num_patterns)

            # only index active documents and not padding tokens
            active_doc_indices = torch.nonzero(
                torch.gt(batch.doc_lens,
                         token_index), as_tuple=True)[0]  # yapf: disable

            # update scores with relevant tensor values
            scores[active_doc_indices] = self.semiring.plus(
                scores[active_doc_indices],
                end_state_values[active_doc_indices])

        # clone raw scores to keep track of it
        if explain:
            interim_scores = scores.clone()

        # extract all infinite indices
        inf_indices = torch.where(scores == float("-inf"))

        # extract scores from semiring to outer set
        scores = self.semiring.from_semiring_to_outer(scores).clone()

        # temporarily replace all negative infinities by positive infinities
        scores[inf_indices] = float("inf")

        # find the minimum of all original scores and replace infinite by this
        batch_minima = torch.min(scores, 1)[0]

        # assing batch-level minima to infinity indices
        scores[inf_indices] = torch.repeat_interleave(
            batch_minima, torch.sum(scores.isinf(), 1))

        # execute normalization of scores
        scores = self.normalizer(scores)

        # binarize scores using STEHeaviside
        scores = self.binarizer(scores)

        # conditionally return different tensors depending on routine
        if explain:
            interim_scores = torch.stack((interim_scores, scores.clone()))
            return interim_scores
        else:
            return self.linear.forward(scores)

    def transition_once(self, hiddens: torch.Tensor,
                        transition_matrix: torch.Tensor,
                        wildcard_matrix: Union[torch.Tensor, None],
                        restart_padding: torch.Tensor) -> torch.Tensor:
        # adding the start state and main transition
        main_transitions = torch.cat(
            (restart_padding,
             self.semiring.times(hiddens[:, :, :-1], transition_matrix)), 2)

        # adding wildcard transitions
        if self.no_wildcards:
            return main_transitions
        else:
            wildcard_transitions = torch.cat(
                (restart_padding,
                 self.semiring.times(hiddens[:, :, :-1], wildcard_matrix)), 2)
            # either main transition or wildcard
            return self.semiring.plus(main_transitions, wildcard_transitions)
