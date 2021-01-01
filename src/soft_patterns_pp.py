#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
from typing import List, Union, Tuple, cast
from torch import FloatTensor, LongTensor, cat, mm, randn, relu
from torch.nn import Module, Parameter, ModuleList, Linear, Dropout
from .utils.model_utils import normalize, Semiring, Batch
from .utils.data_utils import Vocab
import torch

# CW token refers to an arbitrary token with high bias
CW_TOKEN = "CW"
# factor to keep matrix values small but nonzero
NUMERICAL_EPSILON = 1e-10
# shared_sl value for greedily learnable self-loop paramaters
SHARED_SL_PARAM_PER_STATE_PER_PATTERN = 1
# shared_sl value for global learnable self-loop parameter
SHARED_SL_SINGLE_PARAM = 2


class MLP(Module):
    def __init__(self, input_dim: int, mlp_hidden_dim: int,
                 mlp_num_layers: int, num_classes: int) -> None:
        # initialize all class properties from torch.nn.Module
        super(MLP, self).__init__()

        # set up class variables
        self.mlp_num_layers = mlp_num_layers

        # create MLP structure based on input variables
        layers = []
        for i in range(mlp_num_layers):
            d1 = input_dim if i == 0 else mlp_hidden_dim
            d2 = mlp_hidden_dim if i < (mlp_num_layers - 1) else num_classes
            layer = Linear(d1, d2)
            layers.append(layer)

        # register layers as a class specific variable
        # ModuleList registers submodules in a list
        self.layers = ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get output of first layer without relu
        # must be excluded from loop below to prevent
        # relu being applied to input
        res = self.layers[0](x)

        # loop over remaining layer with relu
        for i in range(1, len(self.layers)):
            res = self.layers[i](relu(res))

        # return final output
        return res


class SoftPatternClassifier(Module):
    def __init__(self,
                 pattern_specs: 'OrderedDict[int, int]',
                 mlp_hidden_dim: int,
                 mlp_num_layers: int,
                 num_classes: int,
                 embeddings: Module,
                 vocab: Vocab,
                 semiring: Semiring,
                 bias_scale: float,
                 pre_computed_patterns: Union[List[List[str]], None] = None,
                 shared_self_loops: int = 0,
                 no_self_loops: bool = False,
                 no_epsilons: bool = False,
                 epsilon_scale: Union[float, None] = None,
                 self_loop_scale: Union[torch.Tensor, float, None] = None,
                 dropout: float = 0.) -> None:
        # initialize all class properties from torch.nn.Module
        super(SoftPatternClassifier, self).__init__()

        # assign quick class variables
        self.semiring = semiring
        self.vocab = vocab
        self.embeddings = embeddings
        self.total_num_patterns = sum(pattern_specs.values())
        self.mlp = MLP(self.total_num_patterns, mlp_hidden_dim, mlp_num_layers,
                       num_classes)
        self.num_diags = 1
        self.no_self_loops = no_self_loops
        self.shared_self_loops = shared_self_loops
        self.pattern_specs = pattern_specs
        self.max_pattern_length = max(list(pattern_specs.keys()))
        self.no_epsilons = no_epsilons
        self.bias_scale = bias_scale
        self.dropout = Dropout(dropout)

        # assign class variables from conditionals
        if self.shared_self_loops:
            # shared parameters between main path and self loop
            # 1: one parameter per state per pattern
            # 2: a single global parameter
            if self.shared_self_loops == SHARED_SL_PARAM_PER_STATE_PER_PATTERN:
                # create a tensor for each pattern
                shared_self_loop_data = randn(self.total_num_patterns,
                                              self.max_pattern_length)
            elif self.shared_self_loops == SHARED_SL_SINGLE_PARAM:
                # create a single tensor
                shared_self_loop_data = randn(1)
            # NOTE: assign tensor to a learnable parameter
            self.self_loop_scale = Parameter(shared_self_loop_data)
        elif not self.no_self_loops:
            # workflow for self-loops that are not shared
            # NOTE: self_loop_scale is not a fixed tensor
            if self_loop_scale is not None:
                self.register_buffer("self_loop_scale",
                                     self.semiring.from_float(
                                         FloatTensor([self_loop_scale])),
                                     persistent=False)
            else:
                self.register_buffer("self_loop_scale",
                                     semiring.one(1),
                                     persistent=False)
            # assign two diagonals instead of default one
            self.num_diags = 2

        # create list of end state indexes for each pattern
        end_states = [[
            end
        ] for pattern_len, num_patterns in self.pattern_specs.items()
                      for end in num_patterns * [pattern_len - 1]]
        self.register_buffer("end_states",
                             LongTensor(end_states),
                             persistent=False)

        # create transition matrix diagonal and bias tensors
        # normalize diagonal data tensor
        diags_size = (self.total_num_patterns * self.num_diags *
                      self.max_pattern_length)
        diags = randn(  # type: ignore
            diags_size, self.embeddings.embedding_dim)
        bias = randn(diags_size, 1)
        normalize(diags)

        # load diagonal and bias data from patterns if provided
        if pre_computed_patterns is not None:
            diags, bias = self.load_pre_computed_patterns(
                pre_computed_patterns, diags, bias, pattern_specs)

        # convert both diagonal and bias data into learnable parameters
        self.diags = Parameter(diags)
        self.bias = Parameter(bias)

        # assign class variables if epsilon transitions are allowed
        if not self.no_epsilons:
            # NOTE: this parameter is learned
            self.epsilons = Parameter(
                randn(self.total_num_patterns, self.max_pattern_length - 1))

            # factor by which to scale epsilon parameter
            if epsilon_scale is not None:
                self.register_buffer("epsilon_scale",
                                     self.semiring.from_float(
                                         FloatTensor([epsilon_scale])),
                                     persistent=False)
            else:
                self.register_buffer("epsilon_scale",
                                     semiring.one(1),
                                     persistent=False)

        # register null scores tensor
        self.register_buffer("scores",
                             self.semiring.zero(self.total_num_patterns),
                             persistent=False)

        # register restart_padding tensor
        self.register_buffer("restart_padding",
                             self.semiring.one(self.total_num_patterns, 1),
                             persistent=False)

        # register zero_padding tensor
        self.register_buffer("zero_padding",
                             self.semiring.zero(self.total_num_patterns, 1),
                             persistent=False)

        # register hiddens tensor
        self.register_buffer("hiddens",
                             self.semiring.zero(self.total_num_patterns,
                                                self.max_pattern_length),
                             persistent=False)

    def get_transition_matrices(self, batch: Batch) -> List[torch.Tensor]:
        # initialize local variables
        batch_size = batch.size()
        max_doc_len = batch.max_doc_len

        # load transition scores
        # mm: (diags_size x word_dim) @ (word_dim x batch_vocab_size)
        # transition_score: diags_size x batch_vocab_size
        # these would represent transition scores for each word in vocab
        transition_scores = self.semiring.from_float(
            mm(self.diags, batch.local_embeddings) +
            self.bias_scale * self.bias).t()

        # apply registered dropout
        transition_scores = self.dropout(transition_scores)

        # indexes transition scores for each doc in batch
        batched_transition_scores = [
            torch.index_select(transition_scores, 0, doc) for doc in batch.docs
        ]

        # reformats transition scores
        batched_transition_scores = torch.cat(batched_transition_scores).view(
            batch_size, max_doc_len, self.total_num_patterns, self.num_diags,
            self.max_pattern_length)

        # get transition matrix for each token
        transition_matrices = [
            batched_transition_scores[:, word_index, :, :, :]
            for word_index in range(max_doc_len)
        ]

        # finally return transition matrices for all tokens
        return transition_matrices

    def load_pre_computed_patterns(
        self, pre_computed_patterns: List[List[str]], diags: torch.Tensor,
        bias: torch.Tensor, pattern_specs: 'OrderedDict[int, int]'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # extract pattern length as indices with zero counts
        pattern_indices = dict(
            (pattern_length, 0) for pattern_length in pattern_specs)

        # view diags and bias in appropriate tensor sizes
        diags_size = diags.size()[0]
        diags = diags.view(  # type: ignore
            self.total_num_patterns, self.num_diags, self.max_pattern_length,
            self.embeddings.embedding_dim)
        bias = bias.view(self.total_num_patterns, self.num_diags,
                         self.max_pattern_length)

        # initialize counter
        count = 0

        # pattern indices: which patterns are we loading?
        # the pattern index from which we start loading each pattern length
        # NOTE: offset probably to get pattern index in diagonal data
        for (i, pattern_length) in enumerate(pattern_specs.keys()):
            pattern_indices[pattern_length] = count
            count += pattern_specs[pattern_length]

        # loading all pre-computed patterns
        for pattern in pre_computed_patterns:
            pattern_length = len(pattern) + 1

            # getting pattern index in diagonal data
            # NOTE: this is why previous workflow is present
            index = pattern_indices[pattern_length]

            # loading diagonal and bias for pattern
            diags_subset, bias_subset = self.load_pattern(pattern)

            # updating diagonal and bias
            # NOTE: this is why reformatting was necessary
            diags[index, 1, :(pattern_length - 1), :] = diags_subset
            bias[index, 1, :(pattern_length - 1)] = bias_subset

            # updating pattern_indices
            # NOTE: this ensures next update does not override current
            pattern_indices[pattern_length] += 1

        # return tensors in appropriate data format
        return diags.view(diags_size,
                          self.embeddings.embedding_dim), bias.view(
                              diags_size, 1)

    def load_pattern(self,
                     pattern: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # initialize local variables
        diags_subset = NUMERICAL_EPSILON * randn(  # type: ignore
            len(pattern), self.embeddings.embedding_dim)
        bias_subset = torch.zeros(len(pattern))

        # NOTE: arbitrary choice which can be refined
        factor = 10

        # traversing elements of pattern.
        for (i, element) in enumerate(pattern):
            # CW: high bias (we don't care about the identity of the token
            if element == CW_TOKEN:
                bias_subset[i] = factor
            else:
                # concrete word: we do care about the token (low bias).
                bias_subset[i] = -factor
                # if we have a word vector for this element
                # update the diagonal value with specific vector
                if element in self.vocab:
                    diags_subset[i] = FloatTensor(
                        factor *
                        self.embeddings(LongTensor(self.vocab(element))))

        # return updated tensors
        return diags_subset, bias_subset

    def forward(self, batch: Batch) -> torch.Tensor:
        # start timer and get transition matrices
        transition_matrices = self.get_transition_matrices(batch)

        # set self_loop_scale based on class variables
        self_loop_scale = None
        if self.shared_self_loops:
            self_loop_scale = self.semiring.from_float(self.self_loop_scale)
        elif not self.no_self_loops:
            self_loop_scale = self.self_loop_scale.detach().clone()

        # assign batch_size
        batch_size = batch.size()

        # clone null scores tensor
        scores = self.scores.expand(  # type: ignore
            batch_size, -1).detach().clone()

        # clone restart_padding tensor to add start state for each word
        restart_padding = self.restart_padding.expand(  # type: ignore
            batch_size, -1, -1).detach().clone()

        # clone zero_padding tensor
        zero_padding = self.zero_padding.expand(  # type: ignore
            batch_size, -1, -1).detach().clone()

        # initialize hiddens tensor
        hiddens = self.hiddens.expand(  # type: ignore
            batch_size, -1, -1).detach().clone()

        # enumerate all end pattern states
        end_states = self.end_states.expand(  # type: ignore
            batch_size, self.total_num_patterns, -1).detach().clone()

        # set start state (0) to 1 for each pattern in each doc
        hiddens[:, :, 0] = self.semiring.one(batch_size,
                                             self.total_num_patterns)

        # get epsilon_values based on previous class settings
        epsilon_values = self.get_epsilon_values()

        # start loop over all transition matrices
        for i, transition_matrix in enumerate(transition_matrices):
            # retrieve all hiddens given current state embeddings
            hiddens = self.transition_once(epsilon_values, hiddens,
                                           transition_matrix, zero_padding,
                                           restart_padding, self_loop_scale)

            # look at the end state for each pattern, and "add" it into score
            # NOTE: torch.gather helps to extract values at indices
            end_state_values = torch.gather(hiddens, 2, end_states).view(
                batch_size, self.total_num_patterns)

            # only update score when we're not already past the end of the doc
            # NOTE: this is useful for mixed length documents
            active_doc_indices = torch.nonzero(torch.gt(batch.doc_lens, i),
                                               as_tuple=True)[0]

            # update scores with relevant tensor values
            scores[active_doc_indices] = self.semiring.plus(
                scores[active_doc_indices],
                end_state_values[active_doc_indices])

        # NOTE: scores represent end values on top of SoPa
        scores = self.semiring.to_float(scores)

        # return output of MLP
        return self.mlp.forward(scores)

    def get_epsilon_values(self) -> Union[torch.Tensor, None]:
        return None if self.no_epsilons else self.semiring.times(
            self.epsilon_scale.detach().clone(),  # type: ignore
            self.semiring.from_float(self.epsilons))

    def transition_once(
            self, epsilon_values: Union[torch.Tensor, None],
            hiddens: torch.Tensor, transition_matrix: torch.Tensor,
            zero_padding: torch.Tensor, restart_padding: torch.Tensor,
            self_loop_scale: Union[torch.Tensor, None]) -> torch.Tensor:
        # adding epsilon transitions
        # NOTE: don't consume a token, move forward one state
        # we do this before self-loops and single-steps.
        # we only allow zero or one epsilon transition in a row
        if self.no_epsilons:
            after_epsilons = hiddens
        else:
            # doesn't depend on token, just state
            after_epsilons = self.semiring.plus(
                hiddens,
                cat((zero_padding,
                     self.semiring.times(hiddens[:, :, :-1], epsilon_values)),
                    2))

        # adding the start state
        after_main_paths = cat(
            (restart_padding,
             self.semiring.times(after_epsilons[:, :, :-1],
                                 transition_matrix[:, :, -1, :-1])), 2)

        # adding self-loops
        if self.no_self_loops:
            return after_main_paths
        else:
            # NOTE: mypy-related fix, does not affect torch workflow
            self_loop_scale = cast(torch.Tensor, self_loop_scale)

            # adjust self_loop_scale
            if self.shared_self_loops == SHARED_SL_PARAM_PER_STATE_PER_PATTERN:
                self_loop_scale = self_loop_scale.expand(
                    transition_matrix[:, :, 0, :].size())

            # adding self loops (consume a token, stay in same state)
            after_self_loops = self.semiring.times(
                self_loop_scale,
                self.semiring.times(after_epsilons, transition_matrix[:, :,
                                                                      0, :]))

            # either happy or self-loop, not both
            return self.semiring.plus(after_main_paths, after_self_loops)
