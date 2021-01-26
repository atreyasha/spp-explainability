#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import OrderedDict
from typing import List, Union, cast, Any
from torch.nn import Module, Parameter, Linear, Dropout, LayerNorm, init
from .utils.model_utils import Semiring, Batch
from .utils.data_utils import Vocab
import torch

# shared_sl value for greedily learnable self-loop paramaters
SHARED_SL_PARAM_PER_STATE_PER_PATTERN = 1
# shared_sl value for global learnable self-loop parameter
SHARED_SL_SINGLE_PARAM = 2


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
                 shared_self_loops: int = 0,
                 no_epsilons: bool = False,
                 no_self_loops: bool = False,
                 bias_scale: Union[float, None] = None,
                 epsilon_scale: Union[float, None] = None,
                 self_loop_scale: Union[float, None] = None,
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
        self.shared_self_loops = shared_self_loops
        self.no_epsilons = no_epsilons
        self.no_self_loops = no_self_loops
        self.dropout = Dropout(dropout)
        self.num_diags = 2 if (not self.no_self_loops
                               and self.shared_self_loops == 0) else 1
        self.normalizer = LayerNorm(self.total_num_patterns)
        self.binarizer = STEHeaviside()

        # create transition matrix diagonal and bias tensors
        diags_size = (self.total_num_patterns * self.num_diags *
                      self.max_pattern_length)
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

        # assign epsilon-related variables from conditionals
        if not self.no_epsilons:
            # initialize epsilons and store it as a parameter
            epsilons = torch.Tensor(self.total_num_patterns,
                                    self.max_pattern_length - 1)
            init.xavier_normal_(epsilons)
            self.epsilons = Parameter(epsilons)

            # factor by which to scale epsilon parameter
            if epsilon_scale is not None:
                self.register_buffer("epsilon_scale",
                                     self.semiring.from_outer_to_semiring(
                                         torch.FloatTensor([epsilon_scale])),
                                     persistent=False)
            else:
                self.register_buffer("epsilon_scale",
                                     semiring.one(1),
                                     persistent=False)

        # assign self-loop related variables from conditionals
        if not self.no_self_loops:
            if self.shared_self_loops != 0:
                # shared parameters between main path and self loop
                # 1: one parameter per state per pattern
                if (self.shared_self_loops ==
                        SHARED_SL_PARAM_PER_STATE_PER_PATTERN):
                    # create a tensor for each pattern
                    shared_self_loop_data = torch.Tensor(
                        self.total_num_patterns, self.max_pattern_length)
                    init.xavier_normal_(shared_self_loop_data)
                # 2: a single global parameter
                elif self.shared_self_loops == SHARED_SL_SINGLE_PARAM:
                    # create a single tensor
                    shared_self_loop_data = torch.randn(1)
                # NOTE: assign tensor to a learnable parameter
                self.self_loop_scale = Parameter(shared_self_loop_data)
            else:
                # workflow for self-loops that are not shared
                if self_loop_scale is not None:
                    self.register_buffer("self_loop_scale",
                                         self.semiring.from_outer_to_semiring(
                                             torch.FloatTensor(
                                                 [self_loop_scale])),
                                         persistent=False)
                else:
                    self.register_buffer("self_loop_scale",
                                         semiring.one(1),
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

        # compute transition scores which document transition scores
        # into each state given the current token
        transition_scores = self.semiring.from_outer_to_semiring(
            torch.mm(self.diags, batch.local_embeddings) +
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
            batched_transition_scores[:, token_index, :, :, :]
            for token_index in range(max_doc_len)
        ]

        # finally return transition matrices for all tokens
        return transition_matrices

    def forward(self, batch: Batch) -> torch.Tensor:
        # start timer and get transition matrices
        transition_matrices = self.get_transition_matrices(batch)

        # set self_loop_scale based on class variables
        self_loop_scale = None
        if self.shared_self_loops:
            self_loop_scale = self.semiring.from_outer_to_semiring(
                self.self_loop_scale)
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

        # set start state (0) to semiring 1 for each pattern in each doc
        hiddens[:, :, 0] = self.semiring.one(batch_size,
                                             self.total_num_patterns)

        # get epsilon_values based on previous class settings
        epsilon_values = self.get_epsilon_values()

        # start loop over all transition matrices
        for token_index, transition_matrix in enumerate(transition_matrices):
            # retrieve all hiddens given current state embeddings
            hiddens = self.transition_once(epsilon_values, hiddens,
                                           transition_matrix, zero_padding,
                                           restart_padding, self_loop_scale)

            # look at the end state for each pattern, and "add" it into score
            end_state_values = torch.gather(hiddens, 2, end_states).view(
                batch_size, self.total_num_patterns)

            # only update score when we're not already past the end of the doc
            active_doc_indices = torch.nonzero(torch.gt(
                batch.doc_lens, token_index),
                                               as_tuple=True)[0]

            # update scores with relevant tensor values
            scores[active_doc_indices] = self.semiring.plus(
                scores[active_doc_indices],
                end_state_values[active_doc_indices])

        # extract scores from semiring to outer set
        scores = self.semiring.from_semiring_to_outer(scores)

        # execute normalization of scores
        scores = self.normalizer(scores)

        # binarize scores using STEHeaviside
        scores = self.binarizer(scores)

        # return output of final layer
        return self.linear.forward(scores)

    def get_epsilon_values(self) -> Union[torch.Tensor, None]:
        return None if self.no_epsilons else self.semiring.times(
            self.epsilon_scale.detach().clone(),  # type: ignore
            self.semiring.from_outer_to_semiring(self.epsilons))

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
                torch.cat(
                    (zero_padding,
                     self.semiring.times(hiddens[:, :, :-1], epsilon_values)),
                    2))

        # adding the start state
        after_main_paths = torch.cat(
            (restart_padding,
             self.semiring.times(after_epsilons[:, :, :-1],
                                 transition_matrix[:, :, -1, :-1])), 2)

        # conditionally adding self-loops
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
