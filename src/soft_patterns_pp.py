#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import monotonic
from typing import List, Union, Tuple, cast, MutableMapping, Any
from torch import FloatTensor, LongTensor, cat, mm, randn, relu
from torch.nn import Module, Parameter, ModuleList, Linear
from torch.nn.utils.rnn import pad_packed_sequence
from .utils.model_utils import (to_cuda, argmax, fixed_var, normalize,
                                Semiring, Batch)
from .utils.data_utils import Vocab
import numpy as np
import torch

CW_TOKEN = "CW"
EPSILON = 1e-10
SHARED_SL_PARAM_PER_STATE_PER_PATTERN = 1
SHARED_SL_SINGLE_PARAM = 2


class MLP(Module):
    """
    A multilayer perceptron with one hidden ReLU layer.
    Expects an input tensor of size (batch_size, input_dim) and returns
    a tensor of size (batch_size, output_dim).
    """
    def __init__(self, input_dim: int, hidden_layer_dim: int, num_layers: int,
                 num_classes: int) -> None:
        super(MLP, self).__init__()

        self.num_layers = num_layers

        # create a list of layers of size num_layers
        layers = []
        for i in range(num_layers):
            d1 = input_dim if i == 0 else hidden_layer_dim
            d2 = hidden_layer_dim if i < (num_layers - 1) else num_classes
            layer = Linear(d1, d2)
            layers.append(layer)

        self.layers = ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.layers[0](x)
        for i in range(1, len(self.layers)):
            res = self.layers[i](relu(res))
        return res


class SoftPatternClassifier(Module):
    """
    A text classification model that feeds the document scores from a bunch of
    soft patterns into an MLP
    """
    def __init__(
            self,
            pattern_specs: MutableMapping[int, int],
            mlp_hidden_dim: int,
            num_mlp_layers: int,
            num_classes: int,
            embeddings: List[np.ndarray],
            vocab: Vocab,
            semiring: Semiring,
            bias_scale_param: float,
            gpu: bool = False,
            rnn: Union[Module, None] = None,
            pre_computed_patterns: Union[List, None] = None,
            no_sl: bool = False,
            shared_sl: int = 0,
            no_eps: bool = False,
            eps_scale: Union[float, None] = None,
            self_loop_scale: Union[torch.Tensor, float, None] = None) -> None:
        super(SoftPatternClassifier, self).__init__()
        self.semiring = semiring
        self.vocab = vocab
        self.embeddings = embeddings

        self.to_cuda = to_cuda(gpu)

        self.total_num_patterns = sum(pattern_specs.values())
        print(self.total_num_patterns, pattern_specs)
        self.rnn = rnn
        self.mlp = MLP(self.total_num_patterns, mlp_hidden_dim, num_mlp_layers,
                       num_classes)

        if self.rnn is None:
            self.word_dim = len(embeddings[0])
        else:
            self.word_dim = self.rnn.num_directions * self.rnn.hidden_dim
        self.num_diags = 1  # self-loops and single-forward-steps
        self.no_sl = no_sl
        self.shared_sl = shared_sl

        self.pattern_specs = pattern_specs
        self.max_pattern_length = max(list(pattern_specs.keys()))

        self.no_eps = no_eps
        self.bias_scale_param = bias_scale_param

        # Shared parameters between main path and self loop.
        # 1 -- one parameter per state per pattern
        # 2 -- a single global parameter
        if self.shared_sl > 0:
            if self.shared_sl == SHARED_SL_PARAM_PER_STATE_PER_PATTERN:
                shared_sl_data = randn(self.total_num_patterns,
                                       self.max_pattern_length)
            elif self.shared_sl == SHARED_SL_SINGLE_PARAM:
                shared_sl_data = randn(1)

            self.self_loop_scale = Parameter(shared_sl_data)
        elif not self.no_sl:
            if self_loop_scale is not None:
                self.self_loop_scale = self.semiring.from_float(
                    self.to_cuda(fixed_var(FloatTensor([self_loop_scale]))))
            else:
                self.self_loop_scale = self.to_cuda(fixed_var(semiring.one(1)))
            self.num_diags = 2

        # end state index for each pattern
        end_states = [[
            end
        ] for pattern_len, num_patterns in self.pattern_specs.items()
                      for end in num_patterns * [pattern_len - 1]]

        self.end_states = self.to_cuda(fixed_var(LongTensor(end_states)))

        diag_data_size = self.total_num_patterns * self.num_diags * self.max_pattern_length
        diag_data = randn(diag_data_size, self.word_dim)
        bias_data = randn(diag_data_size, 1)

        normalize(diag_data)

        if pre_computed_patterns is not None:
            diag_data, bias_data = self.load_pre_computed_patterns(
                pre_computed_patterns, diag_data, bias_data, pattern_specs)

        self.diags = Parameter(diag_data)

        # Bias term
        self.bias = Parameter(bias_data)

        if not self.no_eps:
            self.epsilon = Parameter(
                randn(self.total_num_patterns, self.max_pattern_length - 1))

            # TODO: learned? hyperparameter?
            # since these are currently fixed to `semiring.one`, they are not doing anything.
            if eps_scale is not None:
                self.epsilon_scale = self.semiring.from_float(
                    self.to_cuda(fixed_var(FloatTensor([eps_scale]))))
            else:
                self.epsilon_scale = self.to_cuda(fixed_var(semiring.one(1)))

        print("# params:", sum(p.nelement() for p in self.parameters()))

    def get_transition_matrices(
            self,
            batch: Batch,
            dropout: Union[Module, None] = None) -> List[torch.Tensor]:
        b = batch.size()
        n = batch.max_doc_len
        if self.rnn is None:
            transition_scores = \
                self.semiring.from_float(mm(self.diags, batch.embeddings_matrix) + self.bias_scale_param * self.bias).t()
            if dropout is not None and dropout:
                transition_scores = dropout(transition_scores)
            batched_transition_scores = [
                torch.index_select(transition_scores, 0, doc)
                for doc in batch.docs
            ]
            batched_transition_scores = torch.cat(
                batched_transition_scores).view(b, n, self.total_num_patterns,
                                                self.num_diags,
                                                self.max_pattern_length)

        else:
            # run an RNN to get the word vectors to input into our soft-patterns
            outs = self.rnn.forward(batch, dropout=dropout)
            padded, _ = pad_packed_sequence(outs, batch_first=True)
            padded = padded.contiguous().view(b * n, self.word_dim).t()

            if dropout is not None and dropout:
                padded = dropout(padded)

            batched_transition_scores = \
                self.semiring.from_float(mm(self.diags, padded) + self.bias_scale_param * self.bias).t()

            if dropout is not None and dropout:
                batched_transition_scores = dropout(batched_transition_scores)

            batched_transition_scores = \
                batched_transition_scores.contiguous().view(
                    b,
                    n,
                    self.total_num_patterns,
                    self.num_diags,
                    self.max_pattern_length
                )
        # transition matrix for each token idx
        transition_matrices = [
            batched_transition_scores[:, word_index, :, :, :]
            for word_index in range(n)
        ]
        return transition_matrices

    def load_pre_computed_patterns(
        self, pre_computed_patterns: List[Any], diag_data: torch.Tensor,
        bias_data: torch.Tensor, pattern_spec: MutableMapping[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loading a set of pre-coputed patterns into diagonal and bias arrays"""
        pattern_indices = dict((p, 0) for p in pattern_spec)

        # First,view diag_data and bias_data as 4/3d tensors
        diag_data_size = diag_data.size()[0]
        diag_data = diag_data.view(self.total_num_patterns, self.num_diags,
                                   self.max_pattern_length, self.word_dim)
        bias_data = bias_data.view(self.total_num_patterns, self.num_diags,
                                   self.max_pattern_length)

        n = 0

        # Pattern indices: which patterns are we loading?
        # the pattern index from which we start loading each pattern length.
        for (i, patt_len) in enumerate(pattern_spec.keys()):
            pattern_indices[patt_len] = n
            n += pattern_spec[patt_len]

        # Loading all pre-computed patterns
        for p in pre_computed_patterns:
            patt_len = len(p) + 1

            # Getting pattern index in diagonal data
            index = pattern_indices[patt_len]

            # Loading diagonal and bias for p
            diag, bias = self.load_pattern(p)

            # Updating diagonal and bias
            diag_data[index, 1, :(patt_len - 1), :] = diag
            bias_data[index, 1, :(patt_len - 1)] = bias

            # Updating pattern_indices
            pattern_indices[patt_len] += 1

        return diag_data.view(diag_data_size, self.word_dim), bias_data.view(
            diag_data_size, 1)

    def load_pattern(self,
                     patt: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loading diagonal and bias of one pattern"""
        diag = EPSILON * torch.randn(len(patt), self.word_dim)
        bias = torch.zeros(len(patt))

        factor = 10

        # Traversing elements of pattern.
        for (i, element) in enumerate(patt):
            # CW: high bias (we don't care about the identity of the token
            if element == CW_TOKEN:
                bias[i] = factor
            else:
                # Concrete word: we do care about the token (low bias).
                bias[i] = -factor

                # If we have a word vector for this element, the diagonal value if this vector
                if element in self.vocab:
                    diag[i] = FloatTensor(
                        factor * self.embeddings[self.vocab.index[element]])

        return diag, bias

    def forward(self,
                batch: Batch,
                debug: int = 0,
                dropout: Union[Module, None] = None) -> torch.Tensor:
        """ Calculate scores for one batch of documents. """
        time1 = monotonic()
        transition_matrices = self.get_transition_matrices(batch, dropout)
        time2 = monotonic()

        self_loop_scale = None

        if self.shared_sl:
            self_loop_scale = self.semiring.from_float(self.self_loop_scale)
        elif not self.no_sl:
            self_loop_scale = self.self_loop_scale

        batch_size = batch.size()
        num_patterns = self.total_num_patterns
        scores = self.to_cuda(
            fixed_var(self.semiring.zero(batch_size, num_patterns)))

        # to add start state for each word in the document.
        restart_padding = self.to_cuda(
            fixed_var(self.semiring.one(batch_size, num_patterns, 1)))

        zero_padding = self.to_cuda(
            fixed_var(self.semiring.zero(batch_size, num_patterns, 1)))

        eps_value = self.get_eps_value()

        batch_end_state_idxs = self.end_states.expand(batch_size, num_patterns,
                                                      1)
        hiddens = self.semiring.zero(batch_size, num_patterns,
                                     self.max_pattern_length)
        if isinstance(hiddens, torch.Tensor):
            hiddens = self.to_cuda(hiddens)
        else:
            hiddens = self.to_cuda(torch.tensor(hiddens))

        # set start state (0) to 1 for each pattern in each doc
        hiddens[:, :,
                0] = self.to_cuda(self.semiring.one(batch_size, num_patterns))

        if debug % 4 == 3:
            all_hiddens = [hiddens[0, :, :]]
        for i, transition_matrix in enumerate(transition_matrices):
            hiddens = self.transition_once(eps_value, hiddens,
                                           transition_matrix, zero_padding,
                                           restart_padding, self_loop_scale)
            if debug % 4 == 3:
                all_hiddens.append(hiddens[0, :, :])

            # Look at the end state for each pattern, and "add" it into score
            end_state_vals = torch.gather(hiddens, 2,
                                          batch_end_state_idxs).view(
                                              batch_size, num_patterns)
            # but only update score when we're not already past the end of the doc
            active_doc_idxs = torch.nonzero(torch.gt(batch.doc_lens,
                                                     i)).squeeze()
            scores[active_doc_idxs] = \
                self.semiring.plus(
                    scores[active_doc_idxs],
                    end_state_vals[active_doc_idxs]
                )

        if debug:
            time3 = monotonic()
            print("MM: {}, other: {}".format(round(time2 - time1, 3),
                                             round(time3 - time2, 3)))

        scores = self.semiring.to_float(scores)

        if debug % 4 == 3:
            return self.mlp.forward(scores), transition_matrices, all_hiddens
        elif debug % 4 == 1:
            return self.mlp.forward(scores), scores
        else:
            return self.mlp.forward(scores)

    def get_eps_value(self) -> Union[torch.Tensor, None]:
        return None if self.no_eps else self.semiring.times(
            self.epsilon_scale, self.semiring.from_float(self.epsilon))

    def transition_once(
            self, eps_value: Union[torch.Tensor, None], hiddens: torch.Tensor,
            transition_matrix_val: torch.Tensor, zero_padding: torch.Tensor,
            restart_padding: torch.Tensor,
            self_loop_scale: Union[torch.Tensor, float, None]) -> torch.Tensor:
        # Adding epsilon transitions (don't consume a token, move forward one state)
        # We do this before self-loops and single-steps.
        # We only allow zero or one epsilon transition in a row.
        if self.no_eps:
            after_epsilons = hiddens
        else:
            after_epsilons = \
                self.semiring.plus(
                    hiddens,
                    cat((zero_padding,
                         self.semiring.times(
                             hiddens[:, :, :-1],
                             eps_value  # doesn't depend on token, just state
                         )), 2)
                )

        after_main_paths = \
            cat((restart_padding,  # <- Adding the start state
                 self.semiring.times(
                     after_epsilons[:, :, :-1],
                     transition_matrix_val[:, :, -1, :-1])
                 ), 2)

        if self.no_sl:
            return after_main_paths
        else:
            self_loop_scale = cast(torch.Tensor, self_loop_scale)
            self_loop_scale = self_loop_scale.expand(transition_matrix_val[:, :, 0, :].size()) \
                if self.shared_sl == SHARED_SL_PARAM_PER_STATE_PER_PATTERN else self_loop_scale

            # Adding self loops (consume a token, stay in same state)
            after_self_loops = self.semiring.times(
                self_loop_scale,
                self.semiring.times(after_epsilons,
                                    transition_matrix_val[:, :, 0, :]))
            # either happy or self-loop, not both
            return self.semiring.plus(after_main_paths, after_self_loops)

    def predict(self, batch: Batch, debug: int = 0) -> List[int]:
        output = self.forward(batch, debug).data
        return [int(x) for x in argmax(output)]
