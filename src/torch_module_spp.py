#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import cast, Union, Any, List, Tuple
from collections import OrderedDict
from torch.nn import Module, Parameter, Linear, Dropout, init
from .utils.explain_utils import (pad_back_pointers, lambda_back_pointers,
                                  BackPointer)
from .utils.model_utils import Semiring, Batch
from .utils.data_utils import Vocab
import numbers
import torch


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
            ctx: Any, tau_threshold: float, input: Any) -> Any:
        ctx.save_for_backward(input)
        return (input > tau_threshold).float()

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:  # type: ignore
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return None, grad_input


class STE(Module):
    def __init__(self, tau_threshold: float = 0.) -> None:
        super(STE, self).__init__()
        self.tau_threshold = tau_threshold

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch = STEFunction.apply(self.tau_threshold, batch)
        return batch

    def extra_repr(self) -> str:
        return 'tau_threshold={}'.format(self.tau_threshold)


class MaskedLayerNorm(Module):
    # adapted from: https://yangkky.github.io/2020/03/16/masked-batch-norm.html
    def __init__(self,
                 normalized_shape: Union[int, List[int], torch.Size],
                 eps: float = 1e-5) -> None:
        super(MaskedLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape, )  # type: ignore
        self.normalized_shape = tuple(normalized_shape)  # type: ignore
        self.eps = eps
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)

    def forward(self,
                input: torch.Tensor,
                input_mask: Union[torch.Tensor, None] = None) -> torch.Tensor:
        if input_mask is None:
            return torch.nn.functional.layer_norm(
                input,
                self.normalized_shape,  # type: ignore
                self.weight,  # type: ignore
                self.bias,  # type: ignore
                self.eps)
        else:
            # clone masked input and cache hidden values
            masked = input.clone()
            cached = masked[~input_mask]
            divisor = input_mask.sum(dim=1, keepdim=True)

            # calculate masked mean
            masked[~input_mask] = 0
            mean = masked.sum(dim=1, keepdim=True) / divisor

            # calculate masked variance
            variance = (masked - mean)**2
            variance[~input_mask] = 0
            variance = variance.sum(dim=1, keepdim=True) / divisor

            # normalize inputs
            normalized = (masked - mean) / (torch.sqrt(variance + self.eps))
            normalized[~input_mask] = cached
            return normalized

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}'.format(**self.__dict__)


class SoftPatternClassifier(Module):
    def __init__(self,
                 pattern_specs: 'OrderedDict[int, int]',
                 num_classes: int,
                 embeddings: Module,
                 vocab: Vocab,
                 semiring: Semiring,
                 tau_threshold: float,
                 no_wildcards: bool = False,
                 bias_scale: float = 1.,
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
        self.normalizer = MaskedLayerNorm(self.total_num_patterns)
        self.binarizer = STE(tau_threshold)

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
        self.register_buffer("bias_scale",
                             torch.FloatTensor([bias_scale]),
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
            torch.cat(  # type: ignore
                (self.semiring.one(self.total_num_patterns, 1),
                 self.semiring.zero(self.total_num_patterns,
                                    self.max_pattern_length - 1)),
                axis=1),
            persistent=False)

    def get_wildcard_matrix(self) -> Union[torch.Tensor, None]:
        return None if self.no_wildcards else self.semiring.times(
            self.wildcard_scale.clone(),  # type: ignore
            self.semiring.from_outer_to_semiring(self.wildcards))

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

    def forward(self, batch: Batch, interim: bool = False) -> torch.Tensor:
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
        if interim:
            interim_scores = scores.clone()

        # extract scores from semiring to outer set
        scores = self.semiring.from_semiring_to_outer(scores)

        # extract all infinite indices
        isinf = torch.isinf(scores)

        if isinf.sum().item() > 0:
            scores_mask = ~isinf
        else:
            scores_mask = None  # type: ignore

        # execute normalization of scores
        scores = self.normalizer(scores, scores_mask)

        # binarize scores using STE
        scores = self.binarizer(scores)

        # conditionally return different tensors depending on routine
        if interim:
            interim_scores = torch.stack((interim_scores, scores), 1)
            return interim_scores
        else:
            return self.linear.forward(scores)

    def restart_padding_with_trace(self,
                                   token_index: int) -> List[BackPointer]:
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
                self.restart_padding.squeeze().tolist())  # type: ignore
        ]

    def transition_once_with_trace(
            self,
            hiddens: List[List[BackPointer]],
            transition_matrix: List[List[float]],
            wildcard_matrix: Union[List[List[float]], None],  # yapf: disable
            end_states: List[int],
            token_index: int) -> List[List[BackPointer]]:
        # add main transition; consume a token and state
        main_transitions = pad_back_pointers(
            self.restart_padding_with_trace(token_index),
            lambda_back_pointers(
                lambda back_pointer, transition_value: BackPointer(
                    raw_score=self.semiring.float_times(  # type: ignore
                        back_pointer.raw_score, transition_value),
                    binarized_score=0.,
                    pattern_index=back_pointer.pattern_index,
                    previous=back_pointer,
                    transition="main_transition",
                    start_token_index=back_pointer.start_token_index,
                    current_token_index=token_index,
                    end_token_index=token_index + 1),
                [hidden[:-1] for hidden in hiddens],
                transition_matrix,
                end_states))

        # return if no wildcards allowed
        if self.no_wildcards:
            return main_transitions
        else:
            # mypy typing fix
            wildcard_matrix = cast(List[List[float]], wildcard_matrix)
            # add wildcard transition; consume a generic token and state
            wildcard_transitions = pad_back_pointers(
                self.restart_padding_with_trace(token_index),
                lambda_back_pointers(
                    lambda back_pointer, wildcard_value: BackPointer(
                        raw_score=self.semiring.float_times(  # type: ignore
                            back_pointer.raw_score, wildcard_value),
                        binarized_score=0.,
                        pattern_index=back_pointer.pattern_index,
                        previous=back_pointer,
                        transition="wildcard_transition",
                        start_token_index=back_pointer.start_token_index,
                        current_token_index=token_index,
                        end_token_index=token_index + 1),
                    [hidden[:-1] for hidden in hiddens],
                    wildcard_matrix,
                    end_states))

            # return final object
            return lambda_back_pointers(
                self.semiring.float_plus,  # type: ignore
                main_transitions,
                wildcard_transitions,
                end_states)

    def forward_with_trace(
            self,
            batch: Batch,
            atol=1e-6) -> Tuple[List[List[BackPointer]], torch.Tensor]:
        # process all transition matrices
        transition_matrices_list = [
            transition_matrices[:index, :, :]
            for transition_matrices, index in zip(
                self.get_transition_matrices(batch), batch.doc_lens)
        ]

        # process all interim scores
        interim_scores_tensor = self.forward(batch, interim=True)

        # extract relevant interim scores
        interim_scores_list = [
            interim_scores for interim_scores in interim_scores_tensor
        ]

        # create local variables
        wildcard_matrix = self.get_wildcard_matrix().tolist()  # type: ignore
        end_states = self.end_states.squeeze().tolist()  # type: ignore
        end_state_back_pointers_list = []

        # loop over transition matrices and interim scores
        for transition_matrices, interim_scores in zip(
                transition_matrices_list, interim_scores_list):
            # construct hiddens from tensor to back pointers
            hiddens = self.hiddens.tolist()  # type: ignore
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
                back_pointers[end_state]
                for back_pointers, end_state in zip(hiddens, end_states)
            ]

            # iterate over sequence
            for token_index in range(transition_matrices.size(0)):
                transition_matrix = transition_matrices[
                    token_index, :, :].tolist()
                hiddens = self.transition_once_with_trace(
                    hiddens, transition_matrix, wildcard_matrix, end_states,
                    token_index)

                # extract end-states and compare with current bests
                end_state_back_pointers = [
                    self.semiring.float_plus(  # type: ignore
                        best_back_pointer, hidden_back_pointers[end_state])
                    for best_back_pointer, hidden_back_pointers, end_state in
                    zip(end_state_back_pointers, hiddens, end_states)
                ]

            # check that both explainability routine and model match
            assert torch.allclose(
                torch.FloatTensor([
                    back_pointer.raw_score
                    for back_pointer in end_state_back_pointers
                ]).to(interim_scores[0].device),
                interim_scores[0],
                atol=atol), ("Explainability routine does not produce "
                             "matching scores with SoPa++ routine")

            # assign binarized scores
            for pattern_index in range(
                    self.total_num_patterns):  # type: ignore
                end_state_back_pointers[
                    pattern_index].binarized_score = interim_scores[1][
                        pattern_index].item()

            # append end state back pointers to higher list
            end_state_back_pointers_list.append(end_state_back_pointers)

        # return best back pointers
        return end_state_back_pointers_list, self.linear(
            interim_scores_tensor[:, -1, :])
