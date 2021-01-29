#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, List, Union
from functools import total_ordering
import torch


def zip_lambda_nested(function: Callable, input_a: List[List['BackPointer']],
                      input_b: torch.Tensor) -> List[List['BackPointer']]:
    return [[function(x, y) for x, y in zip(X, Y)]
            for X, Y in zip(input_a, input_b)]


def cat_nested(padding: List[List['BackPointer']],
               data: List[List['BackPointer']]) -> List[List['BackPointer']]:
    return [[pad] + data_element for pad, data_element in zip(padding, data)]


def torch_apply_float_function(function: Callable[..., torch.Tensor],
                               input_a: float, input_b: float) -> float:
    return function(torch.FloatTensor([input_a]),
                    torch.FloatTensor([input_b])).item()


@total_ordering
class BackPointer:
    def __init__(self, raw_score: float, binarized_score: float,
                 pattern_index: int, previous: Union['BackPointer', None],
                 transition: Union[str, None], start_token_index: int,
                 current_token_index: int, end_token_index: int) -> None:
        self.raw_score = raw_score
        self.binarized_score = binarized_score
        self.pattern_index = pattern_index
        self.previous = previous
        self.transition = transition
        self.start_token_index = start_token_index
        self.current_token_index = current_token_index
        self.end_token_index = end_token_index

    def __eq__(self, other: 'BackPointer') -> bool:
        return self.raw_score == other.raw_score

    def __lt__(self, other: 'BackPointer') -> bool:
        return self.raw_score < other.raw_score

    def __repr__(self) -> str:
        return ("BackPointer("
                "raw_score={}, "
                "binarized_score={}, "
                "pattern_index={}, "
                "previous={}, "
                "transition={}, "
                "start_token_index={}, "
                "current_token_index={}, "
                "end_token_index={})").format(
                    self.raw_score, self.binarized_score, self.pattern_index,
                    self.previous, self.transition, self.start_token_index,
                    self.current_token_index, self.end_token_index)

    def display(self,
                doc_text: List[str],
                extra: List[str] = [],
                num_padding_tokens: int = 1) -> str:
        if self.previous is None:
            return " ".join(extra)
        if self.transition == "main_transition":
            extra = [doc_text[self.current_token_index]] + extra
            return self.previous.display(doc_text, extra=extra)
        if self.transition == "wildcard_transition":
            extra = ["*"] + extra
            return self.previous.display(doc_text, extra=extra)
