#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, List, Union, Any
from functools import total_ordering
import re


def lambda_back_pointers(function: Callable[..., Any],
                         input_X: List[List['BackPointer']],
                         input_Y: Union[List[List[float]],
                                        List[List['BackPointer']]],
                         end_states: List[int]) -> List[List['BackPointer']]:
    return [
        [
            function(x, y) if j <= end_states[i] else x
            for j, (x, y) in enumerate(zip(X, Y))  # type: ignore
        ] for i, (X, Y) in enumerate(zip(input_X, input_Y))  # type: ignore
    ]


def pad_back_pointers(
        padding: List['BackPointer'],
        data: List[List['BackPointer']]) -> List[List['BackPointer']]:
    return [[pad] + data_element for pad, data_element in zip(padding, data)]


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

    def __eq__(self, other: 'BackPointer') -> bool:  # type: ignore
        if not isinstance(other, BackPointer):
            return NotImplemented
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

    def get_text(  # type: ignore
            self,
            doc_text: List[str],
            extra: List[str] = []) -> List[str]:
        if self.previous is None:
            return extra
        else:
            extra = [doc_text[self.current_token_index]] + extra
            return self.previous.get_text(doc_text, extra=extra)

    def get_regex(  # type: ignore
            self,
            doc_text: List[str],
            extra: List[str] = []) -> List[str]:
        if self.previous is None:
            return extra
        if self.transition == "main_transition":
            extra = [re.escape(doc_text[self.current_token_index])] + extra
            return self.previous.get_regex(doc_text, extra=extra)
        if self.transition == "wildcard_transition":
            extra = ["[^\\s]+"] + extra
            return self.previous.get_regex(doc_text, extra=extra)
