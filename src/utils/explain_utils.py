#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, Any, List, Union
from functools import total_ordering
import torch


def get_nearest_neighbors(weights: torch.Tensor,
                          embeddings: torch.Tensor,
                          threshold: int = 1000) -> torch.Tensor:
    return torch.argmax(torch.mm(weights, embeddings[:threshold, :]), dim=1)


def zip_lambda_2d(function: Callable, input_a: Any,
                  input_b: Any) -> List[List[Any]]:
    return [[function(x, y) for x, y in zip(X, Y)]
            for X, Y in zip(input_a, input_b)]


def cat_2d(padding: Any, data: Any) -> List[List[Any]]:
    return [[p] + x for p, x in zip(padding, data)]


@total_ordering
class BackPointer:
    def __init__(self, score: torch.Tensor, previous: Union['BackPointer',
                                                            None],
                 transition: Union[str, None], start_token_idx: int,
                 end_token_idx: int) -> None:
        self.score = score
        self.previous = previous
        self.transition = transition
        self.start_token_idx = start_token_idx
        self.end_token_idx = end_token_idx

    def __eq__(self, other: 'BackPointer') -> bool:
        return self.score == other.score

    def __ne__(self, other: 'BackPointer') -> bool:
        return not (self == other)

    def __lt__(self, other: 'BackPointer') -> bool:
        return self.score < other.score

    def __repr__(self) -> str:
        return ("BackPointer("
                "score={}, "
                "previous={}, "
                "transition={}, "
                "start_token_idx={}, "
                "end_token_idx={})").format(self.score, self.previous,
                                            self.transition,
                                            self.start_token_idx,
                                            self.end_token_idx)

    def display(self,
                doc_text: List[str],
                extra: str = "",
                num_padding_tokens: int = 0) -> str:
        if self.previous is None:
            return extra
        if self.transition == "self-loop":
            if self.end_token_idx >= len(doc_text):
                extra = "SL {:<15}".format(doc_text[-1]) + extra
            else:
                extra = "SL {:<15}".format(
                    doc_text[self.end_token_idx - 1 -
                             num_padding_tokens]) + extra
            return self.previous.display(doc_text,
                                         extra=extra,
                                         num_padding_tokens=num_padding_tokens)
        if self.transition == "happy path":
            if self.end_token_idx >= len(doc_text):
                extra = "HP {:<15}".format(doc_text[-1]) + extra
            else:
                extra = "HP {:<15}".format(
                    doc_text[self.end_token_idx - 1 -
                             num_padding_tokens]) + extra
            return self.previous.display(doc_text,
                                         extra=extra,
                                         num_padding_tokens=num_padding_tokens)
        extra = "ep {:<15}".format("") + extra
        return self.previous.display(doc_text,
                                     extra=extra,
                                     num_padding_tokens=num_padding_tokens)
