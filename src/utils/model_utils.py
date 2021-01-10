#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, List, Union, Any, Tuple
from .data_utils import (PAD_TOKEN_INDEX, START_TOKEN_INDEX, END_TOKEN_INDEX,
                         UNK_TOKEN_INDEX, Vocab, identity)
import numpy as np
import datetime
import torch

# define numerical epsilon to prevent negative log values
LOG_EPSILON = 1e-6


def timestamp() -> str:
    return str(int(datetime.datetime.now().timestamp()))


def chunked(xs: List[Tuple[List[int], int]],
            chunk_size: int) -> List[List[Tuple[List[int], int]]]:
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


def decreasing_length(
        xs: List[Tuple[List[int], int]]) -> List[Tuple[List[int], int]]:
    return sorted(list(xs), key=lambda x: len(x[0]), reverse=True)


def chunked_sorted(xs: List[Tuple[List[int], int]],
                   chunk_size: int) -> List[List[Tuple[List[int], int]]]:
    return chunked(decreasing_length(xs), chunk_size)


def shuffled_chunked_sorted(
        xs: List[Tuple[List[int], int]],
        chunk_size: int) -> List[List[Tuple[List[int], int]]]:
    shuffled_xs = xs.copy()
    np.random.shuffle(shuffled_xs)
    chunks = chunked_sorted(shuffled_xs, chunk_size)
    np.random.shuffle(chunks)
    return chunks


def right_pad(xs: List[Any], min_len: int, pad_element: Any) -> List[Any]:
    return xs + [pad_element] * (min_len - len(xs))


def to_cuda(gpu_device: Union[torch.device, None]) -> Callable:
    return (lambda v: v.to(gpu_device)) if gpu_device is not None else identity


def enable_gradient_clipping(model: torch.nn.Module,
                             clip_threshold: Union[float, None]) -> None:
    if clip_threshold is not None and clip_threshold > 0:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(lambda grad: torch.clamp(
                    grad, -clip_threshold, clip_threshold))


def neg_infinity(*sizes: int) -> torch.Tensor:
    return -100 * torch.ones(*sizes)


class Batch:
    def __init__(self,
                 docs: List[List[int]],
                 embeddings: torch.nn.Module,
                 to_cuda: Callable,
                 word_dropout: float = 0,
                 max_doc_len: int = -1) -> None:
        mini_vocab = Vocab.from_docs(docs,
                                     pad=PAD_TOKEN_INDEX,
                                     start=START_TOKEN_INDEX,
                                     end=END_TOKEN_INDEX,
                                     unknown=UNK_TOKEN_INDEX)
        # limit maximum document length (for efficiency reasons).
        if max_doc_len != -1:
            docs = [doc[:max_doc_len] for doc in docs]
        doc_lens = [len(doc) for doc in docs]
        self.doc_lens = to_cuda(torch.LongTensor(doc_lens))
        self.max_doc_len = max(doc_lens)
        if word_dropout:
            # for each token, with probability `word_dropout`
            # replace word index with UNK_TOKEN_INDEX.
            docs = [[
                UNK_TOKEN_INDEX if (np.random.rand() < word_dropout) and
                (x not in [START_TOKEN_INDEX, END_TOKEN_INDEX]) else x
                for x in doc
            ] for doc in docs]
        # pad docs so they all have the same length.
        # we pad with UNK, whose embedding is 0
        # so it doesn't mess up sums or averages.
        docs = [
            right_pad(mini_vocab.numberize(doc), self.max_doc_len,
                      PAD_TOKEN_INDEX) for doc in docs
        ]
        self.docs = [to_cuda(torch.LongTensor(doc)) for doc in docs]
        self.local_embeddings = embeddings(
            to_cuda(torch.LongTensor(mini_vocab.names))).t()

    def size(self) -> int:
        return len(self.docs)


class Semiring:
    def __init__(self,
                 zero: Callable[..., torch.Tensor],
                 one: Callable[..., torch.Tensor],
                 plus: Callable[..., torch.Tensor],
                 times: Callable[..., torch.Tensor],
                 from_outer_to_semiring: Callable[..., torch.Tensor],
                 from_semiring_to_outer: Callable[..., torch.Tensor]) -> None:  # yapf: disable
        self.zero = zero
        self.one = one
        self.plus = plus
        self.times = times
        self.from_outer_to_semiring = from_outer_to_semiring
        self.from_semiring_to_outer = from_semiring_to_outer


ProbabilitySemiring = Semiring(torch.zeros, torch.ones, torch.add, torch.mul,
                               torch.sigmoid, identity)

MaxSumSemiring = Semiring(neg_infinity, torch.zeros, torch.max, torch.add,
                          identity, identity)

LogSpaceMaxProductSemiring = Semiring(
    neg_infinity, torch.zeros, torch.max, torch.add,
    lambda x: torch.log(torch.sigmoid(x) + LOG_EPSILON), torch.exp)
