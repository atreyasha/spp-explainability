#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Iterable, Callable, List
from .data_utils import (UNK_IDX, START_TOKEN_IDX, END_TOKEN_IDX, Vocab,
                         identity)
import numpy as np
import torch


def chunked(xs: Iterable, chunk_size: int) -> List:
    xs = list(xs)
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


def decreasing_length(xs: Iterable) -> List:
    return sorted(list(xs), key=lambda x: len(x[0]), reverse=True)


def chunked_sorted(xs: Iterable, chunk_size: int) -> List:
    return chunked(decreasing_length(xs), chunk_size)


def shuffled_chunked_sorted(xs: Iterable, chunk_size: int) -> List:
    chunks = chunked_sorted(xs, chunk_size)
    np.random.shuffle(chunks)
    return chunks


def right_pad(xs: Iterable, min_len: int, pad_element: str) -> List:
    return xs + [pad_element] * (min_len - len(xs))


def to_cuda(gpu: bool) -> Callable:
    return (lambda v: v.cuda()) if gpu else identity


def fixed_var(tensor: torch.Tensor) -> torch.Tensor:
    return torch.autograd.Variable(tensor, requires_grad=False)


def argmax(output: torch.Tensor) -> torch.Tensor:
    _, am = torch.max(output, 1)
    return am


def normalize(data: torch.Tensor) -> None:
    length = data.size()[0]
    for i in range(length):
        data[i] = data[i] / torch.norm(data[i])


def enable_gradient_clipping(model: torch.nn.Module, clip: float) -> None:
    if clip is not None and clip > 0:
        clip_function = lambda grad: grad.clamp(-clip, clip)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(clip_function)


def neg_infinity(*sizes: int) -> torch.Tensor:
    return -100 * torch.ones(*sizes)


class Batch:
    def __init__(self,
                 docs: List[List[int]],
                 embeddings: List[np.ndarray],
                 cuda: Callable,
                 word_dropout: float = 0,
                 max_len: int = -1) -> None:
        mini_vocab = Vocab.from_docs(docs,
                                     default=UNK_IDX,
                                     start=START_TOKEN_IDX,
                                     end=END_TOKEN_IDX)
        # limit maximum document length (for efficiency reasons).
        if max_len != -1:
            docs = [doc[:max_len] for doc in docs]
        doc_lens = [len(doc) for doc in docs]
        self.doc_lens = cuda(torch.LongTensor(doc_lens))
        self.max_doc_len = max(doc_lens)
        if word_dropout:
            # for each token, with probability `word_dropout`
            # replace word index with UNK_IDX.
            docs = [[
                UNK_IDX if np.random.rand() < word_dropout else x for x in doc
            ] for doc in docs]
        # pad docs so they all have the same length.
        # we pad with UNK, whose embedding is 0
        # so it doesn't mess up sums or averages.
        docs = [
            right_pad(mini_vocab.numberize(doc), self.max_doc_len, UNK_IDX)
            for doc in docs
        ]
        self.docs = [cuda(fixed_var(torch.LongTensor(doc))) for doc in docs]
        local_embeddings = [embeddings[i] for i in mini_vocab.names]
        self.embeddings_matrix = cuda(
            fixed_var(torch.FloatTensor(local_embeddings).t()))

    def size(self) -> int:
        return len(self.docs)


class Semiring:
    def __init__(self, zero: Any, one: Any, plus: Any, times: Any,
                 from_float: Any, to_float: Any) -> None:
        self.zero = zero
        self.one = one
        self.plus = plus
        self.times = times
        self.from_float = from_float
        self.to_float = to_float


ProbSemiring = Semiring(torch.zeros, torch.ones, torch.add, torch.mul,
                        torch.sigmoid, identity)

MaxPlusSemiring = Semiring(neg_infinity, torch.zeros, torch.max, torch.add,
                           identity, identity)

LogSpaceMaxTimesSemiring = Semiring(neg_infinity, torch.zeros, torch.max,
                                    torch.add,
                                    lambda x: torch.log(torch.sigmoid(x)),
                                    torch.exp)
