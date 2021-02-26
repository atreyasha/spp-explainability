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


def chunked(inputs: List[Any], chunk_size: int) -> List[List[Any]]:
    return [
        inputs[i:i + chunk_size] for i in range(0, len(inputs), chunk_size)
    ]


def decreasing_length(
        inputs: List[Tuple[List[int], int]]) -> List[Tuple[List[int], int]]:
    return sorted(list(inputs), key=lambda x: len(x[0]), reverse=True)


def chunked_sorted(inputs: List[Tuple[List[int], int]],
                   chunk_size: int) -> List[List[Tuple[List[int], int]]]:
    return chunked(decreasing_length(inputs), chunk_size)


def shuffled_chunked_sorted(
        inputs: List[Tuple[List[int], int]],
        chunk_size: int) -> List[List[Tuple[List[int], int]]]:
    shuffled_inputs = inputs.copy()
    np.random.shuffle(shuffled_inputs)
    chunks = chunked_sorted(shuffled_inputs, chunk_size)
    np.random.shuffle(chunks)
    return chunks


def right_pad(inputs: List[Any], min_len: int, pad_element: Any) -> List[Any]:
    return inputs + [pad_element] * (min_len - len(inputs))


def to_cuda(gpu_device: Union[torch.device, None]) -> Callable:
    return (lambda v: v.to(gpu_device)) if gpu_device is not None else identity


def neg_infinity(*sizes: int) -> torch.Tensor:
    return float("-inf") * torch.ones(*sizes)


def enable_gradient_clipping(model: torch.nn.Module,
                             clip_threshold: Union[float, None]) -> None:
    if clip_threshold is not None and clip_threshold > 0:
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.register_hook(lambda grad: torch.clamp(
                    grad, -clip_threshold, clip_threshold))


def torch_exp_masked(input: torch.Tensor) -> torch.Tensor:
    output = torch.exp(input).clone()
    output[output == 0.] = float("-inf")
    return output


class Batch:
    def __init__(self,
                 docs: List[List[int]],
                 embeddings: torch.nn.Module,
                 to_cuda: Callable,
                 word_dropout: float = 0.,
                 max_doc_len: Union[int, None] = None) -> None:
        mini_vocab = Vocab.from_docs(docs,
                                     pad=PAD_TOKEN_INDEX,
                                     start=START_TOKEN_INDEX,
                                     end=END_TOKEN_INDEX,
                                     unknown=UNK_TOKEN_INDEX)
        # limit maximum document length (for efficiency reasons).
        if max_doc_len is not None:
            docs = [doc[:max_doc_len] for doc in docs]
        doc_lens = [len(doc) for doc in docs]
        self.doc_lens = to_cuda(torch.LongTensor(doc_lens))
        self.max_doc_len = max(doc_lens)
        if word_dropout:
            # replace word index with UNK_TOKEN_INDEX
            docs = [[
                UNK_TOKEN_INDEX if (np.random.rand() < word_dropout) and
                (x not in [START_TOKEN_INDEX, END_TOKEN_INDEX]) else x
                for x in doc
            ] for doc in docs]
        # pad docs so they all have the same length.
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
                 addition: Callable[..., torch.Tensor],
                 multiplication: Callable[..., torch.Tensor],
                 float_addition: Callable[..., Any],
                 float_multiplication: Callable[..., Any],
                 from_outer_to_semiring: Callable[..., torch.Tensor],
                 from_semiring_to_outer: Callable[..., torch.Tensor]) -> None:  # yapf: disable
        self.zero = zero
        self.one = one
        self.addition = addition
        self.multiplication = multiplication
        self.float_addition = float_addition
        self.float_multiplication = float_multiplication
        self.from_outer_to_semiring = from_outer_to_semiring
        self.from_semiring_to_outer = from_semiring_to_outer


MaxSumSemiring = Semiring(neg_infinity, torch.zeros, torch.max, torch.add,
                          lambda x, y: max(x, y), lambda x, y: x + y, identity,
                          identity)

LogSpaceMaxProductSemiring = Semiring(
    neg_infinity, torch.zeros, torch.max, torch.add, lambda x, y: max(x, y),
    lambda x, y: x + y, lambda x: torch.log(torch.sigmoid(x) + LOG_EPSILON),
    torch_exp_masked)
