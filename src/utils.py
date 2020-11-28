#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Iterable, Callable, List, Generator
import numpy as np


def identity(x: Any) -> Any:
    return x


def nub(xs: Iterable) -> Generator:
    return nub_by(xs, identity)


def nub_by(xs: Iterable, key: Callable) -> Generator:
    seen = set()

    def check_and_add(x: Any) -> bool:
        k = key(x)
        if k not in seen:
            seen.add(k)
            return True
        return False

    return (x for x in xs if check_and_add(x))


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
