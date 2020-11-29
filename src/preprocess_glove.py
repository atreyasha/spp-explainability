#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple
from itertools import islice
from .utils.data_utils import (START_TOKEN_IDX, END_TOKEN_IDX, START_TOKEN,
                               END_TOKEN, Vocab, is_printable, pad)
import numpy as np


def read_embeddings(filename: str,
                    fixed_vocab=None,
                    max_vocab_size: int = None) -> Tuple[Vocab, List, int]:
    print("Reading", filename)
    dim, has_header = check_dim_and_header(filename)
    # assign unknown, start and end tokens to zero vector
    unk_vec = np.zeros(dim)
    left_pad_vec = np.zeros(dim)
    right_pad_vec = np.zeros(dim)
    with open(filename, encoding='utf-8') as input_file:
        if has_header:
            input_file.readline()
        word_vecs = ((word, np.fromstring(vec_str, dtype=float, sep=' '))
                     for word, vec_str in (line.rstrip().split(" ", 1)
                                           for line in input_file)
                     if is_printable(word) and (
                         fixed_vocab is None or word in fixed_vocab))
        if max_vocab_size is not None:
            word_vecs = islice(word_vecs, max_vocab_size - 1)
        word_vecs = list(word_vecs)
    print("Done reading", len(word_vecs), "vectors of dimension", dim)
    vocab = Vocab((word for word, _ in word_vecs))
    # prepend special embeddings to (normalized) word embeddings
    vecs = [unk_vec, left_pad_vec, right_pad_vec
            ] + [vec / np.linalg.norm(vec) for _, vec in word_vecs]
    return vocab, vecs, dim


def check_dim_and_header(filename: str) -> Tuple[int, bool]:
    with open(filename, encoding='utf-8') as input_file:
        first_line = input_file.readline().rstrip().split()
        if len(first_line) == 2:
            return int(first_line[1]), True
        else:
            return len(first_line) - 1, False


def read_docs(filename: str,
              vocab: Vocab,
              num_padding_tokens: int = 1) -> Tuple[List[int], List[str]]:
    with open(filename, encoding='ISO-8859-1') as input_file:
        docs = [line.rstrip().split() for line in input_file]
    return ([
        pad(vocab.numberize(doc),
            num_padding_tokens=num_padding_tokens,
            START=START_TOKEN_IDX,
            END=END_TOKEN_IDX) for doc in docs
    ], [
        pad(doc,
            num_padding_tokens=num_padding_tokens,
            START=START_TOKEN,
            END=END_TOKEN) for doc in docs
    ])


def read_labels(filename: str) -> List[int]:
    with open(filename) as input_file:
        return [int(line.rstrip()) for line in input_file]
