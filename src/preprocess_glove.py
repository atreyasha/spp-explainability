#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Iterable, Tuple, Union
from utils import nub
from itertools import chain, islice
import numpy as np
import string

# define global variables
PRINTABLE = set(string.printable)
UNK_TOKEN = "*UNK*"
START_TOKEN = "*START*"
END_TOKEN = "*END*"
UNK_IDX = 0
START_TOKEN_IDX = 1
END_TOKEN_IDX = 2


def is_printable(word: str) -> bool:
    return all(c in PRINTABLE for c in word)


class Vocab:
    def __init__(self,
                 names: Iterable,
                 default: str = UNK_TOKEN,
                 start: int = START_TOKEN,
                 end: int = END_TOKEN) -> None:
        self.default = default
        self.names = list(nub(chain([default, start, end], names)))
        self.index = {name: i for i, name in enumerate(self.names)}

    def __getitem__(self, index: int) -> str:
        return self.names[index] if 0 < index < len(
            self.names) else self.default

    def __call__(self, name: str) -> int:
        return self.index.get(name, UNK_IDX)

    def __contains__(self, item: str) -> bool:
        return item in self.index

    def __len__(self) -> int:
        return len(self.names)

    def __or__(self, other: 'Vocab') -> 'Vocab':
        return Vocab(self.names + other.names)

    def numberize(self, doc: List[str]) -> List[int]:
        return [self(token) for token in doc]

    def denumberize(self, doc: List[int]) -> List[str]:
        return [self[idx] for idx in doc]

    @staticmethod
    def from_docs(docs: List[List[str]],
                  default: str = UNK_TOKEN,
                  start: str = START_TOKEN,
                  end: str = END_TOKEN) -> 'Vocab':
        return Vocab((i for doc in docs for i in doc),
                     default=default,
                     start=start,
                     end=end)


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
    with open(filename, encoding='utf-8') as input_file:
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


def vocab_from_text(filename: str) -> Vocab:
    with open(filename, encoding='utf-8') as input_file:
        return Vocab.from_docs(line.rstrip().split() for line in input_file)


def pad(doc: List[Union[int, str]],
        num_padding_tokens: int = 1,
        START: Union[int, str] = START_TOKEN_IDX,
        END: Union[int, str] = END_TOKEN_IDX) -> List[Union[int, str]]:
    return ([START] * num_padding_tokens) + doc + ([END] * num_padding_tokens)
