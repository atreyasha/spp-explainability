#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import (Any, Iterable, Callable, Generator, List, Union, Tuple,
                    Sequence, overload)
from itertools import chain, islice
import numpy as np
import string

PRINTABLE = set(string.printable)
UNK_TOKEN = "[UNK]"
START_TOKEN = "[START]"
END_TOKEN = "[END]"
UNK_INDEX = 0
START_TOKEN_INDEX = 1
END_TOKEN_INDEX = 2


def identity(x: Any) -> Any:
    return x


def unique(xs: Iterable[Any]) -> Generator[Any, None, None]:
    return unique_by(xs, identity)


def unique_by(xs: Iterable[Any], key: Callable) -> Generator[Any, None, None]:
    seen = set()

    def check_and_add(x: Any) -> bool:
        k = key(x)
        if k not in seen:
            seen.add(k)
            return True
        return False

    return (x for x in xs if check_and_add(x))


class Vocab:
    def __init__(self,
                 names: Iterable[Any],
                 default: Union[int, str] = UNK_TOKEN,
                 start: Union[int, str] = START_TOKEN,
                 end: Union[int, str] = END_TOKEN) -> None:
        self.default = default
        self.names = list(unique(chain([default, start, end], names)))
        self.index = {name: i for i, name in enumerate(self.names)}

    def __getitem__(self, index: int) -> Union[int, str]:
        return self.names[index] if 0 < index < len(
            self.names) else self.default

    def __call__(self, name: Union[int, str]) -> int:
        return self.index.get(name, UNK_INDEX)

    def __contains__(self, item: str) -> bool:
        return item in self.index

    def __len__(self) -> int:
        return len(self.names)

    def __or__(self, other: 'Vocab') -> 'Vocab':
        return Vocab(self.names + other.names)

    def numberize(self, doc: Sequence[Union[int, str]]) -> List[int]:
        return [self(token) for token in doc]

    def denumberize(self, doc: List[int]) -> Sequence[Union[int, str]]:
        return [self[index] for index in doc]

    @classmethod
    def from_docs(cls,
                  docs: Sequence[Sequence[Union[int, str]]],
                  default: Union[int, str] = UNK_TOKEN,
                  start: Union[int, str] = START_TOKEN,
                  end: Union[int, str] = END_TOKEN) -> 'Vocab':
        return cls((i for doc in docs for i in doc),
                   default=default,
                   start=start,
                   end=end)

    @classmethod
    def from_vocab_file(cls,
                        filename: str,
                        default: Union[int, str] = UNK_TOKEN,
                        start: Union[int, str] = START_TOKEN,
                        end: Union[int, str] = END_TOKEN) -> 'Vocab':
        vocab = cls([], default=default, start=start, end=end)
        with open(filename, "r") as input_file_stream:
            vocab.index = {
                line.strip(): index
                for index, line in enumerate(input_file_stream)
            }
        vocab.names = list(zip(*sorted([(key, value)
                                        for key, value in vocab.index.items()],
                                       key=lambda x: x[1])))[0]
        return vocab


def is_printable(word: str) -> bool:
    return all(c in PRINTABLE for c in word)


def read_embeddings(
    filename: str,
    fixed_vocab: Union[Vocab, None] = None,
    max_vocab_size: Union[int, None] = None
) -> Tuple[Vocab, List[np.ndarray], int]:
    dim, has_header = check_dim_and_header(filename)
    # assign unknown, start and end tokens to zero vector
    unk_vec = np.zeros(dim)
    left_pad_vec = np.zeros(dim)
    right_pad_vec = np.zeros(dim)
    with open(filename, 'r', encoding='utf-8') as input_file_stream:
        if has_header:
            input_file_stream.readline()
        word_vecs: Iterable[Tuple[str, np.ndarray]] = (
            (word, np.fromstring(vec_str, dtype=float, sep=' '))
            for word, vec_str in (line.rstrip().split(" ", 1)
                                  for line in input_file_stream)
            if is_printable(word) and (
                fixed_vocab is None or word in fixed_vocab))
        if max_vocab_size is not None:
            word_vecs = islice(word_vecs, max_vocab_size - 1)
        word_vecs = list(word_vecs)
    vocab = Vocab((word for word, _ in word_vecs))
    # prepend special embeddings to (normalized) word embeddings
    vecs = [unk_vec, left_pad_vec, right_pad_vec
            ] + [vec / np.linalg.norm(vec) for _, vec in word_vecs]
    return vocab, vecs, dim


def check_dim_and_header(filename: str) -> Tuple[int, bool]:
    with open(filename, 'r', encoding='utf-8') as input_file_stream:
        first_line = input_file_stream.readline().rstrip().split()
        if len(first_line) == 2:
            return int(first_line[1]), True
        else:
            return len(first_line) - 1, False


def read_docs(
    filename: str,
    vocab: Vocab,
    num_padding_tokens: int = 1
) -> Tuple[Sequence[Sequence[int]], Sequence[Sequence[Union[int, str]]]]:
    with open(filename, 'r', encoding='utf-8') as input_file_stream:
        docs = [line.rstrip().split() for line in input_file_stream]
    return ([
        pad(vocab.numberize(doc),
            num_padding_tokens=num_padding_tokens,
            START=START_TOKEN_INDEX,
            END=END_TOKEN_INDEX) for doc in docs
    ], [
        pad(doc,
            num_padding_tokens=num_padding_tokens,
            START=START_TOKEN,
            END=END_TOKEN) for doc in docs
    ])


def read_labels(filename: str) -> List[int]:
    with open(filename, 'r', encoding='utf-8') as input_file_stream:
        return [int(line.rstrip()) for line in input_file_stream]


def vocab_from_text(filename: str) -> Vocab:
    with open(filename, 'r', encoding='utf-8') as input_file_stream:
        return Vocab.from_docs(
            [line.rstrip().split() for line in input_file_stream])


@overload
def pad(doc: Sequence[int], num_padding_tokens: int, START: int,
        END: int) -> Sequence[int]:
    ...


@overload
def pad(doc: Sequence[str], num_padding_tokens: int, START: str,
        END: str) -> Sequence[str]:
    ...


def pad(doc: Sequence[Union[int, str]],
        num_padding_tokens: int = 1,
        START: Union[int, str] = START_TOKEN_INDEX,
        END: Union[int, str] = END_TOKEN_INDEX) -> Sequence[Union[int, str]]:
    return ([START] * num_padding_tokens) + list(doc) + ([END] *
                                                         num_padding_tokens)
