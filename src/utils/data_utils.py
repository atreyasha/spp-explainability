#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Iterable, Callable, Generator, List, Union
from itertools import chain
import string

PRINTABLE = set(string.printable)
UNK_TOKEN = "*UNK*"
START_TOKEN = "*START*"
END_TOKEN = "*END*"
UNK_IDX = 0
START_TOKEN_IDX = 1
END_TOKEN_IDX = 2


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


def is_printable(word: str) -> bool:
    return all(c in PRINTABLE for c in word)


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


def vocab_from_text(filename: str) -> Vocab:
    with open(filename, encoding='ISO-8859-1') as input_file:
        return Vocab.from_docs(line.rstrip().split() for line in input_file)


def pad(doc: List[Union[int, str]],
        num_padding_tokens: int = 1,
        START: Union[int, str] = START_TOKEN_IDX,
        END: Union[int, str] = END_TOKEN_IDX) -> List[Union[int, str]]:
    return ([START] * num_padding_tokens) + doc + ([END] * num_padding_tokens)
