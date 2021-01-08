#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Any, Dict, Generator, Iterable, Tuple
from nltk import word_tokenize
import csv


def read_tsv(filename: str) -> List[Any]:
    with open(filename, 'r', encoding='utf-8') as input_file_stream:
        raw_list = list(csv.reader(input_file_stream, delimiter='\t'))
    return raw_list


def mapping(classes: List[str]) -> Dict[str, int]:
    return {element: i for i, element in enumerate(sorted(set(classes)))}


def serialize(data: Iterable[str],
              mapping: Dict[str, int]) -> Generator[int, None, None]:
    return (mapping[element] for element in data)


def tokenize(data: Iterable[str]) -> Generator[str, None, None]:
    return (" ".join(word_tokenize(element)) for element in data)


def lowercase(data: Iterable[str]) -> Generator[str, None, None]:
    return (element.lower() for element in data)


def repeat_items(data: List[str], count: int) -> List[str]:
    # source: https://stackoverflow.com/a/54864336
    return data * (count // len(data)) + data[:(count % len(data))]


def categorize_by_label(
        full_data: List[Tuple[str, int]]) -> Dict[int, List[str]]:
    label_data_mapping = {}
    for data, label in full_data:
        if label not in label_data_mapping:
            label_data_mapping[label] = [data]
        else:
            label_data_mapping[label].append(data)
    return label_data_mapping


def upsample(full_data: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    # create label to data mapping
    label_data_mapping = categorize_by_label(full_data)

    # find maximum length to upsample
    maximum_length = max([len(value) for value in label_data_mapping.values()])

    # loop through each key and upsample to maximum length
    for key in label_data_mapping.keys():
        if len(label_data_mapping[key]) < maximum_length:
            label_data_mapping[key] = repeat_items(label_data_mapping[key],
                                                   maximum_length)

    # convert dicionary back to list of tuples and return it
    return [(text, key) for key, value in label_data_mapping.items()
            for text in value]
