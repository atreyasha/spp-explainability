#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
from typing import List, Iterable, Dict, Any, Union
from nltk import word_tokenize
from .utils.parser_utils import argparse_formatter
from .utils.logging_utils import make_logger
from .arg_parser import preprocess_arg_parser, logging_arg_parser
import argparse
import json
import csv
import os


def read_tsv(filename: str) -> List[Any]:
    with open(filename, 'r', encoding='utf-8') as input_file_stream:
        raw_list = list(csv.reader(input_file_stream, delimiter='\t'))
    return raw_list


def mapping(classes: List[str]) -> Dict[str, int]:
    return {element: i for i, element in enumerate(sorted(set(classes)))}


def serialize(data: List[str], mapping: Dict[str, int]) -> List[int]:
    return [mapping[element] for element in data]


def tokenize(data: List[str]) -> List[str]:
    return [
        " ".join(word_tokenize(element))
        for element in tqdm(data, disable=DISABLE_TQDM)
    ]


def make_unique(full_data: Iterable[Any]) -> List[Any]:
    unique_list = []
    for element in tqdm(full_data, disable=DISABLE_TQDM):
        if element not in unique_list:
            unique_list.append(element)
    return unique_list


def write_file(full_data: List[List[Union[str, int]]], mapping: Dict[str, int],
               prefix: str, write_directory: str) -> None:
    # make write directory if it does not exist
    os.makedirs(write_directory, exist_ok=True)
    # split compund data into two
    data, labels = zip(*sorted(full_data))
    # write data
    with open(os.path.join(write_directory, prefix + ".data"),
              'w') as output_file_stream:
        for item in data:
            output_file_stream.write("%s\n" % item)
    # write labels
    with open(os.path.join(write_directory, prefix + ".labels"),
              'w') as output_file_stream:
        for item in labels:
            output_file_stream.write("%s\n" % str(item))


def main(args: argparse.Namespace) -> None:
    # define key directory and files
    write_directory = os.path.join(args.data_directory, "clean")
    train = os.path.join(args.data_directory, "raw", "en", "train-en.tsv")
    valid = os.path.join(args.data_directory, "raw", "en", "eval-en.tsv")
    test = os.path.join(args.data_directory, "raw", "en", "test-en.tsv")

    # process files
    LOGGER.info("Reading input data")
    train = read_tsv(train)
    valid = read_tsv(valid)
    test = read_tsv(test)

    # extract data and labels
    train_data, train_labels = zip(*[[element[2], element[0]]
                                     for element in train])
    valid_data, valid_labels = zip(*[[element[2], element[0]]
                                     for element in valid])
    test_data, test_labels = zip(*[[element[2], element[0]]
                                   for element in test])

    # create indexed classes
    class_mapping = mapping(train_labels)

    # replace all classes with indices
    LOGGER.info("Serializing output classes")
    train_labels = serialize(train_labels, class_mapping)
    valid_labels = serialize(valid_labels, class_mapping)
    test_labels = serialize(test_labels, class_mapping)

    # tokenize all datasets
    LOGGER.info("Tokenizing training data")
    train_data = tokenize(train_data)
    LOGGER.info("Tokenizing validation data")
    valid_data = tokenize(valid_data)
    LOGGER.info("Tokenizing test data")
    test_data = tokenize(test_data)

    # make everything unique
    LOGGER.info("Making training data unique")
    train = make_unique(list(zip(train_data, train_labels)))
    LOGGER.info("Making validation data unique")
    valid = make_unique(list(zip(valid_data, valid_labels)))
    LOGGER.info("Making test data unique")
    test = make_unique(list(zip(test_data, test_labels)))

    # write main files
    LOGGER.info("Sorting and writing data")
    write_file(train, class_mapping, "train", write_directory)
    write_file(valid, class_mapping, "valid", write_directory)
    write_file(test, class_mapping, "test", write_directory)

    # write class mapping
    with open(os.path.join(write_directory, "class_mapping.json"),
              'w') as output_file_stream:
        json.dump(class_mapping, output_file_stream, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse_formatter,
        parents=[preprocess_arg_parser(),
                 logging_arg_parser()])
    args = parser.parse_args()
    LOGGER = make_logger(args.logging_level)
    DISABLE_TQDM = args.disable_tqdm
    main(args)
