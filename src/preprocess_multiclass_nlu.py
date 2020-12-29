#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict, Any, Generator, Iterable, Tuple
from nltk import word_tokenize
from .utils.parser_utils import ArgparseFormatter
from .utils.logging_utils import make_logger
from .utils.data_utils import unique
from .arg_parser import preprocess_arg_parser, logging_arg_parser
import argparse
import logging
import json
import csv
import os

# get root LOGGER in case script is called by another
LOGGER = logging.getLogger(__name__)


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


def write_file(full_data: Generator[Tuple[str, int], None,
                                    None], mapping: Dict[str, int],
               prefix: str, suffix: str, write_directory: str) -> None:
    # make write directory if it does not exist
    os.makedirs(write_directory, exist_ok=True)
    # split compund data into two
    data, labels = zip(*sorted(full_data))
    # write data
    with open(
            os.path.join(write_directory, ".".join([prefix, suffix, "data"])),
            'w') as output_file_stream:
        for item in data:
            output_file_stream.write("%s\n" % item)
    # write labels
    with open(os.path.join(write_directory, ".".join([prefix, "labels"])),
              'w') as output_file_stream:
        for item in labels:
            output_file_stream.write("%s\n" % item)


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

    # process casing of data
    if not args.truecase:
        LOGGER.info("Lower-casing training data")
        train_data = lowercase(train_data)
        LOGGER.info("Lower-casing validation data")
        valid_data = lowercase(valid_data)
        LOGGER.info("Lower-casing test data")
        test_data = lowercase(test_data)
        suffix = "uncased"
    else:
        suffix = "truecased"

    # make everything unique
    LOGGER.info("Making training data unique")
    train = unique(zip(train_data, train_labels))
    LOGGER.info("Making validation data unique")
    valid = unique(zip(valid_data, valid_labels))
    LOGGER.info("Making test data unique")
    test = unique(zip(test_data, test_labels))

    # write main files
    LOGGER.info("Sorting and writing data")
    write_file(train, class_mapping, "train", suffix, write_directory)
    write_file(valid, class_mapping, "valid", suffix, write_directory)
    write_file(test, class_mapping, "test", suffix, write_directory)

    # write class mapping
    with open(os.path.join(write_directory, "class_mapping.json"),
              'w') as output_file_stream:
        json.dump(class_mapping, output_file_stream, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=ArgparseFormatter,
        parents=[preprocess_arg_parser(),
                 logging_arg_parser()])
    args = parser.parse_args()
    LOGGER = make_logger(args.logging_level)
    main(args)
