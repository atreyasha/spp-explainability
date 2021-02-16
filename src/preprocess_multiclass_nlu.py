#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, cast
from .utils.data_utils import unique
from .utils.parser_utils import ArgparseFormatter
from .utils.logging_utils import stdout_root_logger
from .utils.preprocess_utils import (read_tsv, mapping, serialize, lowercase,
                                     tokenize, upsample,
                                     remove_commons_from_first)
from .arg_parser import preprocess_arg_parser, logging_arg_parser
import argparse
import json
import os


def write_file(full_data: List[Tuple[str, int]], mapping: Dict[str, int],
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

    # make everything unique internally
    LOGGER.info("Making training data unique")
    train = list(unique(zip(train_data, train_labels)))
    assert len(train) == len(set(train))
    LOGGER.info("Making validation data unique")
    valid = list(unique(zip(valid_data, valid_labels)))
    assert len(valid) == len(set(valid))
    LOGGER.info("Making test data unique")
    test = list(unique(zip(test_data, test_labels)))
    assert len(test) == len(set(test))

    # making cross-partitions unique
    LOGGER.info("Removing duplicates between partitions")
    train, test = remove_commons_from_first(train, test)
    assert len(set(train).intersection(set(test))) == 0
    train, valid = remove_commons_from_first(train, valid)
    assert len(set(train).intersection(set(valid))) == 0
    test, valid = remove_commons_from_first(test, valid)
    assert len(set(test).intersection(set(valid))) == 0
    train = cast(List[Tuple[str, int]], train)
    valid = cast(List[Tuple[str, int]], valid)
    test = cast(List[Tuple[str, int]], test)

    # write main files
    LOGGER.info("Sorting and writing data in directory: %s" % write_directory)
    write_file(train, class_mapping, "train", suffix, write_directory)
    write_file(valid, class_mapping, "valid", suffix, write_directory)
    write_file(test, class_mapping, "test", suffix, write_directory)

    # write class mapping
    with open(os.path.join(write_directory, "class_mapping.json"),
              'w') as output_file_stream:
        json.dump(class_mapping, output_file_stream, ensure_ascii=False)

    # upsample training and validation data if allowed
    if not args.disable_upsampling:
        LOGGER.info("Upsampling training and validation data")
        train = upsample(train)
        valid = upsample(valid)
        LOGGER.info("Sorting and writing upsampled data in directory: %s" %
                    write_directory)
        write_file(train, class_mapping, "train.upsampled", suffix,
                   write_directory)
        write_file(valid, class_mapping, "valid.upsampled", suffix,
                   write_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=ArgparseFormatter,
        parents=[preprocess_arg_parser(),
                 logging_arg_parser()])
    LOGGER = stdout_root_logger(
        logging_arg_parser().parse_known_args()[0].logging_level)
    main(parser.parse_args())
