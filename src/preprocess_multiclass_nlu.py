#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Iterable
from nltk import word_tokenize
from .utils.parser_utils import argparse_formatter
from .utils.logging_utils import provide_logger
from .arg_parser import preprocess_arg_parser, logging_arg_parser
import argparse
import json
import csv
import os


def read_tsv(input_file: str) -> List:
    with open(input_file, 'r') as input_file_stream:
        raw_list = list(csv.reader(input_file_stream, delimiter='\t'))
    return raw_list


def mapping(classes: List) -> dict:
    return {element: i for i, element in enumerate(sorted(set(classes)))}


def serialize(data: List, mapping: dict) -> List:
    return [mapping[element] for element in data]


def tokenize(data: List) -> List:
    return [" ".join(word_tokenize(element)) for element in data]


def make_unique(full_data: Iterable) -> List:
    unique_list = []
    for element in full_data:
        if element not in unique_list:
            unique_list.append(element)
    return unique_list


def write_file(full_data: List, mapping: dict, prefix: str,
               write_directory: str) -> None:
    # make write directoy if it does not exist
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


def main(data_directory: str) -> None:
    # define key directory and files
    write_directory = os.path.join(data_directory, "clean")
    train = os.path.join(data_directory, "raw", "en", "train-en.tsv")
    dev = os.path.join(data_directory, "raw", "en", "eval-en.tsv")
    test = os.path.join(data_directory, "raw", "en", "test-en.tsv")
    # process files
    logger.info("Reading input data")
    train = read_tsv(train)
    dev = read_tsv(dev)
    test = read_tsv(test)
    # extract data and labels
    train_data, train_labels = zip(*[[element[2], element[0]]
                                     for element in train])
    dev_data, dev_labels = zip(*[[element[2], element[0]] for element in dev])
    test_data, test_labels = zip(*[[element[2], element[0]]
                                   for element in test])
    # create indexed classes
    class_mapping = mapping(train_labels)
    # replace all classes with indices
    logger.info("Serializing output classes")
    train_labels = serialize(train_labels, class_mapping)
    dev_labels = serialize(dev_labels, class_mapping)
    test_labels = serialize(test_labels, class_mapping)
    # tokenize all datasets
    logger.info("Tokenizing with NLTK word_tokenize")
    train_data = tokenize(train_data)
    dev_data = tokenize(dev_data)
    test_data = tokenize(test_data)
    # make everything unique
    logger.info("Making data unique")
    train = make_unique(zip(train_data, train_labels))
    dev = make_unique(zip(dev_data, dev_labels))
    test = make_unique(zip(test_data, test_labels))
    # write main files
    logger.info("Sorting and writing data")
    write_file(train, class_mapping, "train", write_directory)
    write_file(dev, class_mapping, "dev", write_directory)
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
    logger = provide_logger(args.logging_level)
    main(args.data_directory)
