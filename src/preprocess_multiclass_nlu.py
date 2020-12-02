#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
from nltk import word_tokenize
from .utils.parser_utils import argparse_formatter
from .utils.logging_utils import provide_logger
from .arg_parser import preprocess_arg_parser, logging_arg_parser
import argparse
import json
import csv
import os


def read_file(input_file: str) -> List:
    with open(input_file, 'r') as input_file_stream:
        raw_list = list(csv.reader(input_file_stream, delimiter='\t'))
    return [[element[2], element[0]] for element in raw_list]


def mapping(classes: List) -> dict:
    return {element: i for i, element in enumerate(sorted(set(classes)))}


def serialize(full_data: List, mapping: dict) -> List:
    return [[element[0], mapping[element[1]]] for element in full_data]


def tokenize(full_data: List) -> List:
    return [[" ".join(word_tokenize(element[0])), element[1]]
            for element in full_data]


def make_unique(full_data: List) -> List:
    unique_list = []
    for element in full_data:
        if element not in unique_list:
            unique_list.append(element)
    return unique_list


def write_file(full_data: List,
               mapping: dict,
               prefix: str,
               write_directory: str,
               make_unique: bool = True) -> None:
    # make write directoy if it does not exist
    os.makedirs(write_directory, exist_ok=True)
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
    train = read_file(train)
    dev = read_file(dev)
    test = read_file(test)
    # create indexed classes
    class_mapping = mapping(list(zip(*train))[1])
    # replace all classes with indices
    logger.info("Serializing output classes")
    train = serialize(train, class_mapping)
    dev = serialize(dev, class_mapping)
    test = serialize(test, class_mapping)
    # tokenize all datasets
    logger.info("Tokenizing with NLTK word_tokenize")
    train = tokenize(train)
    dev = tokenize(dev)
    test = tokenize(test)
    # make everything unique
    logger.info("Making data unique")
    train = make_unique(train)
    dev = make_unique(dev)
    test = make_unique(test)
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
    parser = argparse.ArgumentParser(formatter_class=argparse_formatter,
                                     parents=[preprocess_arg_parser(),
                                              logging_arg_parser()])
    args = parser.parse_args()
    logger = provide_logger(args.logging_level)
    main(args.data_directory)
