#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging

FORMAT = (
    '%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s')


def stdout_root_logger(level: str) -> logging.Logger:
    # get root logger
    logger = logging.getLogger()

    # define logger levels
    levels = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }

    # set logger level
    logger.setLevel(levels[level.lower()])

    # create formatter
    formatter = logging.Formatter(FORMAT)

    # set output stream to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    # add stream to logger
    logger.addHandler(stdout_handler)

    # return final logger
    return logger


def remove_all_file_handlers(logger: logging.Logger) -> logging.Logger:
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)

    # return adjusted logger
    return logger


def add_unique_file_handler(logger: logging.Logger,
                            filename: str) -> logging.Logger:
    # create formatter
    formatter = logging.Formatter(FORMAT)

    # create file handler
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # add file handler to logger
    logger.addHandler(file_handler)

    # return adjusted logger
    return logger
