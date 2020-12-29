#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import logging


def make_logger(level: str) -> logging.Logger:
    # create logger
    logger = logging.getLogger(level)

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
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s'
    )

    # set output stread to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # return final logger
    return logger
