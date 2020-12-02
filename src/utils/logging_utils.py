#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    # create console handler and set level to debug
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    # create formattr
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s'
    )
    # add formatter to ch
    stream_handler.setFormatter(formatter)
    # add stream_handler to logger
    logger.addHandler(stream_handler)
    return logger
