#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from glob import glob
from typing import Dict, List, Any
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator)
from .utils.parser_utils import ArgparseFormatter
from .utils.logging_utils import stdout_root_logger
from .arg_parser import logging_arg_parser, tensorboard_arg_parser
import numpy as np
import argparse
import os
import csv


def dict2csv(out: Dict[str, List[Any]], dpath: str) -> None:
    """
    Function to write dictionary object as csv file

    Args:
        out: Dictionary containing values/steps to write
        dpath: Path of the directory containing tensorboard logs
    """
    with open(os.path.join(dpath, "%s.csv" % os.path.basename(dpath)),
              "w") as f:
        writer = csv.DictWriter(f, out.keys())
        writer.writeheader()
        for i in range(len(out["steps"])):
            writer.writerow({key: out[key][i] for key in out.keys()})


def tabulate_events(dpath: str) -> Dict[str, List[Any]]:
    """
    Function to tabulate and aggregate event logs into single dictionary
    with post-processing to ensure data sanity

    Args:
        dpath: Path of the directory containing tensorboard logs

    Returns:
        out: Dictionary containing relevant tensorboard data
    """
    summary_iterators = [
        EventAccumulator(os.path.join(dpath, dname)).Reload()
        for dname in os.listdir(dpath) if ".csv" not in dname
    ]
    # find lowest common denominator in tags
    tags = set(summary_iterators[0].Tags()["scalars"])
    for summary_iterator in summary_iterators[1:]:
        tags.intersection_update(summary_iterator.Tags()["scalars"])
    tags = list(tags)
    # create variable dictionary
    out = defaultdict(list)
    # loop over summary iterators
    for summary_iterator in summary_iterators:
        # find lowest common denominator of step elements
        steps_list = [[event.step for event in summary_iterator.Scalars(tag)]
                      for tag in tags]
        steps = set(steps_list[0])
        for inner_list in steps_list[1:]:
            steps.intersection_update(inner_list)
        # sort these in ascending order
        steps = sorted(list(steps))
        out["steps"].extend(steps)
        for tag in tags:
            hold: List[Any] = []
            for event in summary_iterator.Scalars(tag):
                current_steps = [] if hold == [] else list(zip(*hold))[0]
                if event.step in steps and event.step not in current_steps:
                    hold.append([event.step, event.value])
            # sort hold again in case this was disrupted during event access
            hold = [element[1] for element in sorted(hold, key=lambda x: x[0])]
            # ensure all lengths are the same
            assert len(hold) == len(steps)
            # append to out after sorting/cleaning
            out[tag].extend(hold)
    # sort everything based on steps
    sorting_indices = np.argsort(out["steps"])
    for tag in tags + ["steps"]:
        out[tag] = [out[tag][i] for i in sorting_indices]
    return out


def main(args: argparse.Namespace) -> None:
    """ Main function to tabulate tensorboard data and write to disk as csv """
    # parse for tensorboard logs
    tb_event_directories = glob(args.tb_event_directory)
    # loop over log directories
    for tb_event_directory in tb_event_directories:
        if os.path.exists(
                os.path.join(
                    tb_event_directory, "%s.csv" %
                    os.path.basename(tb_event_directory))) and not args.force:
            continue
        else:
            LOGGER.info("Processing: %s", tb_event_directory)
            out = tabulate_events(tb_event_directory)
            LOGGER.info("Writing results to directory: %s", tb_event_directory)
            dict2csv(out, tb_event_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter,
                                     parents=[
                                         tensorboard_arg_parser(),
                                         logging_arg_parser(),
                                     ])
    LOGGER = stdout_root_logger(
        logging_arg_parser().parse_known_args()[0].logging_level)
    main(parser.parse_args())
