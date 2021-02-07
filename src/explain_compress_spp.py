#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm
from typing import List
from .utils.parser_utils import ArgparseFormatter
from .utils.logging_utils import stdout_root_logger
from .arg_parser import (explain_compress_arg_parser, logging_arg_parser,
                         tqdm_arg_parser)
from .explain_simplify_spp import save_regex_model
import argparse
import torch
import os


def rational_clustering(pattern_regex: List[str]) -> List[str]:
    # intitialize storage list
    clustered_pattern_regex = []
    pattern_regex = list(
        map(lambda x: x.replace("\\b", "").split(), pattern_regex))

    # loop over pattern regular expressions until all processed
    while len(pattern_regex) != 0:
        start = pattern_regex.pop(0)
        index_segmenter = {i: [] for i in range(len(start))}
        all_to_compare = pattern_regex.copy()

        # collect patterns that match and segment them
        for to_compare in all_to_compare:
            count = 0
            for i, (a, b) in enumerate(zip(start, to_compare)):
                if a != b:
                    count += 1
                    index = i
                if count == 2:
                    break
            else:
                index_segmenter[index].append(to_compare)
                pattern_regex.remove(to_compare)

        # conditionally append to tracking list
        if all(
                len(index_segmenter[key]) == 0
                for key in index_segmenter.keys()):
            clustered_pattern_regex.append("\\b" + " ".join(start) + "\\b")
        else:
            dense_keys = [
                key for key in index_segmenter.keys()
                if len(index_segmenter[key]) > 0
            ]
            for i, key in enumerate(dense_keys):
                # create new combined regex
                joint = start.copy()

                # gather all strings to concatenate
                join_list = [joint[key]] + [
                    regex[key] for regex in index_segmenter[key]
                ] if i == 0 else [
                    regex[key] for regex in index_segmenter[key]
                ]

                # make join_list unique
                join_list = list(set(join_list))

                # combine list into string
                if len(join_list) == 1:
                    joint[key] = join_list[0]
                else:
                    # filter wildcards and replace string with combined regex
                    if "[^\\s]+" in join_list:
                        joint[key] = "[^\\s]+"
                    else:
                        # combine remaining words and add them in
                        joint[key] = "(" + "|".join(join_list) + ")"

                # finally append to tracking list
                clustered_pattern_regex.append("\\b" + " ".join(joint) + "\\b")

    return clustered_pattern_regex


def main(args: argparse.Namespace) -> None:
    # load regex model
    LOGGER.info("Loading regex model: %s" % args.regex_model)
    regex_model = torch.load(args.regex_model,
                             map_location=torch.device("cpu"))

    LOGGER.info("Compressing with method: %s" % args.compression_method)
    # conduct compression as required
    if args.compression_method == "rational":
        regex_model["activating_regex"] = {
            key: rational_clustering(regex_model["activating_regex"][key])
            for key in tqdm(regex_model["activating_regex"],
                            disable=args.disable_tqdm)
        }

    # save model as required
    save_regex_model(
        regex_model["pattern_specs"], regex_model["activating_regex"],
        regex_model["linear_state_dict"],
        os.path.join(
            args.regex_model.replace(".pt",
                                     "_" + args.compression_method + ".pt")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter,
                                     parents=[
                                         explain_compress_arg_parser(),
                                         logging_arg_parser(),
                                         tqdm_arg_parser()
                                     ])
    LOGGER = stdout_root_logger(
        logging_arg_parser().parse_known_args()[0].logging_level)
    main(parser.parse_args())
