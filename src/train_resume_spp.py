#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .utils.parser_utils import ArgparseFormatter
from .utils.logging_utils import stdout_root_logger
from .arg_parser import (logging_arg_parser, tqdm_arg_parser,
                         hardware_arg_parser, resume_training_arg_parser,
                         grid_training_arg_parser)
from .train_spp import (train_outer, get_grid_args_superset,
                        parse_configs_to_args)
import argparse
import json
import os


def main(args: argparse.Namespace) -> None:
    # Continue grid-training:
    if args.grid_training:
        # parse base configs into args and update models_directory
        args = parse_configs_to_args(args, args.model_log_directory, "base_")

        # read grid_config into param_grid_mapping
        with open(os.path.join(args.model_log_directory, "grid_config.json"),
                  "r") as input_file_stream:
            param_grid_mapping = json.load(input_file_stream)

        # convert args and param_grid_mapping into list of all args
        args_superset = get_grid_args_superset(args, param_grid_mapping)

        # update all model_log_directory variables with indices
        for i, args in enumerate(args_superset):
            args.model_log_directory = os.path.join(
                args.model_log_directory, "spp_single_train_" + str(i))
    else:
        args_superset = [args]

    # loop and resume training
    for args in args_superset:
        try:
            train_outer(args, resume_training=True)
        except FileNotFoundError:
            if args.grid_training:
                train_outer(args, resume_training=False)
            else:
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=ArgparseFormatter,
        parents=[
            resume_training_arg_parser(),
            grid_training_arg_parser(resume_training=True),
            hardware_arg_parser(),
            logging_arg_parser(),
            tqdm_arg_parser()
        ])
    args = parser.parse_args()
    LOGGER = stdout_root_logger(args.logging_level)
    main(args)
