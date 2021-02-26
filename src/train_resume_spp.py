#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .utils.parser_utils import ArgparseFormatter
from .utils.logging_utils import stdout_root_logger
from .arg_parser import (logging_arg_parser, tqdm_arg_parser,
                         hardware_arg_parser, train_resume_arg_parser,
                         grid_train_arg_parser)
from .train_spp import (train_outer, get_grid_args_superset,
                        parse_configs_to_args)
import argparse
import json
import os


def main(args: argparse.Namespace) -> None:
    # Continue grid-training:
    if args.grid_training:
        # parse base configs into args and update models_directory
        args = parse_configs_to_args(args, "base_")

        # read grid_config into param_grid_mapping
        grid_config = os.path.join(args.model_log_directory,
                                   "grid_config.json")
        if os.path.exists(grid_config):
            with open(grid_config, "r") as input_file_stream:
                param_grid_mapping = json.load(input_file_stream)
        else:
            raise FileNotFoundError("File not found: %s" % grid_config)

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
        train_outer(args, resume_training=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=ArgparseFormatter,
        parents=[
            train_resume_arg_parser(),
            grid_train_arg_parser(start_training=False),
            hardware_arg_parser(),
            logging_arg_parser(),
            tqdm_arg_parser()
        ])
    LOGGER = stdout_root_logger(
        logging_arg_parser().parse_known_args()[0].logging_level)
    main(parser.parse_args())
