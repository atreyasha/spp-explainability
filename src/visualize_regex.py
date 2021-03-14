#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from glob import glob
from math import ceil
from tqdm import tqdm
from typing import Tuple, Dict
from graphviz import Digraph
from matplotlib.colors import to_rgba
from .utils.model_utils import timestamp
from .utils.parser_utils import ArgparseFormatter
from .utils.logging_utils import stdout_root_logger
from .arg_parser import (visualize_regex_arg_parser, logging_arg_parser,
                         tqdm_arg_parser)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import json
import re
import os

# specify matplotlib high-level engine parameters
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Palatino"]
})


def get_neuron_colors() -> np.ndarray:
    alarm = to_rgba("red")
    reminder = to_rgba("black")
    weather = to_rgba("dodgerblue")
    alarms = np.tile(alarm, [6, 1])
    reminders = np.tile(reminder, [3, 1])
    weathers = np.tile(weather, [3, 1])
    alarms[:, -1] = np.linspace(0.4, 0.9, 6)
    reminders[:, -1] = np.linspace(0.6, 0.9, 3)
    weathers[:, -1] = np.linspace(0.6, 0.9, 3)
    return np.concatenate((alarms, reminders, weathers))


def get_rev_class_mapping(class_mapping_config: str) -> Dict[int, str]:
    with open(class_mapping_config, "r") as input_file_stream:
        class_mapping = json.load(input_file_stream)
    return {int(value): key for key, value in class_mapping.items()}


def get_model_linear_weights(regex_model_checkpoint: str,
                             softmax: bool = True) -> Tuple[Dict, np.ndarray]:
    # load model here
    model_dict = torch.load(regex_model_checkpoint)

    # get weights and process them
    weights = model_dict["linear_state_dict"]["weight"].t()

    # perform softmax over weights if necessary
    if softmax:
        weights = torch.softmax(weights, 1).numpy()
    else:
        weights = weights.numpy()

    # return the processed weights
    return model_dict, weights


def visualize_only_neurons(args: argparse.Namespace) -> None:
    # load pre-requisite data
    _, weights = get_model_linear_weights(args.regex_model_checkpoint)
    rev_class_mapping = get_rev_class_mapping(args.class_mapping_config)

    # get pre-defined neuron colors
    colors = get_neuron_colors()

    # create legend patches
    patches = [
        mpatches.Patch(color=colors[key],
                       label=re.sub("\\_", "\\_", rev_class_mapping[key]))
        for key in rev_class_mapping
    ]

    # create figure outline
    ncol = 10
    nrow = ceil(weights.shape[0] / ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(14, 7))

    # start adding pie charts to figure
    for i, ax in enumerate(axs.flatten()):
        ax.pie(weights[i],
               colors=colors,
               wedgeprops=dict(width=0.5, edgecolor='w'),
               normalize=True)
        ax.set_title("N$_{%s}$" % i, size=15)

    # add legend and title to figure
    fig.legend(handles=patches,
               loc="lower center",
               ncol=len(patches) // 2,
               frameon=False,
               prop={"size": 11})
    fig.suptitle("\\textbf{STE neuron relative-weights by output class}",
                 fontsize=16)

    # adjust formatting and save plot to pdf
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout(rect=[0, 0.075, 1, 0.98])
    fig.savefig(
        os.path.join("./docs/visuals/pdfs/generated/",
                     "neurons_" + timestamp() + ".pdf"))
    plt.close("all")


def visualize_regex_neurons(args: argparse.Namespace) -> None:
    # load pre-requisite data
    model_dict, weights = get_model_linear_weights(args.regex_model_checkpoint)
    rev_class_mapping = get_rev_class_mapping(args.class_mapping_config)

    # get pre-defined neuron colors
    colors = get_neuron_colors()

    # create legend patches
    patches = [
        mpatches.Patch(color=colors[key],
                       label=re.sub("\\_", "\\_", rev_class_mapping[key]))
        for key in rev_class_mapping
    ]

    # compute pattern lengths for each pattern
    pattern_lengths = [
        end
        for pattern_len, num_patterns in model_dict["pattern_specs"].items()
        for end in num_patterns * [pattern_len]
    ]

    # declare save directory
    save_directory = os.path.join("./docs/visuals/pdfs/generated/",
                                  "neurons_regex_" + timestamp())

    # start for-loop over all WFSAs
    for i in tqdm(range(len(pattern_lengths)), disable=args.disable_tqdm):
        # get and subset regex's
        regexes = model_dict["activating_regex"][i]
        try:
            indices = np.random.choice(len(regexes),
                                       args.num_regex,
                                       replace=False)
            regexes = [regexes[index] for index in indices]
        except ValueError:
            pass

        # clean up regular expressions
        regexes = [
            regex.replace("(\\s|^)(", "").replace(")(\\s|$)", "").split()
            for regex in regexes
        ]

        # start plotting graph with nodes first
        d = Digraph()
        d.attr(rankdir="LR")

        # lay out all nodes
        for pattern_index in range(pattern_lengths[i]):
            if pattern_index == 0:
                with d.subgraph() as s:
                    s.attr("node",
                           rank="same",
                           shape="doublecircle",
                           width="1")
                    for regex_index in range(len(regexes)):
                        s.node('S_%s_%s' % (regex_index, pattern_index),
                               label="START / %s" % pattern_index)
            elif pattern_index == max(range(pattern_lengths[i])):
                with d.subgraph() as s:
                    s.attr("node",
                           rank="same",
                           shape="doublecircle",
                           width="1")
                    for regex_index in range(len(regexes)):
                        s.node('S_%s_%s' % (regex_index, pattern_index),
                               label="END / %s" % pattern_index)
            else:
                with d.subgraph() as s:
                    s.attr("node", rank='same', shape="circle")
                    for regex_index in range(len(regexes)):
                        s.node('S_%s_%s' % (regex_index, pattern_index),
                               label=str(pattern_index))

        # lay out all edges
        for regex_index in range(len(regexes)):
            for pattern_index in range(pattern_lengths[i] - 1):
                transitions = regexes[regex_index][pattern_index].replace(
                    "(", "").replace(")", "").split("|")
                if len(transitions) == 1 and transitions[0] == "[^\\s]+":
                    transitions[0] = "*"
                elif len(transitions) > args.max_transition_tokens:
                    transitions = transitions[:(args.max_transition_tokens +
                                                1)]
                    transitions[-1] = "..."
                for transition in transitions:
                    d.edge("S_%s_%s" % (regex_index, pattern_index),
                           "S_%s_%s" % (regex_index, pattern_index + 1),
                           label=transition)

        # render dotfile as pdf and save it
        d.render(os.path.join(save_directory,
                              "activating_regex_sample_%s" % i),
                 cleanup=True)

        # create figure outline
        fig, ax = plt.subplots(figsize=(8, 5))

        # embed pie chart
        ax.pie(weights[i],
               colors=colors,
               wedgeprops=dict(width=0.5, edgecolor='w'),
               normalize=True)
        ax.set_title("N$_{%s}$" % i, size=20)

        # add legend and title to figure
        fig.legend(handles=patches,
                   loc="right",
                   ncol=1,
                   frameon=False,
                   prop={"size": 11})

        # adjust formatting and save plot to pdf
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout(rect=[0, 0.075, 0.9, 0.98])
        fig.savefig(os.path.join(save_directory, "neuron_%s.pdf" % i),
                    bbox_inches='tight')
        plt.close("all")


def main(args: argparse.Namespace) -> None:
    # collect all checkpoints
    LOGGER.info("Collecting checkpoints via glob: %s" %
                args.regex_model_checkpoint)
    regex_model_checkpoint_collection = glob(args.regex_model_checkpoint)

    # set random seed for visualization determinism
    np.random.seed(args.seed)

    # loop over all provided models
    for regex_model_checkpoint in regex_model_checkpoint_collection:
        # load back specific file path
        args.regex_model_checkpoint = regex_model_checkpoint
        LOGGER.info("Visualizing checkpoint: %s" % args.regex_model_checkpoint)

        # apply conditional for next function
        if args.only_neurons:
            visualize_only_neurons(args)
        else:
            visualize_regex_neurons(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=ArgparseFormatter,
                                     parents=[
                                         visualize_regex_arg_parser(),
                                         logging_arg_parser(),
                                         tqdm_arg_parser()
                                     ])
    LOGGER = stdout_root_logger(
        logging_arg_parser().parse_known_args()[0].logging_level)
    main(parser.parse_args())
