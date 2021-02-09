#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict
from collections import OrderedDict
from torch.nn import Module
import torch
import re


class RegexSoftPatternClassifier(Module):
    def __init__(
            self,
            pattern_specs: 'OrderedDict[int, int]',
            activating_regex: Dict[int, List[str]],  # yapf: disable
            linear: Module) -> None:
        # initialize all class properties from torch.nn.Module
        super(RegexSoftPatternClassifier, self).__init__()
        self.pattern_specs = pattern_specs
        self.activating_regex = {
            key: [re.compile(regex) for regex in activating_regex[key]]
            for key in activating_regex.keys()
        }
        self.linear = linear

    def regex_forward(self, doc: str) -> List[int]:
        scores_doc = []
        for key in sorted(self.activating_regex.keys()):
            for regex in self.activating_regex[key]:
                if regex.search(doc):
                    scores_doc.append(1)
                    break
            else:
                scores_doc.append(0)
        return scores_doc

    def forward(self, batch: List[str]) -> torch.Tensor:
        # start loop over regular expressions
        scores = torch.FloatTensor([self.regex_forward(doc) for doc in batch
                                    ]).to(self.linear.weight.device)

        # convert scores to tensor
        return self.linear(scores)
