#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict, Tuple, Union
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

    def regex_lookup(self, doc: str) -> List[int]:
        scores_doc: List[int] = []
        for key in sorted(self.activating_regex.keys()):
            for index, regex in enumerate(self.activating_regex[key]):
                if regex.search(doc):
                    scores_doc.append(1)
                    break
            else:
                scores_doc.append(0)
        return scores_doc

    def forward(self, batch: List[str]) -> torch.Tensor:
        # start loop over regular expressions
        scores = torch.FloatTensor(
            [self.regex_lookup(doc)
             for doc in batch]).to(self.linear.weight.device)  # type: ignore

        # convert scores to tensor
        return self.linear(scores)

    def regex_lookup_with_trace(
            self, doc: str) -> Tuple[List[int], List[Union[re.Match, None]]]:
        scores_doc: List[int] = []
        lookup_doc: List[Union[re.Match, None]] = []
        for key in sorted(self.activating_regex.keys()):
            for index, regex in enumerate(self.activating_regex[key]):
                regex_lookup = regex.search(doc)
                if regex_lookup:
                    scores_doc.append(1)
                    lookup_doc.append(regex_lookup)
                    break
            else:
                scores_doc.append(0)
                lookup_doc.append(None)
        return scores_doc, lookup_doc

    def forward_with_trace(
        self, batch: List[str]
    ) -> Tuple[torch.Tensor, List[List[Union[re.Match, None]]]]:
        # start loop over regular expressions
        all_data = [self.regex_lookup_with_trace(doc) for doc in batch]
        scores = torch.FloatTensor([data[0] for data in all_data]).to(
            self.linear.weight.device)  # type: ignore
        lookup = [data[1] for data in all_data]

        # convert scores to tensor
        return self.linear(scores), lookup
