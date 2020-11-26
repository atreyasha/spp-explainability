#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse


class Metavar_Circum_Symbols(argparse.HelpFormatter):
    """
    Help message formatter which uses the argument 'type' as the default
    metavar value (instead of the argument 'dest')

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """
    def _get_default_metavar_for_optional(self, action):
        """
        Function to return option metavariable type with circum-symbols
        """
        return "<" + action.type.__name__ + ">"

    def _get_default_metavar_for_positional(self, action):
        """
        Function to return positional metavariable type with circum-symbols
        """
        return "<" + action.type.__name__ + ">"


class Metavar_Indenter(argparse.HelpFormatter):
    """
    Formatter for generating usage messages and argument help strings.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """
    def _format_action(self, action):
        """
        Function to define how actions are printed in help message
        """
        # determine the required width and the entry label
        help_position = min(self._action_max_length + 2,
                            self._max_help_position)
        help_width = max(self._width - help_position, 11)
        action_width = help_position - self._current_indent - 2
        action_header = self._format_action_invocation(action)

        # no help; start on same line and add a final newline
        if not action.help:
            tup = self._current_indent, '', action_header
            action_header = '%*s%s\n' % tup

        # short action name; start on the same line and pad two spaces
        elif len(action_header) <= action_width:
            tup = self._current_indent, '', action_width, action_header
            action_header = '%*s%-*s  ' % tup
            indent_first = 0

        # long action name; start on the next line
        else:
            tup = self._current_indent, '', action_header
            action_header = '%*s%s\n' % tup
            indent_first = help_position

        # collect the pieces of the action help
        parts = [action_header]

        # if there was help for the action, add lines of help text
        if action.help:
            help_text = self._expand_help(action)
            help_lines = self._split_lines(help_text, help_width)
            if action.nargs != 0:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                parts.append('%*s%s\n' % (indent_first, '', args_string))
            else:
                parts.append('%*s%s\n' % (indent_first, '', help_lines[0]))
                help_lines.pop(0)
            for line in help_lines:
                parts.append('%*s%s\n' % (help_position, '', line))

        # or add a newline if the description doesn't end with one
        elif not action_header.endswith('\n'):
            parts.append('\n')

        # if there are any sub-actions, add their help as well
        for subaction in self._iter_indented_subactions(action):
            parts.append(self._format_action(subaction))

        # return a single string
        return self._join_parts(parts)

    def _format_action_invocation(self, action):
        """
        Lower function to define how actions are printed in help message
        """
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar, = self._metavar_formatter(action, default)(1)
            return metavar
        else:
            parts = []
            parts.extend(action.option_strings)
            return ', '.join(parts)


class arg_metav_formatter(argparse.ArgumentDefaultsHelpFormatter,
                          Metavar_Circum_Symbols, Metavar_Indenter):
    """
    Class to combine argument parsers in order to display meta-variables
    and defaults for arguments
    """
    pass
