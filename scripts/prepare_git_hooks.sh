#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: prepare_git_hooks.sh [-h|--help]
Force copy git hooks to git repository config

Optional arguments:
  -h, --help    Show this help message and exit
EOF
}

# check for help
check_help() {
  for arg; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
      usage
      exit 1
    fi
  done
}

# define function
prepare_git_hooks() {
  local input
  for input in ./hooks/*; do
    cp "$input" "./.git/hooks/"
  done
}

# execute function
check_help "$@"
prepare_git_hooks
