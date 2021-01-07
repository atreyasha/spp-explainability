#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: preprocess_multiclass_nlu.sh [-h|--help]

Preprocess Facebook multiclass NLU data using repository defaults

Optional arguments:
  -h, --help    Show this help message and exit
EOF
}

# check for help
check_help() {
  for arg; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
      usage
      exit 0
    fi
  done
}

# define function
preprocess_multiclass_nlu() {
  python3 -m src.preprocess_multiclass_nlu
}

# execute all functions
check_help "$@"
preprocess_multiclass_nlu
