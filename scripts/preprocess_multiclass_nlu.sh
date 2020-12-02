#!/usr/bin/env bash
# Preprocess Facebook multiclass NLU data
set -e

# usage function
usage() {
  cat <<EOF
Usage: preprocess_multiclass_nlu.sh [-h|--help]
Preprocess Facebook multiclass NLU data

Optional arguments:
  -h, --help         Show this help message and exit
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

preprocess_multiclass_nlu() {
  python3 -m src.preprocess_multiclass_nlu
}

# execute all functions
check_help "$@"
preprocess_multiclass_nlu