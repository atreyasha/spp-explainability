#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: preprocess_fmtod.sh [-h|--help]

Preprocess the FMTOD data set

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
preprocess_fmtod() {
  python3 -m src.preprocess_fmtod
}

# execute all functions
check_help "$@"
preprocess_fmtod
