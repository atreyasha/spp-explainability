#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: visualize_fmtod.sh [-h|--help]

Visualize FMTOD data set summary statistics

Optional arguments:
  -h, --help  Show this help message and exit
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
visualize_fmtod() {
  Rscript src/visualize_fmtod.R \
    -t "./data/fmtod/clean/train.labels" \
    -v "./data/fmtod/clean/valid.labels" \
    -e "./data/fmtod/clean/test.labels"
}

# execute function
check_help "$@"
visualize_fmtod
