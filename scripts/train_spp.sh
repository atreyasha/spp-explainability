#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: train_spp.sh [-h|--help]

Execute single SoPa++ model training run using repository defaults

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
train_spp() {
  python3 -m src.train_spp \
    --embeddings "./data/glove_6B_uncased/glove.6B.300d.txt" \
    --train-data "./data/fmtod/clean/train.upsampled.uncased.data" \
    --train-labels "./data/fmtod/clean/train.upsampled.labels" \
    --valid-data "./data/fmtod/clean/valid.upsampled.uncased.data" \
    --valid-labels "./data/fmtod/clean/valid.upsampled.labels"
}

# execute function
check_help "$@"
train_spp
