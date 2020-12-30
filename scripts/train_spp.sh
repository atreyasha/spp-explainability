#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: train_spp.sh [-h|--help]
Execute single SoPa++ training run using repository defaults

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
train_spp() {
  python3 -m src.train_spp \
    --embeddings "./data/glove_6B_uncased/glove.6B.300d.txt" \
    --train-data "./data/facebook_multiclass_nlu/clean/train.uncased.data" \
    --train-labels "./data/facebook_multiclass_nlu/clean/train.labels" \
    --valid-data "./data/facebook_multiclass_nlu/clean/valid.uncased.data" \
    --valid-labels "./data/facebook_multiclass_nlu/clean/valid.labels"
}

# execute function
check_help "$@"
train_spp
