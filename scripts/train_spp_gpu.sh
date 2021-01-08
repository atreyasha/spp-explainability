#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: train_spp_gpu.sh [-h|--help]

Execute single SoPa++ model training run using repository defaults
on a GPU

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
train_spp_gpu() {
  python3 -m src.train_spp \
    --embeddings "./data/glove_6B_uncased/glove.6B.300d.txt" \
    --train-data "./data/facebook_multiclass_nlu/clean/train.upsampled.uncased.data" \
    --train-labels "./data/facebook_multiclass_nlu/clean/train.upsampled.labels" \
    --valid-data "./data/facebook_multiclass_nlu/clean/valid.upsampled.uncased.data" \
    --valid-labels "./data/facebook_multiclass_nlu/clean/valid.upsampled.labels"
  --gpu
}

# execute function
check_help "$@"
train_spp_gpu
