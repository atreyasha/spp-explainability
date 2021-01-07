#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: train_spp_grid_cpu.sh [-h|--help] [grid_config]

Execute SoPa++ model grid training run using repository defaults
on the CPU

Optional arguments:
  -h, --help               Show this help message and exit
  grid_config <file_path>  Path to grid configuration file
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
train_spp_grid_cpu() {
  local grid_config
  grid_config="$1"

  if [ -z "$grid_config" ]; then
    python3 -m src.train_spp \
      --embeddings "./data/glove_6B_uncased/glove.6B.300d.txt" \
      --train-data "./data/facebook_multiclass_nlu/clean/train.uncased.data" \
      --train-labels "./data/facebook_multiclass_nlu/clean/train.labels" \
      --valid-data "./data/facebook_multiclass_nlu/clean/valid.uncased.data" \
      --valid-labels "./data/facebook_multiclass_nlu/clean/valid.labels" \
      --grid-training
  else
    python3 -m src.train_spp \
      --embeddings "./data/glove_6B_uncased/glove.6B.300d.txt" \
      --train-data "./data/facebook_multiclass_nlu/clean/train.uncased.data" \
      --train-labels "./data/facebook_multiclass_nlu/clean/train.labels" \
      --valid-data "./data/facebook_multiclass_nlu/clean/valid.uncased.data" \
      --valid-labels "./data/facebook_multiclass_nlu/clean/valid.labels" \
      --grid-training --grid-config "$grid_config"
  fi
}

# execute function
check_help "$@"
train_spp_grid_cpu "$@"
