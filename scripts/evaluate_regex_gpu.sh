#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: evaluate_regex_gpu.sh [-h|--help] regex_model_checkpoint

Evaluate regex model(s) on an evaluation data set on a GPU

Optional arguments:
  -h, --help                          Show this help message and exit

Required arguments:
  regex_model_checkpoint <glob_path>  Glob path to regex model checkpoint(s)
                                      with '.pt' extension
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
evaluate_regex_gpu() {
  local regex_model_checkpoint
  regex_model_checkpoint="$1"

  python3 -m src.evaluate_regex \
    --eval-data "./data/fmtod/clean/test.uncased.data" \
    --eval-labels "./data/fmtod/clean/test.labels" \
    --model-checkpoint "$regex_model_checkpoint" --gpu
}

# execute function
check_help "$@"
evaluate_regex_gpu "$@"
