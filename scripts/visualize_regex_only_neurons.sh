#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: visualize_regex_only_neurons.sh [-h|--help] regex_model_checkpoint

Visualize only STE neurons in regex model(s)

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
visualize_regex_only_neurons() {
  local regex_model_checkpoint
  regex_model_checkpoint="$1"

  python3 -m src.visualize_regex --only-neurons \
    --class-mapping-config "./data/fmtod/clean/class_mapping.json" \
    --regex-model-checkpoint "$regex_model_checkpoint"
}

# execute function
check_help "$@"
visualize_regex_only_neurons "$@"
