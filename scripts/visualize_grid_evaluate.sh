#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: visualize_grid_evaluate.sh [-h|--help] model_log_directory

Visualize grid evaluations for neural SoPa++ and regex
model pairs, given that grid allows for the following varying arguments:
patterns, tau_threshold, seed

Optional arguments:
  -h, --help                       Show this help message and exit

Required arguments:
  model_log_directory <glob_path>  Model log directory/directories
                                   containing neural SoPa++ and regex
                                   models, as well as all evaluation json's
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
visualize_grid_evaluate() {
  local model_log_directory
  model_log_directory="$1"

  Rscript src/visualize_grid.R -e -g "$model_log_directory"
}

# execute function
check_help "$@"
visualize_grid_evaluate "$@"
