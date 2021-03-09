#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: visualize_evaluate_spp_grid.sh [-h|--help] model_log_directory

Visualize grid evaluations with SoPa++ neural and regex
models, given that grid allows for the following varying arguments:
patterns, tau_threshold, seed

Optional arguments:
  -h, --help                       Show this help message and exit

Required arguments:
  model_log_directory <glob_path>  Model log directory/directories
                                   containing SoPa++ neural and regex
                                   models, as well as evaluation json's
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
visualize_evaluate_spp_grid() {
  local model_log_directory
  model_log_directory="$1"

  Rscript src/visualize_spp.R -e -g "$model_log_directory"
}

# execute function
check_help "$@"
visualize_evaluate_spp_grid "$@"
