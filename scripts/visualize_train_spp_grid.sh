#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: visualize_train_spp_grid.sh [-h|--help] tb_event_directory

Visualize grid training performance for SoPa++ neural models,
given that grid allows for the following varying arguments:
patterns, tau_threshold, seed

Optional arguments:
  -h, --help                      Show this help message and exit

Required arguments:
  tb_event_directory <glob_path>  Tensorboard event log directory/
                                  directories
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
visualize_train_spp_grid() {
  local tb_event_directory
  tb_event_directory="$1"

  python3 -m src.tensorboard_event2csv --tb-event-directory "$tb_event_directory"
  Rscript src/visualize_spp.R -t -g "$tb_event_directory"
}

# execute function
check_help "$@"
visualize_train_spp_grid "$@"
