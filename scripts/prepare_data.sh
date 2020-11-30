#!/usr/bin/env bash
# This function downloads key data sets for extending SoPa
set -e

# usage function
usage() {
  cat <<EOF
Usage: prepare_data.sh [-h|--help]
Prepare data sets for extending SoPa

Optional arguments:
  -h, --help         Show this help message and exit
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

# download and prepare small NLU intent data sets
facebook_multi_task_nlu() {
  local directory="./data/facebook_multi_task_nlu/multilingual_task_oriented_dialog_slotfilling"
  mkdir -p "$directory"
  wget -N -P "$directory" "https://download.pytorch.org/data/multilingual_task_oriented_dialog_slotfilling.zip"
  unzip "$directory/multilingual_task_oriented_dialog_slotfilling.zip" -d "$directory"
}

glove_6B() {
  local directory="./data/glove_6B"
  mkdir -p "$directory"
  wget -N -P "$directory" "http://nlp.stanford.edu/data/glove.6B.zip"
  unzip "$directory/glove.6B.zip" -d "$directory"
}

# execute all functions
check_help "$@"
facebook_multi_task_nlu
glove_6B
