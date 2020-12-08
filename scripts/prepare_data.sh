#!/usr/bin/env bash
# Download key data sets for SoPa++
set -e

# usage function
usage() {
  cat <<EOF
Usage: prepare_data.sh [-h|--help]
Prepare data sets for SoPa++

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

# download and prepare Facebook multi-class NLU data set
facebook_multi_task_nlu() {
  local directory="./data/facebook_multiclass_nlu/raw"
  mkdir -p "$directory"
  wget -N -P "$directory" "https://download.pytorch.org/data/multilingual_task_oriented_dialog_slotfilling.zip"
  unzip "$directory/multilingual_task_oriented_dialog_slotfilling.zip" -d "$directory"
}

# download GloVe 6B word vectors
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
