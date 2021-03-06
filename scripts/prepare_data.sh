#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
Usage: prepare_data.sh [-h|--help]

Download and prepare GloVe-6B uncased word-vectors
and the FMTOD data set to be used in this repository

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

# download and prepare Facebook multi-class NLU data set
fmtod() {
  local directory="./data/fmtod/raw"
  mkdir -p "$directory"
  wget -N -P "$directory" "https://download.pytorch.org/data/multilingual_task_oriented_dialog_slotfilling.zip"
  unzip "$directory/multilingual_task_oriented_dialog_slotfilling.zip" -d "$directory"
}

# download and prepare GloVe 6B uncased word vectors
glove_6B() {
  local directory="./data/glove_6B_uncased"
  mkdir -p "$directory"
  wget -N -P "$directory" "http://nlp.stanford.edu/data/glove.6B.zip"
  unzip "$directory/glove.6B.zip" -d "$directory"
}

# execute all functions
check_help "$@"
fmtod
glove_6B
