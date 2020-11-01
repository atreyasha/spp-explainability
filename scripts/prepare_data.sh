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
nlu_intent_small() {
  local directory="./data/nlu_intent_small"
  mkdir -p "$directory"
  wget -N -P "$directory" "https://raw.githubusercontent.com/sebischair/NLU-Evaluation-Corpora/master/AskUbuntuCorpus.json"
  wget -N -P "$directory" "https://raw.githubusercontent.com/sebischair/NLU-Evaluation-Corpora/master/ChatbotCorpus.json"
  wget -N -P "$directory" "https://raw.githubusercontent.com/sebischair/NLU-Evaluation-Corpora/master/WebApplicationsCorpus.json"
}

# execute all functions
check_help "$@"
nlu_intent_small
