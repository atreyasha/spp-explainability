#!/usr/bin/env bash
# This script sets up a pre-commit hook for use
set -e

# usage function
usage() {
  cat <<EOF
Usage: setup_git_hooks.sh [-h|--help]
Force copy git hooks to git repository config

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

# define function
setup_git_hooks() {
  local input
  for input in ./hooks/*.sh; do
    cp "$input" "./.git/hooks/$(basename ${input%%.sh})"
  done
}

# execute function
check_help "$@"
setup_git_hooks
