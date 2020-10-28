#!/usr/bin/env bash
# Pre-commit git hook which offers multiple functionalities,
# including updating python dependencies, formatting shell and R scripts
# and converting an org document to markdown for logging purposes

find_non_deleted_staged() {
  # function to check if input has been staged
  local input="$1"
  git diff --name-only --cached --diff-filter=d "$input" 2>/dev/null
}

update_python_dependencies() {
  # function to synchronize poetry.lock with requirements.txt
  local poetry_staged
  # update requirements.txt given conditions
  if command -v poetry &>/dev/null; then
    mapfile -t poetry_staged < <(find_non_deleted_staged "poetry.lock")
    if [ "${#poetry_staged[@]}" -ne "0" ]; then
      printf "%s\n" "Syncing python dependencies with poetry"
      poetry export -f requirements.txt --without-hashes -o requirements.txt
      git add "requirements.txt"
    fi
  fi
}

format_shell_scripts() {
  # function to format all shell files
  local shell_staged
  # format staged shell scripts
  if command -v shfmt &>/dev/null; then
    mapfile -t shell_staged < <(find_non_deleted_staged "*.sh")
    if [ "${#shell_staged[@]}" -ne "0" ]; then
      printf "%s\n" "Formatting shell scripts with shfmt"
      shfmt -w -i 2 "${shell_staged[@]}"
      git add "${shell_staged[@]}"
    fi
  fi
}

format_R_scripts() {
  # function to format all R files
  local R_staged
  # format staged R files
  if Rscript -e 'styler::style_file' &>/dev/null; then
    mapfile -t R_staged < <(find_non_deleted_staged "*.R")
    if [ "${#R_staged[@]}" -ne "0" ]; then
      printf "%s\n" "Formatting R scripts"
      for R_file in "${R_staged[@]}"; do
        Rscript -e "styler::style_file(\"$R_file\")"
      done
      git add "${R_staged[@]}"
    fi
  fi
}

convert_org_to_md() {
  # function to convert org doc to markdown
  local input output org_staged
  input="$1"
  output="${input//.org/.md}"
  # conduct checks and convert
  if command -v pandoc &>/dev/null; then
    mapfile -t org_staged < <(find_non_deleted_staged "$input")
    if [ "${#org_staged[@]}" -eq "1" ]; then
      printf "%s\n" "Converting relevant org files to markdown"
      # basic conversion to markdown
      pandoc -f org -t markdown -o "$output" "$input"
      # add TOC to markdown
      pandoc -s -t markdown --toc -o "$output" "$output"
      # add TOC title
      sed -i '1 i\## Table of Contents' "$output"
      # replace org-agenda markers cleanly
      sed -i 's/\[TODO\]{.*}/**TODO**/g; s/\[DONE\]{.*}/**DONE**/g' "$output"
      # replace startup visibility cleanly
      sed -i '/```{=org}/,/```/d' "$output"
      # stage new markdown for commit
      git add "$output"
    fi
  fi
}

main() {
  # main call to functions
  # NOTE: user edit(s) go here
  update_python_dependencies
  format_shell_scripts
  format_R_scripts
  convert_org_to_md "./docs/develop.org"
}

main
