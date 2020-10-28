#!/usr/bin/env bash
# Pre-push hook to mirror branch with main

mirror_branch() {
  local main mirror local_hash
  local current_branch stash_entry
  main="$1"
  mirror="$2"
  # exit if main and mirror are undefined
  [[ -z "$main" || -z "$mirror" ]] && exit 1
  # check existence of mirror branch
  git show-ref --verify --quiet "refs/heads/$mirror" || exit 1
  # check current branch
  current_branch="$(git rev-parse --abbrev-ref HEAD)"
  # start mirroring
  if [ "$current_branch" == "$main" ]; then
    printf "%s\n" "Syncing $mirror with $main"
    local_hash=$(
      date +%s | sha256sum | base64 | head -c 32
      echo
    )
    git stash push -u -m "$local_hash"
    git checkout "$mirror"
    git rebase master
    git push --force origin "$mirror"
    git checkout master
    stash_entry=$(git stash list | grep "$local_hash" | grep -Eo "stash@{[0-9]+}")
    if [ -n "$stash_entry" ]; then
      git stash apply "$stash_entry"
      git stash drop "$stash_entry"
    fi
  fi
}

# main call to function
# NOTE: user edits go here
mirror_branch "main" "mirror"
