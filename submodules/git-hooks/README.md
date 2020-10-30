## Git hooks :anchor:

This repository documents two git hooks which assist with `python`, `shell`, `R` and `org-mode` development workflows; as well as remote branch mirroring.

### Overview :book:

#### Pre-commit hook

`pre-commit.sh` contains a useful hook which is, from its name, a workflow that is executed before every commit. The various functions and dependencies in the shell script are described below:

| Function                   | Description                                                                                                         | Dependencies                                      |
| :-------------             | :-------------                                                                                                      | :-----                                            |
| update_python_dependencies | Updates `requirements.txt` to maintain log of python dependencies                                                   | [poetry](https://github.com/python-poetry/poetry) |
| format_shell_scripts       | Formats all shell scripts with consistent indents                                                                   | [shfmt](https://github.com/mvdan/sh)              |
| format_R_scripts           | Formats all R scripts for clean code                                                                                | [styler](https://github.com/r-lib/styler)         |
| convert_org_to_md          | Converts specified `org` file(s) to github-flavored `markdown`, adds a `TOC` and cleans up `TODO` and `DONE` markers | [pandoc](https://github.com/jgm/pandoc)           |

In addition, we provide a `main` function where the user can decide which of the above functions to use; as well as fine-tune the input parameters.

#### Pre-push hook

`pre-push.sh` contains a simpler hook which, from its name, executes a workflow before pushing commits upstream. Here, we provide only one function:

| Function       | Description                                                                                                                                   | Dependencies                                      |
| :------------- | :-------------                                                                                                                                | :-----                                            |
| mirror_branch  | Mirrors a named branch with another main branch. This could be useful to keep one branch up-to-date with another while still offering new features.  | -                                                 |

The names of the main and mirror branches can be specified in the `main` function.

### Usage :cyclone:

1. Edit the `main` function(s) of the hooks to customize callable functions and input parameters.

2. In order to initialize both hooks, copy the edited hooks to `./git/hooks/` in your desired `git` repository and remove the `.sh` extension. For example:

    ```shell
    $ cp /path/to/pre-commit.sh ./git/hooks/pre-commit
    $ cp /path/to/pre-push.sh ./git/hooks/pre-push
    ```

**Note:** These hooks are generally non-invasive, ie. they exit gracefully if dependencies or staged changes are missing and do not interfere with the overall commit or push process in case of failures.

### Bugs/Issues :bug:

In case of bugs or suggestions for improvements, feel free to open a GitHub issue.

<!--  LocalWords:  Pre md github ie
 -->
