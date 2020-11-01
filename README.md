## Explainable Natural Language Processing

This repository documents research into explainable and high-performance techniques in NLP. An interim manuscript can be seen [here](./docs/manuscript/main.pdf).

### Dependencies :neckbeard:

This repository's code was tested with Python versions `3.6+`. To sync dependencies, we recommend creating a virtual environment and installing the relevant packages via `pip`:

```shell
$ pip install -r requirements.txt
```

### Repository initialization :fire:

1. Automatically download relevant data for this repository:

    ```shell
    $ bash scripts/prepare_data.sh
    ```

2. **Optional:** Set up git hooks to keep `requirements.txt` up-to-date, format shell and `R` files and keep the development log synced:

    ```shell
    $ bash scripts/setup_git_hooks.sh
    ```

### Development :snail:

Ongoing development of this repository is documented in this [log](./docs/develop.md).
