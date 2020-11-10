## SoPa++

This repository documents thesis research with the working title *"SoPa++: Leveraging performance and explainability from hybridized RNN, CNN and weighted finite-state neural architectures"*.

### Dependencies :neckbeard:

This repository's code was tested with Python versions `3.6+`. To sync dependencies, we recommend creating a virtual environment and installing the relevant packages via `pip`:

```shell
pip install -r requirements.txt
```

### Repository initialization :fire:

1. Automatically download relevant data for this repository:

    ```shell
    bash scripts/prepare_data.sh
    ```

2. **Optional:** Set up git hooks to manage development workflows:

    ```shell
    bash scripts/setup_git_hooks.sh
    ```

### Development :snail:

Ongoing development of this repository is documented in this [log](./docs/develop.md).
