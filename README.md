## SoPa++

This repository documents thesis research with the working title *"SoPa++: Leveraging performance and explainability from hybridized RNN, CNN and weighted finite-state neural architectures"*.

### Dependencies :neckbeard:

This repository's code was tested with Python versions `3.7.*`. To sync dependencies, we recommend creating a virtual environment and installing the relevant packages via `pip`:

```shell
pip install -r requirements.txt
```

### Repository initialization :fire:

1. Download [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings and Facebook's multi-class Natural Language Understanding (NLU) [data set](https://github.com/nghuyong/rasa-nlu-benchmark):

    ```shell
    bash scripts/prepare_data.sh
    ```

2. **Optional:** Set up git hooks to manage development workflows:

    ```shell
    bash scripts/setup_git_hooks.sh
    ```

### Usage :snowflake:

#### i. Preprocessing

In order to preprocess Facebook's multi-class NLU data set, use `./src/preprocess_multiclass_nlu.py`:

```
usage: preprocess_multiclass_nlu.py [-h] [--data-directory <str>]
                                    [--logging-level {debug,info,warning,error,critical}]

optional arguments:
  -h, --help        show this help message and exit

optional preprocessing arguments:
  --data-directory  <str>
                    Data directory containing facebook multi-class NLU data
                    (default: ./data/facebook_multiclass_nlu/)

optional logging arguments:
  --logging-level   {debug,info,warning,error,critical}
                    Set logging level (default: info)
```

This script will format the aforementioned data set and prepare it for downstream use. To run this script with our defaults, simply execute:

```shell
bash scripts/preprocess_multiclass_nlu.sh
```

### Development :snail:

Ongoing development of this repository is documented in this [log](./docs/develop.md).
