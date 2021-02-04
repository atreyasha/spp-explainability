## SoPa++

This repository documents thesis research with the working title *"SoPa++: Leveraging explainability from hybridized RNN, CNN and weighted finite-state neural architectures"*.

### Dependencies :neckbeard:

This repository's code was tested with Python versions `3.7.*`. To sync dependencies, we recommend creating a virtual environment and installing the relevant packages via `pip`:

```shell
pip install -r requirements.txt
```

**Note:** If you intend to use the GPU, the `torch==1.7.0` dependency in `requirements.txt` works out-of-the-box with CUDA version `10.2`. If you have a different version of CUDA, refer to the official [PyTorch](https://pytorch.org/get-started/locally/) webpage for alternative `pip` installation commands which will provide `torch` optimized for your CUDA version.

### Repository initialization :fire:

1. Download and prepare [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings and Facebook's multi-class Natural Language Understanding (NLU) [data set](https://research.fb.com/publications/cross-lingual-transfer-learning-for-multilingual-task-oriented-dialog/):

    ```shell
    bash scripts/prepare_data.sh
    ```

2. **Optional:** Set up git hooks to manage development workflows such as formatting shell scripts and keeping python dependencies up-to-date:

    ```shell
    bash scripts/setup_git_hooks.sh
    ```

### Usage :snowflake:

<details><summary><strong>i. Preprocessing</strong></summary>
<p>

For preprocessing Facebook's multi-class NLU data set, we use `src/preprocess_multiclass_nlu.py`:

```
usage: preprocess_multiclass_nlu.py [-h] [--data-directory <dir_path>]
                                    [--disable-upsampling]
                                    [--logging-level {debug,info,warning,error,critical}]
                                    [--truecase]

optional arguments:
  -h, --help            show this help message and exit

optional preprocessing arguments:
  --data-directory      <dir_path>
                        Data directory containing clean input data (default:
                        ./data/facebook_multiclass_nlu/)
  --disable-upsampling  Disable upsampling on the train and validation data
                        sets (default: False)
  --truecase            Retain true casing when preprocessing data. Otherwise
                        data will be lowercased by default (default: False)

optional logging arguments:
  --logging-level       {debug,info,warning,error,critical}
```

The default workflow cleans the original NLU data, forces it to lowercased format and upsamples all minority classes. To run the default workflow, execute:

```shell
bash scripts/preprocess_multiclass_nlu.sh
```

</p>
</details>

<details><summary><strong>ii. Training</strong></summary>
<p>

For training of the SoPa++ model, we use `src/train_spp.py`:

```
usage: train_spp.py [-h] --embeddings <file_path> --train-data <file_path>
                    --train-labels <file_path> --valid-data <file_path>
                    --valid-labels <file_path> [--batch-size <int>]
                    [--bias-scale <float>] [--clip-threshold <float>]
                    [--disable-scheduler] [--disable-tqdm] [--dropout <float>]
                    [--epochs <int>] [--gpu] [--gpu-device <str>]
                    [--grid-config <file_path>] [--grid-training]
                    [--learning-rate <float>]
                    [--logging-level {debug,info,warning,error,critical}]
                    [--max-doc-len <int>] [--models-directory <dir_path>]
                    [--no-wildcards] [--num-random-iterations <int>]
                    [--num-train-instances <int>] [--patience <int>]
                    [--patterns <str>] [--pre-computed-patterns <file_path>]
                    [--scheduler-factor <float>] [--scheduler-patience <int>]
                    [--seed <int>]
                    [--semiring {MaxSumSemiring,MaxProductSemiring}]
                    [--static-embeddings] [--torch-num-threads <int>]
                    [--tqdm-update-period <int>] [--wildcard-scale <float>]
                    [--word-dropout <float>]

optional arguments:
  -h, --help               show this help message and exit

required training arguments:
  --embeddings             <file_path>
                           Path to GloVe token embeddings file (default: None)
  --train-data             <file_path>
                           Path to train data file (default: None)
  --train-labels           <file_path>
                           Path to train labels file (default: None)
  --valid-data             <file_path>
                           Path to validation data file (default: None)
  --valid-labels           <file_path>
                           Path to validation labels file (default: None)

optional training arguments:
  --batch-size             <int>
                           Batch size for training (default: 256)
  --clip-threshold         <float>
                           Gradient clipping threshold (default: None)
  --disable-scheduler      Disable learning rate scheduler which reduces
                           learning rate on performance plateau (default:
                           False)
  --dropout                <float>
                           Neuron dropout probability (default: 0.2)
  --epochs                 <int>
                           Maximum number of training epochs (default: 50)
  --learning-rate          <float>
                           Learning rate for Adam optimizer (default: 0.001)
  --max-doc-len            <int>
                           Maximum document length allowed (default: None)
  --models-directory       <dir_path>
                           Base directory where all models will be saved
                           (default: ./models)
  --num-train-instances    <int>
                           Maximum number of training instances (default:
                           None)
  --patience               <int>
                           Number of epochs with no improvement after which
                           training will be stopped (default: 10)
  --pre-computed-patterns  <file_path>
                           Path to file containing per-computed patterns
                           (default: None)
  --scheduler-factor       <float>
                           Factor by which the learning rate will be reduced
                           (default: 0.1)
  --scheduler-patience     <int>
                           Number of epochs with no improvement after which
                           learning rate will be reduced (default: 5)
  --seed                   <int>
                           Global random seed for numpy and torch (default:
                           42)
  --word-dropout           <float>
                           Word dropout probability (default: 0.2)

optional grid-training arguments:
  --grid-config            <file_path>
                           Path to grid configuration file (default:
                           ./src/resources/flat_grid_light_config.json)
  --grid-training          Use grid-training instead of single-training
                           (default: False)
  --num-random-iterations  <int>
                           Number of random iteration(s) for each grid
                           instance (default: 1)

optional hardware-acceleration arguments:
  --gpu                    Use GPU hardware acceleration (default: False)
  --gpu-device             <str>
                           GPU device specification in case --gpu option is
                           used (default: cuda:0)
  --torch-num-threads      <int>
                           Set the number of threads used for CPU intraop
                           parallelism with PyTorch (default: None)

optional sopa-architecture arguments:
  --bias-scale             <float>
                           Scale biases by this parameter (default: None)
  --no-wildcards           Do not use wildcard transitions (default: False)
  --patterns               <str>
                           Pattern lengths and counts with the following
                           syntax: PatternLength1-PatternCount1_PatternLength2
                           -PatternCount2_... (default: 6-25_5-25_4-25_3-25)
  --semiring               {MaxSumSemiring,MaxProductSemiring}
                           Specify which semiring to use (default:
                           MaxSumSemiring)
  --static-embeddings      Freeze learning of token embeddings (default:
                           False)
  --wildcard-scale         <float>
                           Scale wildcard(s) by this parameter (default: None)

optional logging arguments:
  --logging-level          {debug,info,warning,error,critical}
                           Set logging level (default: info)

optional progress-bar arguments:
  --disable-tqdm           Disable tqdm progress bars (default: False)
  --tqdm-update-period     <int>
                           Specify after how many training updates should the
                           tqdm progress bar be updated with model diagnostics
                           (default: 5)
```

#### Single SoPa++ model training

To train a single SoPa++ model using our defaults on the CPU, execute:

```shell
bash scripts/train_spp_cpu.sh
```

To train a single SoPa++ model using our defaults on a single GPU, execute:

```shell
bash scripts/train_spp_gpu.sh
```

#### Grid SoPa++ model training

To apply grid-based training on SoPa++ models using our defaults on the CPU, execute:

```shell
bash scripts/train_spp_grid_cpu.sh
```

To apply grid-based training on SoPa++ models using our defaults on a single GPU, execute:

```shell
bash scripts/train_spp_grid_gpu.sh
```

</p>
</details>

<details><summary><strong>iii. Resume training</strong></summary>
<p>

For resuming the aforementioned training workflow in case of interruptions, we use `src/train_resume_spp.py`:

```
usage: train_resume_spp.py [-h] --model-log-directory <dir_path>
                           [--disable-tqdm] [--gpu] [--gpu-device <str>]
                           [--grid-training]
                           [--logging-level {debug,info,warning,error,critical}]
                           [--torch-num-threads <int>]
                           [--tqdm-update-period <int>]

optional arguments:
  -h, --help             show this help message and exit

required training arguments:
  --model-log-directory  <dir_path>
                         Base model directory containing model data to be
                         resumed for training (default: None)

optional grid-training arguments:
  --grid-training        Use grid-training instead of single-training
                         (default: False)

optional hardware-acceleration arguments:
  --gpu                  Use GPU hardware acceleration (default: False)
  --gpu-device           <str>
                         GPU device specification in case --gpu option is used
                         (default: cuda:0)
  --torch-num-threads    <int>
                         Set the number of threads used for CPU intraop
                         parallelism with PyTorch (default: None)

optional logging arguments:
  --logging-level        {debug,info,warning,error,critical}
                         Set logging level (default: info)

optional progress-bar arguments:
  --disable-tqdm         Disable tqdm progress bars (default: False)
  --tqdm-update-period   <int>
                         Specify after how many training updates should the
                         tqdm progress bar be updated with model diagnostics
                         (default: 5)
```

#### Resume single SoPa++ model training

To resume training of a single SoPa++ model using our defaults on the CPU, execute:

```shell
bash scripts/train_resume_spp_cpu.sh /path/to/model/directory
```

To resume training of a single SoPa++ model using our defaults on a single GPU, execute:

```shell
bash scripts/train_resume_spp_gpu.sh /path/to/model/directory
```

#### Resume grid SoPa++ model training

To resume grid-based training of SoPa++ models using our defaults on the CPU, execute:

```shell
bash scripts/train_resume_spp_grid_cpu.sh /path/to/model/directory
```

To resume grid-based training of SoPa++ models using our defaults on a single GPU, execute:

```shell
bash scripts/train_resume_spp_grid_gpu.sh /path/to/model/directory
```

</p>
</details>

<details><summary><strong>iv. Evaluation</strong></summary>
<p>

For evaluating a trained SoPa++ model, we use `src/evaluate_spp.py`:

```
usage: evaluate_spp.py [-h] --eval-data <file_path> --eval-labels <file_path>
                       --model-checkpoint <glob_path> [--batch-size <int>]
                       [--evaluation-metric {recall,precision,f1-score,accuracy}]
                       [--evaluation-metric-type {weighted avg,macro avg}]
                       [--gpu] [--gpu-device <str>] [--grid-evaluation]
                       [--logging-level {debug,info,warning,error,critical}]
                       [--max-doc-len <int>] [--output-prefix <str>]
                       [--torch-num-threads <int>]

optional arguments:
  -h, --help                show this help message and exit

required evaluation arguments:
  --eval-data               <file_path>
                            Path to evaluation data file (default: None)
  --eval-labels             <file_path>
                            Path to evaluation labels file (default: None)
  --model-checkpoint        <glob_path>
                            Glob path to model checkpoint with '.pt' extension
                            (default: None)

optional evaluation arguments:
  --batch-size              <int>
                            Batch size for evaluation (default: 256)
  --max-doc-len             <int>
                            Maximum document length allowed (default: None)
  --output-prefix           <str>
                            Prefix for output classification report (default:
                            test)

optional grid-evaluation arguments:
  --evaluation-metric       {recall,precision,f1-score,accuracy}
                            Specify which evaluation metric to use for
                            comparison (default: f1-score)
  --evaluation-metric-type  {weighted avg,macro avg}
                            Specify which type of evaluation metric to use
                            (default: weighted avg)
  --grid-evaluation         Use grid-evaluation framework to find/summarize
                            best model (default: False)

optional hardware-acceleration arguments:
  --gpu                     Use GPU hardware acceleration (default: False)
  --gpu-device              <str>
                            GPU device specification in case --gpu option is
                            used (default: cuda:0)
  --torch-num-threads       <int>
                            Set the number of threads used for CPU intraop
                            parallelism with PyTorch (default: None)

optional logging arguments:
  --logging-level           {debug,info,warning,error,critical}
                            Set logging level (default: info)
```

#### Single SoPa++ model evaluation

To evaluate a single SoPa++ model using our defaults on the CPU, execute:

```shell
bash scripts/evaluate_spp_cpu.sh /path/to/model/checkpoint
```

To evaluate a single SoPa++ model using our defaults on a single GPU, execute:

```shell
bash scripts/evaluate_spp_gpu.sh /path/to/model/checkpoint
```

#### Grid SoPa++ model evaluation

To evaluate grid-based SoPa++ models using our defaults on the CPU, execute:

```shell
bash scripts/evaluate_spp_grid_cpu.sh "/glob/to/model/*/checkpoints"
```

To evaluate grid-based SoPa++ models using our defaults on a single GPU, execute:

```shell
bash scripts/evaluate_spp_grid_gpu.sh "/glob/to/model/*/checkpoints"
```

</p>
</details>

<details><summary><strong>v. Explainability</strong></summary>
<p>

For explaining a SoPa++ model by simplifying it into an ensemble of weighted regular expressions, we use `src/explain_simplify_spp.py`:

```
usage: explain_simplify_spp.py [-h] --model-checkpoint <file_path>
                               --train-data <file_path> --train-labels
                               <file_path> --valid-data <file_path>
                               --valid-labels <file_path> [--disable-tqdm]
                               [--gpu] [--gpu-device <str>]
                               [--logging-level {debug,info,warning,error,critical}]
                               [--max-doc-len <int>]
                               [--num-train-instances <int>]
                               [--torch-num-threads <int>]
                               [--tqdm-update-period <int>]

optional arguments:
  -h, --help             show this help message and exit

required explainability arguments:
  --model-checkpoint     <file_path>
                         Path to model checkpoint with '.pt' extension
                         (default: None)
  --train-data           <file_path>
                         Path to train data file (default: None)
  --train-labels         <file_path>
                         Path to train labels file (default: None)
  --valid-data           <file_path>
                         Path to validation data file (default: None)
  --valid-labels         <file_path>
                         Path to validation labels file (default: None)

optional explainability arguments:
  --max-doc-len          <int>
                         Maximum document length allowed (default: None)
  --num-train-instances  <int>
                         Maximum number of training instances (default: None)

optional hardware-acceleration arguments:
  --gpu                  Use GPU hardware acceleration (default: False)
  --gpu-device           <str>
                         GPU device specification in case --gpu option is used
                         (default: cuda:0)
  --torch-num-threads    <int>
                         Set the number of threads used for CPU intraop
                         parallelism with PyTorch (default: None)

optional logging arguments:
  --logging-level        {debug,info,warning,error,critical}
                         Set logging level (default: info)

optional progress-bar arguments:
  --disable-tqdm         Disable tqdm progress bars (default: False)
  --tqdm-update-period   <int>
                         Specify after how many training updates should the
                         tqdm progress bar be updated with model diagnostics
                         (default: 5)
```

To explain a single SoPa++ model using our defaults on the CPU, execute:

```shell
bash scripts/explain_simplify_spp_cpu.sh /path/to/model/checkpoint
```

To explain a single SoPa++ model using our defaults on a single GPU, execute:

```shell
bash scripts/explain_simplify_spp_gpu.sh /path/to/model/checkpoint
```

</p>
</details>

### Development :snail:

Ongoing development of this repository is documented in this [log](./docs/develop.md).
