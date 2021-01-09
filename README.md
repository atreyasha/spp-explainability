## SoPa++

This repository documents thesis research with the working title *"SoPa++: Leveraging explainability from hybridized RNN, CNN and weighted finite-state neural architectures"*.

### Dependencies :neckbeard:

This repository's code was tested with Python versions `3.7.*`. To sync dependencies, we recommend creating a virtual environment and installing the relevant packages via `pip`:

```shell
pip install -r requirements.txt
```

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
                    [--epochs <int>] [--epsilon-scale <float>] [--gpu]
                    [--gpu-device <str>] [--grid-config <file_path>]
                    [--grid-training] [--learning-rate <float>]
                    [--logging-level {debug,info,warning,error,critical}]
                    [--max-doc-len <int>] [--models-directory <dir_path>]
                    [--no-epsilons] [--no-self-loops]
                    [--num-random-iterations <int>] [--num-threads <int>]
                    [--num-train-instances <int>] [--patience <int>]
                    [--patterns <str>] [--pre-computed-patterns <file_path>]
                    [--scheduler-factor <float>] [--scheduler-patience <int>]
                    [--seed <int>] [--self-loop-scale <float>]
                    [--semiring {MaxSumSemiring,MaxProductSemiring,ProbabilitySemiring}]
                    [--shared-self-loops {0,1,2}] [--static-embeddings]
                    [--tqdm-update-freq <int>] [--word-dropout <float>]

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
                           Maximum number of training epochs (default: 200)
  --learning-rate          <float>
                           Learning rate for Adam optimizer (default: 0.001)
  --max-doc-len            <int>
                           Maximum document length allowed. -1 refers to no
                           length restriction (default: -1)
  --models-directory       <dir_path>
                           Base directory where all models will be saved
                           (default: ./models)
  --num-train-instances    <int>
                           Maximum number of training instances (default:
                           None)
  --patience               <int>
                           Number of epochs with no improvement after which
                           training will be stopped (default: 30)
  --pre-computed-patterns  <file_path>
                           Path to file containing per-computed patterns
                           (default: None)
  --scheduler-factor       <float>
                           Factor by which the learning rate will be reduced
                           (default: 0.1)
  --scheduler-patience     <int>
                           Number of epochs with no improvement after which
                           learning rate will be reduced (default: 10)
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
  --num-threads            <int>
                           Set the number of threads used for intraop
                           parallelism on CPU (default: None)

optional sopa-architecture arguments:
  --bias-scale             <float>
                           Scale biases by this parameter (default: None)
  --epsilon-scale          <float>
                           Scale epsilons by this parameter (default: None)
  --no-epsilons            Do not use epsilon transitions (default: False)
  --no-self-loops          Do not use self loops (default: False)
  --patterns               <str>
                           Pattern lengths and counts with the following
                           syntax: PatternLength1-PatternCount1_PatternLength2
                           -PatternCount2_... (default:
                           7-10_6-10_5-10_4-10_3-10_2-10)
  --self-loop-scale        <float>
                           Scale self-loops by this parameter (default: None)
  --semiring               {MaxSumSemiring,MaxProductSemiring,ProbabilitySemiring}
                           Specify which semiring to use (default:
                           MaxSumSemiring)
  --shared-self-loops      {0,1,2}
                           Option to share main path and self loop parameters.
                           0: do not share parameters, 1: share one parameter
                           per state per pattern, 2: share one global
                           parameter (default: 0)
  --static-embeddings      Freeze learning of token embeddings (default:
                           False)

optional logging arguments:
  --logging-level          {debug,info,warning,error,critical}
                           Set logging level (default: info)

optional progress-bar arguments:
  --disable-tqdm           Disable tqdm progress bars (default: False)
  --tqdm-update-freq       <int>
                           Specify after how many training updates should the
                           tqdm progress bar be updated with model diagnostics
                           (default: 1)
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
                           [--num-threads <int>] [--tqdm-update-freq <int>]

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
  --num-threads          <int>
                         Set the number of threads used for intraop
                         parallelism on CPU (default: None)

optional logging arguments:
  --logging-level        {debug,info,warning,error,critical}
                         Set logging level (default: info)

optional progress-bar arguments:
  --disable-tqdm         Disable tqdm progress bars (default: False)
  --tqdm-update-freq     <int>
                         Specify after how many training updates should the
                         tqdm progress bar be updated with model diagnostics
                         (default: 1)
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
                       [--gpu] [--gpu-device <str>]
                       [--logging-level {debug,info,warning,error,critical}]
                       [--num-threads <int>] [--output-prefix <str>]

optional arguments:
  -h, --help          show this help message and exit

required evaluation arguments:
  --eval-data         <file_path>
                      Path to evaluation data file (default: None)
  --eval-labels       <file_path>
                      Path to evaluation labels file (default: None)
  --model-checkpoint  <glob_path>
                      Glob path to model checkpoint with '.pt' extension. Note
                      that 'model_config.json' must be in the same directory
                      level as the model checkpoint file (default: None)

optional evaluation arguments:
  --batch-size        <int>
                      Batch size for training (default: 256)
  --output-prefix     <str>
                      Prefix for output classification report (default: test)

optional hardware-acceleration arguments:
  --gpu               Use GPU hardware acceleration (default: False)
  --gpu-device        <str>
                      GPU device specification in case --gpu option is used
                      (default: cuda:0)
  --num-threads       <int>
                      Set the number of threads used for intraop parallelism
                      on CPU (default: None)

optional logging arguments:
  --logging-level     {debug,info,warning,error,critical}
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
bash scripts/evaluate_spp_cpu.sh "/glob/to/model/checkpoints"
```

To evaluate grid-based SoPa++ models using our defaults on a single GPU, execute:

```shell
bash scripts/evaluate_spp_gpu.sh "/glob/to/model/checkpoints"
```

</p>
</details>

### Development :snail:

Ongoing development of this repository is documented in this [log](./docs/develop.md).
