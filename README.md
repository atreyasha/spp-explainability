# SoPa++

This repository documents M.Sc. thesis research titled *"SoPa++: Leveraging explainability from hybridized RNN, CNN and weighted finite-state neural architectures"*. The manuscript for this thesis can be found [here](./docs/manuscript/main.pdf).

## Dependencies :neckbeard:

1. This repository's code was tested with Python versions `3.7.*`. To sync dependencies, we recommend creating a virtual environment and installing the relevant packages via `pip`:

    ```shell
    pip install -r requirements.txt
    ```

    **Note:** If you intend to use the GPU, the `torch==1.7.0` dependency in `requirements.txt` works out-of-the-box with CUDA version `10.2`. If you have a different version of CUDA, refer to the official [PyTorch](https://pytorch.org/get-started/locally/) webpage for alternative `pip` installation commands which will provide `torch` optimized for your CUDA version.

2. We use `R` for visualizations integrated with `TikZ` and `ggplot`. Below is the `sessionInfo()` output, which can be used for replicating our dependencies explicitly.

    ```
    R version 4.0.4 (2021-02-15)
    Platform: x86_64-pc-linux-gnu (64-bit)
    Running under: Arch Linux

    Matrix products: default
    BLAS:   /usr/lib/libblas.so.3.9.0
    LAPACK: /usr/lib/liblapack.so.3.9.0

    locale:
     [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C              
     [3] LC_TIME=en_US.UTF-8        LC_COLLATE=en_US.UTF-8    
     [5] LC_MONETARY=en_US.UTF-8    LC_MESSAGES=en_US.UTF-8   
     [7] LC_PAPER=en_US.UTF-8       LC_NAME=C                 
     [9] LC_ADDRESS=C               LC_TELEPHONE=C            
    [11] LC_MEASUREMENT=en_US.UTF-8 LC_IDENTIFICATION=C       

    attached base packages:
    [1] tools     stats     graphics  grDevices utils     datasets  methods  
    [8] base     

    other attached packages:
    [1] RColorBrewer_1.1-2  plyr_1.8.6          reshape2_1.4.4     
    [4] optparse_1.6.6      tikzDevice_0.12.3.1 rjson_0.2.20       
    [7] ggh4x_0.1.2.1       ggplot2_3.3.3      
    ```

## Repository initialization :fire:

1. Download and prepare [GloVe-6B](https://nlp.stanford.edu/projects/glove/) word embeddings and the Facebook Multilingual Task Oriented Dialogue (FMTOD) [data set](https://research.fb.com/publications/cross-lingual-transfer-learning-for-multilingual-task-oriented-dialog/):

    ```shell
    bash scripts/prepare_data.sh
    ```

2. **Optional:** Prepare git hooks to manage development workflows such as formatting shell scripts and keeping python dependencies up-to-date:

    ```shell
    bash scripts/prepare_git_hooks.sh
    ```

## Usage :snowflake:

### Neural SoPa++

<details><summary>i. Preprocessing</summary>
<p>

For preprocessing the FMTOD data set, we use `src/preprocess_fmtod.py`:

```
usage: preprocess_fmtod.py [-h] [--data-directory <dir_path>]
                           [--disable-upsampling]
                           [--logging-level {debug,info,warning,error,critical}]
                           [--truecase]

optional arguments:
  -h, --help            show this help message and exit

optional preprocessing arguments:
  --data-directory      <dir_path>
                        Data directory containing clean input data (default:
                        ./data/fmtod/)
  --disable-upsampling  Disable upsampling on the train and validation data
                        sets (default: False)
  --truecase            Retain true casing when preprocessing data. Otherwise
                        data will be lowercased by default (default: False)

optional logging arguments:
  --logging-level       {debug,info,warning,error,critical}
                        Set logging level (default: info)
```

The default workflow cleans the original FMTOD data, forces it to lowercased format and upsamples all minority classes. To run the default workflow, execute:

```shell
bash scripts/preprocess_fmtod.sh
```

</p>
</details>

<details><summary>ii. Training</summary>
<p>

For training the neural SoPa++ model, we use `src/train_spp.py`:

```
usage: train_spp.py [-h] --embeddings <file_path> --train-data <file_path>
                    --train-labels <file_path> --valid-data <file_path>
                    --valid-labels <file_path> [--batch-size <int>]
                    [--bias-scale <float>] [--clip-threshold <float>]
                    [--disable-scheduler] [--disable-tqdm] [--dropout <float>]
                    [--epochs <int>] [--evaluation-period <int>] [--gpu]
                    [--gpu-device <str>] [--grid-config <file_path>]
                    [--grid-training] [--learning-rate <float>]
                    [--logging-level {debug,info,warning,error,critical}]
                    [--max-doc-len <int>] [--models-directory <dir_path>]
                    [--no-wildcards] [--num-random-iterations <int>]
                    [--num-train-instances <int>] [--only-epoch-eval]
                    [--patience <int>] [--patterns <str>]
                    [--scheduler-factor <float>] [--scheduler-patience <int>]
                    [--seed <int>]
                    [--semiring {MaxSumSemiring,MaxProductSemiring}]
                    [--static-embeddings] [--tau-threshold <float>]
                    [--torch-num-threads <int>] [--tqdm-update-period <int>]
                    [--wildcard-scale <float>] [--word-dropout <float>]

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
  --evaluation-period      <int>
                           Specify after how many training updates should
                           model evaluation(s) be conducted. Evaluation will
                           always be conducted at the end of epochs (default:
                           100)
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
  --only-epoch-eval        Only evaluate model at the end of epoch, instead of
                           evaluation by updates (default: False)
  --patience               <int>
                           Number of epochs with no improvement after which
                           training will be stopped (default: 10)
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
                           ./src/resources/flat_grid_heavy_config.json)
  --grid-training          Use grid-training instead of single-training
                           (default: False)
  --num-random-iterations  <int>
                           Number of random iteration(s) for each grid
                           instance (default: 10)

optional spp-architecture arguments:
  --bias-scale             <float>
                           Scale biases by this parameter (default: 1.0)
  --no-wildcards           Do not use wildcard transitions (default: False)
  --patterns               <str>
                           Pattern lengths and counts with the following
                           syntax: PatternLength1-PatternCount1_PatternLength2
                           -PatternCount2_... (default: 6-50_5-50_4-50_3-50)
  --semiring               {MaxSumSemiring,MaxProductSemiring}
                           Specify which semiring to use (default:
                           MaxSumSemiring)
  --static-embeddings      Freeze learning of token embeddings (default:
                           False)
  --tau-threshold          <float>
                           Specify value of TauSTE binarizer tau threshold
                           (default: 0.0)
  --wildcard-scale         <float>
                           Scale wildcard(s) by this parameter (default: None)

optional hardware-acceleration arguments:
  --gpu                    Use GPU hardware acceleration (default: False)
  --gpu-device             <str>
                           GPU device specification in case --gpu option is
                           used (default: cuda:0)
  --torch-num-threads      <int>
                           Set the number of threads used for CPU intraop
                           parallelism with PyTorch (default: None)

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

#### Neural SoPa++ model training

To train a single neural SoPa++ model using our defaults on the CPU, execute:

```shell
bash scripts/train_spp.sh
```

To train a single neural SoPa++ model using our defaults on a single GPU, execute:

```shell
bash scripts/train_spp_gpu.sh
```

#### Grid-based neural SoPa++ model training

To apply grid-based training on neural SoPa++ models using our defaults on the CPU, execute:

```shell
bash scripts/train_spp_grid.sh
```

To apply grid-based training on neural SoPa++ models using our defaults on a single GPU, execute:

```shell
bash scripts/train_spp_grid_gpu.sh
```

</p>
</details>

<details><summary>iii. Resume training</summary>
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

#### Resume neural SoPa++ model training

To resume training of a single neural SoPa++ model using our defaults on the CPU, execute:

```shell
bash scripts/train_resume_spp.sh /path/to/model/log/directory
```

To resume training of a single neural SoPa++ model using our defaults on a single GPU, execute:

```shell
bash scripts/train_resume_spp_gpu.sh /path/to/model/log/directory
```

#### Resume grid-based neural SoPa++ model training

To resume grid-based training of neural SoPa++ models using our defaults on the CPU, execute:

```shell
bash scripts/train_resume_spp_grid.sh /path/to/model/log/directory
```

To resume grid-based training of neural SoPa++ models using our defaults on a single GPU, execute:

```shell
bash scripts/train_resume_spp_grid_gpu.sh /path/to/model/log/directory
```

</p>
</details>

<details><summary>iv. Evaluation</summary>
<p>

For evaluating trained neural SoPa++ model(s), we use `src/evaluate_spp.py`:

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
                            Glob path to model checkpoint(s) with '.pt'
                            extension (default: None)

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

#### Neural SoPa++ model evaluation

To evaluate neural SoPa++ model(s) using our defaults on the CPU, execute:

```shell
bash scripts/evaluate_spp.sh "/glob/to/neural/model/*/checkpoint(s)"
```

To evaluate neural SoPa++ model(s) using our defaults on a single GPU, execute:

```shell
bash scripts/evaluate_spp_gpu.sh "/glob/to/neural/model/*/checkpoint(s)"
```

#### Grid-based neural SoPa++ model evaluation

To evaluate grid-based neural SoPa++ models using our defaults on the CPU, execute:

```shell
bash scripts/evaluate_spp_grid.sh "/glob/to/neural/model/*/checkpoints"
```

To evaluate grid-based neural SoPa++ models using our defaults on a single GPU, execute:

```shell
bash scripts/evaluate_spp_grid_gpu.sh "/glob/to/neural/model/*/checkpoints"
```

</p>
</details>

### Regex proxy

<details><summary>i. Explanations by simplification</summary>
<p>

For explaining neural SoPa++ model(s) by simplifying it into a regex proxy model, we use `src/explain_simplify_spp.py`:

```
usage: explain_simplify_spp.py [-h] --neural-model-checkpoint <glob_path>
                               --train-data <file_path> --train-labels
                               <file_path> --valid-data <file_path>
                               --valid-labels <file_path> [--atol <float>]
                               [--batch-size <int>] [--disable-tqdm] [--gpu]
                               [--gpu-device <str>]
                               [--logging-level {debug,info,warning,error,critical}]
                               [--max-doc-len <int>]
                               [--num-train-instances <int>]
                               [--torch-num-threads <int>]
                               [--tqdm-update-period <int>]

optional arguments:
  -h, --help                 show this help message and exit

required explainability arguments:
  --neural-model-checkpoint  <glob_path>
                             Glob path to neural model checkpoint(s) with
                             '.pt' extension (default: None)
  --train-data               <file_path>
                             Path to train data file (default: None)
  --train-labels             <file_path>
                             Path to train labels file (default: None)
  --valid-data               <file_path>
                             Path to validation data file (default: None)
  --valid-labels             <file_path>
                             Path to validation labels file (default: None)

optional explainability arguments:
  --atol                     <float>
                             Specify absolute tolerance when comparing
                             equivalences between tensors (default: 1e-06)
  --batch-size               <int>
                             Batch size for explainability (default: 256)
  --max-doc-len              <int>
                             Maximum document length allowed (default: None)
  --num-train-instances      <int>
                             Maximum number of training instances (default:
                             None)

optional hardware-acceleration arguments:
  --gpu                      Use GPU hardware acceleration (default: False)
  --gpu-device               <str>
                             GPU device specification in case --gpu option is
                             used (default: cuda:0)
  --torch-num-threads        <int>
                             Set the number of threads used for CPU intraop
                             parallelism with PyTorch (default: None)

optional logging arguments:
  --logging-level            {debug,info,warning,error,critical}
                             Set logging level (default: info)

optional progress-bar arguments:
  --disable-tqdm             Disable tqdm progress bars (default: False)
  --tqdm-update-period       <int>
                             Specify after how many training updates should
                             the tqdm progress bar be updated with model
                             diagnostics (default: 5)
```

To simplify neural SoPa++ model(s) using our defaults on the CPU, execute:

```shell
bash scripts/explain_simplify_spp.sh "/glob/to/neural/model/*/checkpoint(s)"
```

To simplify neural SoPa++ model(s) using our defaults on a GPU, execute:

```shell
bash scripts/explain_simplify_spp_gpu.sh "/glob/to/neural/model/*/checkpoint(s)"
```

</p>
</details>

<details><summary>ii. Compression</summary>
<p>

For compressing regex proxy model(s), we use `src/explain_compress_regex.py`:

```
usage: explain_compress_regex.py [-h] --regex-model-checkpoint <glob_path>
                                 [--disable-tqdm]
                                 [--logging-level {debug,info,warning,error,critical}]
                                 [--tqdm-update-period <int>]

optional arguments:
  -h, --help                show this help message and exit

required explainability arguments:
  --regex-model-checkpoint  <glob_path>
                            Glob path to regex model checkpoint(s) with '.pt'
                            extension (default: None)

optional logging arguments:
  --logging-level           {debug,info,warning,error,critical}
                            Set logging level (default: info)

optional progress-bar arguments:
  --disable-tqdm            Disable tqdm progress bars (default: False)
  --tqdm-update-period      <int>
                            Specify after how many training updates should the
                            tqdm progress bar be updated with model
                            diagnostics (default: 5)
```

To compress regex proxy model(s) using our defaults on the CPU, execute:

```shell
bash scripts/explain_compress_regex.sh "/glob/to/regex/model/*/checkpoint(s)"
```

</p>
</details>

<details><summary>iii. Evaluation</summary>
<p>

For evaluating regex proxy model(s), we use `src/evaluate_regex.py`:

```
usage: evaluate_regex.py [-h] --eval-data <file_path> --eval-labels
                         <file_path> --model-checkpoint <glob_path>
                         [--batch-size <int>] [--disable-tqdm] [--gpu]
                         [--gpu-device <str>]
                         [--logging-level {debug,info,warning,error,critical}]
                         [--max-doc-len <int>] [--output-prefix <str>]
                         [--torch-num-threads <int>]
                         [--tqdm-update-period <int>]

optional arguments:
  -h, --help            show this help message and exit

required evaluation arguments:
  --eval-data           <file_path>
                        Path to evaluation data file (default: None)
  --eval-labels         <file_path>
                        Path to evaluation labels file (default: None)
  --model-checkpoint    <glob_path>
                        Glob path to model checkpoint(s) with '.pt' extension
                        (default: None)

optional evaluation arguments:
  --batch-size          <int>
                        Batch size for evaluation (default: 256)
  --max-doc-len         <int>
                        Maximum document length allowed (default: None)
  --output-prefix       <str>
                        Prefix for output classification report (default:
                        test)

optional hardware-acceleration arguments:
  --gpu                 Use GPU hardware acceleration (default: False)
  --gpu-device          <str>
                        GPU device specification in case --gpu option is used
                        (default: cuda:0)
  --torch-num-threads   <int>
                        Set the number of threads used for CPU intraop
                        parallelism with PyTorch (default: None)

optional logging arguments:
  --logging-level       {debug,info,warning,error,critical}
                        Set logging level (default: info)

optional progress-bar arguments:
  --disable-tqdm        Disable tqdm progress bars (default: False)
  --tqdm-update-period  <int>
                        Specify after how many training updates should the
                        tqdm progress bar be updated with model diagnostics
                        (default: 5)
```

To evaluate regex proxy model(s) using our defaults on the CPU, execute:

```shell
bash scripts/evaluate_regex.sh "/glob/to/regex/model/*/checkpoint(s)"
```

To evaluate regex proxy model(s) using our defaults on a single GPU, execute:

```shell
bash scripts/evaluate_regex_gpu.sh "/glob/to/regex/model/*/checkpoint(s)"
```

</p>
</details>

### Comparisons and visualizations

<details><summary>i. Model pair comparison</summary>
<p>

For comparing neural SoPa++ and regex proxy model pair(s), we use `src/compare_model_pairs.py`:

```
usage: compare_model_pairs.py [-h] --eval-data <file_path> --eval-labels
                              <file_path> --model-log-directory <glob_path>
                              [--atol <float>] [--batch-size <int>]
                              [--disable-tqdm] [--gpu] [--gpu-device <str>]
                              [--logging-level {debug,info,warning,error,critical}]
                              [--max-doc-len <int>] [--output-prefix <str>]
                              [--torch-num-threads <int>]
                              [--tqdm-update-period <int>]

optional arguments:
  -h, --help             show this help message and exit

required evaluation arguments:
  --eval-data            <file_path>
                         Path to evaluation data file (default: None)
  --eval-labels          <file_path>
                         Path to evaluation labels file (default: None)
  --model-log-directory  <glob_path>
                         Glob path to model log directory/directories which
                         contain both the best neural and compressed regex
                         models (default: None)

optional evaluation arguments:
  --atol                 <float>
                         Specify absolute tolerance when comparing
                         equivalences between tensors (default: 1e-06)
  --batch-size           <int>
                         Batch size for evaluation (default: 256)
  --max-doc-len          <int>
                         Maximum document length allowed (default: None)
  --output-prefix        <str>
                         Prefix for output classification report (default:
                         test)

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

To compare neural SoPa++ and regex proxy model pair(s) using our defaults on the CPU, execute:

```shell
bash scripts/compare_model_pairs.sh "/glob/to/model/log/*/director(ies)"
```

To compare neural SoPa++ and regex proxy model pair(s) using our defaults on a GPU, execute:

```shell
bash scripts/compare_model_pairs_gpu.sh "/glob/to/model/log/*/director(ies)"
```

</p>
</details>

<details><summary>ii. FMTOD summary statistics</summary>
<p>

For visualizing the FMTOD data set summary statistics, we apply functions from `src/visualize_fmtod.R`. This workflow is wrapped using `scripts/visualize_fmtod.sh`:

```
Usage: visualize_fmtod.sh [-h|--help]

Visualize FMTOD data set summary statistics

Optional arguments:
  -h, --help  Show this help message and exit
```

To visualize the FMTOD data set summary statistics, simply execute:

```shell
bash scripts/visualize_fmtod.sh
```

</p>
</details>

<details><summary>iii. Grid-based training</summary>
<p>

For visualizing grid-based training performance, we use `src/tensorboard_event2csv.py` to convert tensorboard event logs to `csv` files and apply functions from `src/visualize_grid.R` to plot them. These two scripts are bound together by `scripts/visualize_grid_train.sh`:

```
Usage: visualize_grid_train.sh [-h|--help] tb_event_directory

Visualize grid training performance for SoPa++ models,
given that grid allows for the following varying arguments:
patterns, tau_threshold, seed

Optional arguments:
  -h, --help                      Show this help message and exit

Required arguments:
  tb_event_directory <glob_path>  Tensorboard event log directory/
                                  directories
```

To produce a facet-based visualization of grid-based training, simply execute:

```shell
bash scripts/visualize_grid_train.sh "/glob/to/tb/event/*/director(ies)"
```

**Note:** This script has been hard-coded for grid-based training scenarios where only the following three training/model arguments are varied: `patterns`, `tau_threshold` and `seed`.

</p>
</details>

<details><summary>iv. Grid-based evaluation</summary>
<p>

For visualizing grid-based evaluation performance and model-pair distances, we apply functions from `src/visualize_grid.R`. This workflow is wrapped using `scripts/visualize_grid_evaluate.sh`:

```
Usage: visualize_grid_evaluate.sh [-h|--help] model_log_directory

Visualize grid evaluations for neural SoPa++ and regex
model pairs, given that grid allows for the following varying arguments:
patterns, tau_threshold, seed

Optional arguments:
  -h, --help                       Show this help message and exit

Required arguments:
  model_log_directory <glob_path>  Model log directory/directories
                                   containing neural SoPa++ and regex
                                   models, as well as all evaluation json's
```

To produce a facet-based visualization of grid-based evaluation, simply execute:

```shell
bash scripts/visualize_grid_evaluate.sh "/glob/to/model/log/*/director(ies)"
```

**Note:** This script has been hard-coded for grid-based evaluation scenarios where only the following three training/model arguments are varied: `patterns`, `tau_threshold` and `seed`.

</p>
</details>

<details><summary>v. Regex samples and STE neurons</summary>
<p>

For visualizing regex samples and STE neurons, we use `src/visualize_regex.py`:

```
usage: visualize_regex.py [-h] --class-mapping-config <file_path>
                          --regex-model-checkpoint <glob_path>
                          [--disable-tqdm]
                          [--logging-level {debug,info,warning,error,critical}]
                          [--max-transition-tokens <int>] [--num-regex <int>]
                          [--only-neurons] [--seed <int>]
                          [--tqdm-update-period <int>]

optional arguments:
  -h, --help                show this help message and exit

required visualization arguments:
  --class-mapping-config    <file_path>
                            Path to class mapping configuration (default:
                            None)
  --regex-model-checkpoint  <glob_path>
                            Glob path to regex model checkpoint(s) with '.pt'
                            extension (default: None)

optional visualization arguments:
  --max-transition-tokens   <int>
                            Maximum number of tokens to display per transition
                            (default: 5)
  --num-regex               <int>
                            Number of regex's to produce for each STE neuron
                            (default: 5)
  --only-neurons            Only produces plots of neurons without regex's
                            (default: False)
  --seed                    <int>
                            Random seed for numpy (default: 42)

optional logging arguments:
  --logging-level           {debug,info,warning,error,critical}
                            Set logging level (default: info)

optional progress-bar arguments:
  --disable-tqdm            Disable tqdm progress bars (default: False)
  --tqdm-update-period      <int>
                            Specify after how many training updates should the
                            tqdm progress bar be updated with model
                            diagnostics (default: 5)
```

To visualize activating regex samples with corresponding STE neurons, execute the following:

```shell
bash scripts/visualize_regex_with_neurons.sh "/glob/to/regex/model/*/checkpoint(s)" 
```

To visualize only STE neurons, execute the following:

```shell
bash scripts/visualize_regex_only_neurons.sh "/glob/to/regex/model/*/checkpoint(s)" 
```

</p>
</details>

## Development :snail:

Ongoing development of this repository is documented in this [log](./docs/develop.md).
