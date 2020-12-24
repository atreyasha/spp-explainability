## Table of Contents
-   [Tasks](#tasks)
    -   [Current](#current)
    -   [Long-term](#long-term)
-   [Notes](#notes)
    -   [Research](#research)
    -   [Admin](#admin)
-   [Completed](#completed)
-   [Legacy](#legacy)
    -   [Interpretable RNN
        architectures](#interpretable-rnn-architectures)
    -   [Interpretable surrogate
        extraction](#interpretable-surrogate-extraction)
    -   [Neuro-symbolic paradigms](#neuro-symbolic-paradigms)
    -   [Neural decision trees](#neural-decision-trees)
    -   [Inductive logic on NLP search
        spaces](#inductive-logic-on-nlp-search-spaces)

## Tasks

### Current

1.  **TODO** Major changes to model

    **DEADLINE:** *\<2020-12-24 Thu\>*

    1.  Quick changes

        1.  improve code quality with continue training workflow

            1.  use train arg parser to read/overwrite arguments

            2.  optimizer and scheduler state dictionaries would need to
                passed on via optional argument to train, maybe model
                can be initialized inside train function instead of in
                main

            3.  read epoch information so training/tensorboard-logging
                can be done correctly -\> if epochs are still remaining,
                train until they are finished otherwise train for
                another set of specified epochs

            4.  settle issue of whether to use epoch index or count
                while saving information -\> re-think and make
                implementation consistent most importantly

            5.  test if training can be continued seamlessly with
                information stored in checkpoint

        2.  figure out saving and loading models on both cpu and gpu -\>
            might need to insert map_location argument for specific
            devices -\> can test out with some models saved on s3it gpu
            -\> always save model on cpu: see
            <https://discuss.pytorch.org/t/how-to-get-a-cpu-state-dict/24712>

        3.  consider making implementation of torch device more explicit
            where possible with torch.device and \"to\" function -\>
            would help with cpu/gpu save/load issues seamlessly -\> can
            specify on command line which GPU to use for single case

        4.  test if model works on GPU with new changes -\> do quick
            check without too much detail

        5.  update all `eps` with `epsilons` and `sl` with `self_loops`
            where possible

    2.  Medium-level changes

        1.  address scattered TODOs in code if still remaining OR
            otherwise add them to below tasks

        2.  look through shuffling that happens in training/model_utils
            and whether it is necessary in all places -\> or whether it
            can be compressed in some places such as `model_utils.py`
            functions

        3.  consider changing padding token to dedicated token instead
            of unknown -\> these are not included within soft_pattern
            processing due to construction of Batch class only
            considering length of input sequences

        4.  maybe padding the whole dataset might make more sense than
            doing this repeatedly within each Batch object

            1.  combined model padding probably does not work well
                because of inefficiencies, Batch object helps to keep
                similar document lengths together to ensure most
                computation is used correctly for updating scores and
                not being ignored

        5.  make batch object more efficient, look at existing pytorch
            classes that could help with this -\> probably not the best
            case since the Batch class has a specific use

        6.  might make overall more sense to use max-\* semirings since
            they are easier to interpret -\> try to replicate model from
            defaults of paper instead of code defaults during main runs
            -\> change defaults directly in argument parser -\> add
            scheduler and other perks by default -\> make other defaults
            more sensible for runs such as increasing epochs, learning
            rate etc.

    3.  Core modeling developments

        1.  reduce circum-padding token count to 1 instead of length of
            longest pattern

        2.  add test evaluation workflow for the model with separate
            script -\> this would be useful for the grid-search model
            selection and might need additional argument parser options
            -\> need to set model.eval() before evaluating

            1.  compute test F1 on models at the end of training for
                completeness

        3.  add temperature parameter to encourage more discrete
            learning of pattern scores -\> or binarize patterns via
            `torch.gt` or `torch.relu` if possible which would make
            interpretation much easier -\> look into how other paper on
            RNN-FSA did this with temperature

        4.  modify final layer to a general additive layer; preferably
            with tree structure or soft logic where possible -\>
            otherwise simple linear layer with various basis functions
            would work

            1.  think of ways to use patterns only when there are enough
                words in front and not to always compute, if this is
                possible at all

            2.  make use of torch cuda calls in improved model -\>
                tensor.cuda() returns a copy of the tensor in the GPU
                while module.cuda() sends the model/parameters to the
                GPU -\> not useful for static case since this would
                create memory overhaul

        5.  add thorough and efficient grid-search workflow -\> emulate
            similar workflow from single train -\> perhaps nest single
            train inside grid directory so everything remains modular

        6.  think about how grid search workflow should work and which
            models should be saved/deleted -\> good idea to keep best
            model checkpoints from each grid run and separate event
            files

        7.  think about whether loss or accuracy should be monitored for
            early stopping -\> especially given that losses are
            calculated using forward without evaluation; implying that
            dropout could have stochastic noise impact

        8.  incremental changes with grid-search -\> multiple runs of
            each best model with different random seeds to get standard
            deviation of performance -\> experiment more gracious
            self-loops and epsilon transitions for improved
            generalization

2.  Run SoPa++ for multiple runs to survey performance -\> run on all
    variants and data-set portions with (repeated) grid-search to get
    plenty of candidates, means and standard deviations

    **DEADLINE:** *\<2021-02-01 Mon\>*

3.  With decent model performance, branch off to improve explainability
    with weighting of patterns to address other research questions

    **DEADLINE:** *\<2021-02-01 Mon\>*

    1.  Mimic model

        1.  final ensemble of regular expressions should give insights
            and perform similar to main SoPa++ neural model

        2.  think about how to work with unknown tokens on new data for
            mimic model -\> maybe some mapping of embeddings to find
            closest token/pattern or mean score might help

        3.  best case scenario: user should be able to transfer easily
            between models and regex-ensemble in both directions for
            \"human-computer interaction\"

        4.  for mimic model, find best patterns that match, if not use a
            mean value for the pattern score that can be used as an
            analog -\> or try other heuristics that can bring results of
            mimic and oracle closer to each other

        5.  aim to produce pretty and compact ensemble of regular
            expressions which can analyzed and manipulated by a human

        6.  posted question to OP on self-loops visualization, see:
            <https://github.com/Noahs-ARK/soft_patterns/issues/8#issuecomment-728257052>

    2.  Oracle model

        1.  refactor `soft_patterns_rnn` (if necessary),
            `visualization`, `interpretation` (two of highest priority)
            and `testing` scripts from git backlog to repository

        2.  why are `*START*` and `*END*` tokens repeated before and
            after, and why is `*UNK*` used for padding when a separate
            `*PAD*` token could be used?

            1.  overfitting that occurs to extra `*START*` and `*END*`
                tokens would be transferred to epsilon transitions if
                replaced with single padding instead of multiple

            2.  posted as question to OP, see:
                <https://github.com/Noahs-ARK/soft_patterns/issues/8#issuecomment-746797695>

    3.  Distance between oracle and mimic

        1.  it would still be useful to show when mimic and oracle align
            and when they don\'t -\> with some kind of distance
            measurement between their output scores

        2.  compare confusion matrices between orace and mimic and
            compute euclidean distances on scores or binary predictions

### Long-term

1.  Performance

    1.  tests run in paper show almost perfect accuracy, which could be
        a baseline to match or otherwise come close to, in order to
        probe explainability

    2.  improve learning rate scheduler implementation to more
        soft-coded than hard-coded, if possible at all

    3.  replace all legacy tensor.data calls with tensor.detach() for
        safety

    4.  work on `slurm-s3it` branch as a mirrored branch -\> easier to
        keep workflow simple for `jarvis` only and ignore `s3it`

2.  Visualization

    1.  remember that tensorboard events start at epoch index 0, which
        means after the first epoch of training

    2.  if necessary, the x-axis should be scaled forward by 1 to give
        the correct training epochs

3.  Dynamic and sub-word embeddings (optional)

    1.  use both word and sub-word tokenizers such as nltk or
        sentencepiece tokenizer

        1.  sub-word non-contextual embeddings: fastText or
            <https://nlp.h-its.org/bpemb/#cite>

        2.  word-level non-contextual embeddings: stick to GloVe

    2.  use both static and dynamic token embeddings

        1.  dynamic: start, end and padding tokens should be fixed,
            while unknown and others could be learned

        2.  dynamic: can use a lower learning rate for embeddings to
            reduce overfitting as much as possible

        3.  dynamic: convert embeddings into a tensor and register as
            parameter inside model which gets saved with state
            dictionary object -\> not useful for static case since this
            would create memory overhaul

4.  Argparse, logging and dependencies

    1.  consider whether to pass `logger`, `disable_tqdm` and
        `tqdm_update_freq` variables directly via functions

    2.  use `renv` for managing and shipping R dependencies -\> keep
        just `renv.lock` for easier shipping and ignore other files

    3.  perform sanity check to ensure cross-module imports are not
        affected by presence of `logger`, or otherwise use root logger
        in case it cannot be imported

    4.  **extra:** pass tqdm directly to logger instead of directly to
        stdout: see <https://github.com/tqdm/tqdm/issues/313>

    5.  **brainstorm:** replace input arg namespace with explicit
        arguments, OR possible to make separate argparse Namespace which
        can be passed to main, this could help with portability (needs
        brainstorming since there are tradeoffs between argparse
        Namespace and explicit variable definitions)

5.  Typing and testing

    1.  include test code by instantiating class and/or other simple
        methods which are inherent to the workflow

    2.  ensure that redefined variables are given all possible unioned
        types used inside code

    3.  add mypy as a test case suite, design new and improved test
        cases using pytest after understanding code completely

    4.  consider adding Optional type to all optional arguments

    5.  fine-tune typing in internal functions of
        `SoftPatternClassifier` since some of them require batch-level
        testing to ascertain, eg. `get_transition_matrices`,
        `load_pattern` -\> need to ascertain wither
        `pre_computed_patterns` is List or List\[List\[str\]\] -\>
        consider removing `float` from
        `self_loop_scale: Union[torch.Tensor, float, None]` in
        `transition_once`

    6.  look into cases where List was replaced by Sequential and how
        this can be changed or understood to keep consistency (ie. keep
        everything to List)

6.  Documentation

    1.  improve cryptic parts of code to be more easily readable, such
        as workflow for loading pre-computed patterns inside the soft
        patterns classifier and model checkpointing -\> it can only be
        understood by studying the code whereas it should be more
        structured with clear conditionals

    2.  ensure consistent variable names for variables used in different
        scopes

    3.  ensure consistent variable names for reading/writing such as
        `filename`, `*_file_stream`

    4.  reduce source code chunk newlines to no newlines -\> this makes
        things slightly more concise given the existence of multiple
        comments in between -\> also remove unnecessary comments

    5.  consider changing default helpers in readme to python helpers
        instead of those from shell scripts,

    6.  where applicable, improve documentation of argparse variables
        within argparse script

    7.  update metadata in scripts later with new workflows, eg. with
        help scripts, comments describing functionality and readme
        descriptions for git hooks

    8.  add pydocstrings to all functions for improved documentation -\>
        plus comments where relevant

    9.  provide description of data structures (eg. data, labels)
        required for training processes

    10. make list of all useful commands for slurm -\> useful to re-use
        later on

    11. add MIT license when made public

## Notes

### Research

1.  SoPa++

    1.  extensions

        1.  leverage dynamic sub-word-level embeddings from recent
            advancements in Transformer-based language modeling.

        2.  modify the architecture and hyperparameters to use more
            wildcards or self-loops, and verify the usefulness of these
            in the mimic WFSA models.

        3.  modify the output multi-layer perceptron layer to a general
            additive layer, such as a linear regression layer, with
            various basis functions. This would allow for easier
            interpretation of the importance of patterns without the use
            of occlusion -\> perhaps consider adding soft logic
            functions which could emulate negation/inclusion of rules,
            or possibly a soft decision tree at the top layer

        4.  test SoPa++ on multi-class text classification tasks

2.  SoPa

    1.  goods: practical new architecture which maps to RNN-CNN mix via
        WFSAs, decent code quality in PyTorch (still functional),
        contact made with author and could get advice for possible
        extensions

    2.  limitations

        1.  SoPa utilizes static word-level token embeddings which might
            contribute to less dynamic learning and more overfitting
            towards particular tokens

        2.  SoPa encourages minimal learning of wildcards/self-loops and
            $\epsilon$-transitions, which leads to increased overfitting
            on rare words such as proper nouns

        3.  while SoPa provides an interpretable architecture to learn
            discrete word-level patterns, it is also utilizes occlusion
            to determine the importance of various patterns. Occlusion
            is usually a technique reserved for uninterpretable model
            architectures and contributes little to global
            explainability

        4.  SoPa was only tested empirically on binary text
            classification tasks

    3.  general: likely higher performance due to direct inference and
        less costly conversion methods

3.  Data sets

    1.  NLU data sets -\> single sequence intent classification,
        typically many classes involved -\> eg. ATIS, Snips,
        AskUbuntuCorpus, FB task oriented dataset (mostly intent
        classifications)

    2.  SOTA scores for NLU can be found on
        <https://github.com/nghuyong/rasa-nlu-benchmark#result>

    3.  vary training data sizes from 10% to 70% for perspective on data
        settings

4.  Constraints

    1.  work with RNNs only

    2.  seq2cls tasks -\> eg. NLU/NLI/semantic tasks, try to work with
        simpler single (vs. double) sequence classification task

    3.  base main ideas off peer-reviewed articles

5.  Research questions

    1.  To what extent does SoPa++ contribute to competitive performance
        on NLU tasks?

    2.  To what extent does SoPa++ contribute to improved explainability
        by simplification?

    3.  What interesting and relevant explanations does SoPa++ provide
        on NLU task(s)?

### Admin

1.  Timeline

    1.  ~~Initial thesis document: **15.09.2020**~~

    2.  ~~Topic proposal draft: **06.11.2020**~~

    3.  ~~Topic proposal final: **15.11.2020**~~

    4.  Topic registration: **01.02.2021**

    5.  Manuscript submission: **18.03.2021**

2.  Manuscript notes

    1.  Text-related feedback

        1.  20-90 pages thesis length -\> try to keep ideas
            well-motivated yet succinct

        2.  make abstract more specific in terms of \"highly
            performant\"

        3.  sub-word embeddings are both useful for performance and
            explainability

        4.  fix absolute terms such as \"automated reasoning\", or quote
            directly from paper

        5.  re-consider reference to Transformers for dynamic sub-word
            level word-embeddings

        6.  improve capitalization with braces in bibtex file

    2.  Concept-related feedback

        1.  clarify meaning and concept of \"occlusion\" as
            leave-one-out perturbation analysis

        2.  improve arbitrary vs. contrained oracle phrasing -\> perhaps
            black-box vs. white-box but more specific

        3.  expound on trade-off between performance and explainability
            and process of mimic extraction

        4.  add more information on what competitive performance means
            (eg. within few F_1 points)

        5.  how to evaluate improved explainability -\> make hierarchy
            for local vs. global explainability -\> also explainability
            is only relevant if the oracle and mimic models both
            **perform competitively and have similar confusion matrix
            profiles** (both conditions must be satisfied)

        6.  further work: porting this technique to a transformer where
            possible

    3.  Self-thoughts

        1.  semirings, abstract algebra and how they are used for
            finite-state machines in Forward and Viterbi algorithms -\>
            go deeper into this to get some background

        2.  use more appropriate and generalized semiring terminology
            from Peng et al. 2019 -\> more generalized compared to SoPa
            paper

        3.  Chomsky hierarchy of languages -\> might be relevant
            especially relating to CFGs

        4.  FSA/WFSAs -\> input theoretical CS, mathematics background
            to describe these

        5.  ANN\'s historical literature -\> describe how ANNs
            approximate symbolic representations

        6.  extension/recommendations -\> transducer for seq2seq tasks

## Completed

**DONE** log model metrics with intra/inter-epoch frequency
which can be shared with tqdm for displaying -\> would require some
recoding with modulos -\> how to manage updates with batch vs. epochs
conflict and how to continue training as well, think about whether to
recompute accuracy as well on a batch-basis

**CLOSED:** *\[2020-12-22 Tue 12:22\]*

**DONE** add argparse option of how often to update tqdm
metrics in training -\> should be shared parameter for tensorboard
logging

**CLOSED:** *\[2020-12-22 Tue 12:22\]*

**DONE** make consistent use of `validation` versus `dev`
throughout all source code -\> redo all log messages and also file
naming especially related to inputs, preprocessing and argparse -\> will
require time and effort

**CLOSED:** *\[2020-12-20 Sun 17:49\]*

**DONE** remove `rnn` option from code altogether -\> keep
things simple for now

**CLOSED:** *\[2020-12-19 Sat 02:33\]*

**DONE** change argparse variable names within train script
to reflect parser and make this consistent throughout, including in
other auxiliary scripts

**CLOSED:** *\[2020-12-19 Sat 01:33\]*

**DONE** need to understand `nn.Module` functionality before
anything else -\> investigate whether `fixed_var` function is indeed
necessary or can be removed since `requires_grad` is set to False by
default, but could be some conflict with `nn.Module` default parameter
construction with `requires_grad = True` -\> left intact for now and
appears to work well

**CLOSED:** *\[2020-12-12 Sat 12:28\]*

**DONE** look through `train.py` and make comments on general
processes -\> fix minor issues where present such as variable naming,
formatting etc.

**CLOSED:** *\[2020-12-08 Tue 18:38\]*

**DONE** major code refactoring for main model with
conversion to recent PyTorch (eg. 1.\*) and CUDA versions (eg. 10.\*)

**CLOSED:** *\[2020-12-05 Sat 18:47\]* **DEADLINE:** *\<2020-12-06
Sun\>*

**DONE** add tensorboard to explicit dependencies to view
relevant logs during training

**CLOSED:** *\[2020-12-03 Thu 14:40\]*

**DONE** replace all Variable calls with simple Tensors and
add `requires_grad` argument directly to tensors where this is
necessary: see
<https://stackoverflow.com/questions/57580202/whats-the-purpose-of-torch-autograd-variable>

**CLOSED:** *\[2020-12-02 Wed 21:50\]*

**DONE** UserWarning: Implicit dimension choice for
log_softmax has been deprecated. Change the call to include dim=X as an
argument

**CLOSED:** *\[2020-12-02 Wed 18:57\]*

**DONE** UserWarning: size_average and reduce args will be
deprecated, please use reduction=\'sum\' instead

**CLOSED:** *\[2020-12-02 Wed 18:39\]*

**DONE** make workflow to download Facebook Multilingual Task
Oriented Dataset and pre-process to sopa-ready format -\> text data and
labels with dictionary mapping as to what the labels mean

**CLOSED:** *\[2020-12-01 Tue 20:29\]* **DEADLINE:** *\<2020-12-03
Thu\>*

**DONE** fixed: UserWarning: nn.functional.sigmoid is
deprecated. Use torch.sigmoid instead

**CLOSED:** *\[2020-11-30 Mon 18:16\]*

**DONE** sort CLI arguments into proper groups, sort them
alphabetically for easier reading

**CLOSED:** *\[2020-11-30 Mon 18:07\]*

**DONE** add types to `parser_utils.py` script internals

**CLOSED:** *\[2020-11-30 Mon 18:07\]*

**DONE** separate extras in `soft_patterns.py` into
`utils.py` -\> test out how batch is utilized -\> fix batch issue, then
move on to other steps -\> batch mini-vocab appears to be a hack to
create a meta-vocabulary for indices -\> try to push with this again
another time -\> consider reverting Vocab index/token defaults in case
this was wrong

**CLOSED:** *\[2020-11-30 Mon 18:07\]*

**DONE** appears to be major bug in Batch class, try to
verify if it is indeed a bug and how it can be fixed

**CLOSED:** *\[2020-11-30 Mon 18:07\]*

**DONE** extract all arg parser chunks and place in dedicated
file

**CLOSED:** *\[2020-11-30 Mon 18:07\]*

**DONE** clean preprocessing script for GloVe vectors and
understand inner mechanisms

**CLOSED:** *\[2020-11-28 Sat 17:02\]*

**DONE** find better location to place code from `util.py`

**CLOSED:** *\[2020-11-27 Fri 19:38\]*

**DONE** migrate to soft-patterns-pp and clean from there

**CLOSED:** *\[2020-11-26 Thu 20:11\]*

**DONE** update proposal with comments from supervisors -\>
update same information here

**CLOSED:** *\[2020-11-17 Tue 14:52\]* **DEADLINE:** *\<2020-11-17
Tue\>*

**DONE** write proposal with key research questions -\>
address points directly from step 3 document requirements -\> prepare
some basic accuracy metrics and interpretations from best model

**CLOSED:** *\[2020-11-10 Tue 18:45\]* **DEADLINE:** *\<2020-11-06
Fri\>*

**DONE** analyze pattern log more closely with code on the
side to understand what it means -\> can start writing early when things
start to make sense

**CLOSED:** *\[2020-11-10 Tue 18:44\]* **DEADLINE:** *\<2020-11-05
Thu\>*

**DONE** add large amounts of binary data for testing with
CPU/GPU -\> requires pre-processing

**CLOSED:** *\[2020-11-10 Tue 18:21\]*

**DONE** find re-usable code for running grid search -\>
otherwise construct makeshift quick code

**CLOSED:** *\[2020-11-05 Thu 20:38\]*

**DONE** test SoPa on sample data in repository to ensure it
works out-of-the-box -\> try this on laptop and s3it

**CLOSED:** *\[2020-11-02 Mon 16:40\]*

**DONE** make workflow to reproduce virtual environment
cleanly via poetry

**CLOSED:** *\[2020-11-02 Mon 16:34\]*

**DONE** make workflow to download simple but high-quality
NLU dataset and glove data sets

**CLOSED:** *\[2020-11-01 Sun 20:15\]* **DEADLINE:** *\<2020-11-01
Sun\>*

**DONE** read more into these tasks and find one that has
potential for interpretability -\> likely reduce task to binary case for
easier processing (eg. entailment)

**CLOSED:** *\[2020-10-28 Wed 15:32\]* **DEADLINE:** *\<2020-10-28
Wed\>*

**DONE** search for popular NLI datasets which have existing
RNN models as (almost) SOTAs, possibly use ones that were already tested
for eg. RTC or ones used in papers that may have semantic element

**CLOSED:** *\[2020-10-26 Mon 17:57\]* **DEADLINE:** *\<2020-10-28
Wed\>*

**DONE** explore below frameworks (by preference) and find
most feasible one

**CLOSED:** *\[2020-10-26 Mon 14:28\]* **DEADLINE:** *\<2020-10-26
Mon\>*

**DONE** add org-mode hook to remove startup visibility
headers in org-mode to markdown conversion

**CLOSED:** *\[2020-10-22 Thu 13:28\]*

**DONE** Set up repo, manuscript and develop log

**CLOSED:** *\[2020-10-22 Thu 12:36\]*

## Legacy

### Interpretable RNN architectures

1.  State-regularized-RNNs (SR-RNNs)

    1.  good: very powerful and easily interpretable architecture with
        extensions to NLP and CV

    2.  good: simple code which can probably be ported to PyTorch
        relatively quickly

    3.  good: contact made with author and could get advice for possible
        extensions

    4.  problematic: code is outdated and written in Theano, TensorFlow
        version likely to be out by end of year

    5.  problematic: DFA extraction from SR-RNNs is clear, but DPDA
        extraction/visualization from SR-LSTMs is not clear probably
        because of no analog for discrete stack symbols from continuous
        cell (memory) states

    6.  possible extensions: port state-regularized RNNs to PyTorch
        (might be simple since code-base is generally simple), final
        conversion to REs for interpretability, global explainability
        for natural language, adding different loss to ensure words
        cluster to same centroid as much as possible -\> or construct
        large automata, perhaps pursue sentiment analysis from SR-RNNs
        perspective instead and derive DFAs to model these

2.  Rational recurences (RRNNs)

    1.  good: code quality in PyTorch, succinct and short

    2.  good: heavy mathematical background which could lend to more
        interesting mathematical analyses

    3.  problematic: seemingly missing interpretability section in paper
        -\> theoretical and mathematical, which is good for
        understanding

    4.  problematic: hard to draw exact connection to interpretability,
        might take too long to understand everything

3.  Finite-automation-RNNs (FA-RNNs)

    1.  source code likely released by November, but still requires
        initial REs which may not be present -\> might not be the best
        fit

    2.  FA-RNNs involving REs and substitutions could be useful
        extensions as finite state transducers for interpretable neural
        machine translation

### Interpretable surrogate extraction

1.  overall more costly and less chance of high performance

2.  FSA/WFSA extraction

    1.  spectral learning, clustering

    2.  less direct interpretability

    3.  more proof of performance needed -\> need to show it is better
        than simple data learning

### Neuro-symbolic paradigms

1.  research questions

    1.  can we train use a neuro-symbolic paradigm to attain high
        performance (similar to NNs) for NLP task(s)?

    2.  if so, can this paradigm provide us with greater explainability
        about the inner workings of the model?

### Neural decision trees

1.  decision trees are the same as logic programs -\> the objective
    should be to learn logic programs

2.  hierarchies are constructed in weight-space which lends itself to
    non-sequential models very well -\> but problematic for token-level
    hierarchies

3.  research questions

    1.  can we achieve similar high performance using decision tree
        distillation techniques (by imitating NNs)?

    2.  can this decision tree improve interpretability/explainability?

    3.  can this decision tree distillation technique outperform simple
        decision tree learning from training data?

### Inductive logic on NLP search spaces

1.  can potentially use existing IM models such as paraphrase detector
    for introspection purposes in thesis

2.  n-gram power sets to explore for statistical artefacts -\> ANNs can
    only access the search space of N-gram power sets -\> solution to
    NLP tasks must be a statistical solution within the power sets which
    links back to symbolism

3.  eg. differentiable ILP from DeepMind

4.  propositional logic only contains atoms while predicate/first-order
    logic contain variables
