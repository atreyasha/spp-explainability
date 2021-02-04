## Table of Contents
-   [Tasks](#tasks)
    -   [Current](#current)
    -   [Long-term](#long-term)
-   [Notes](#notes)
    -   [Research](#research)
    -   [Administrative](#administrative)
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

1.  Dedicated explainability

    1.  Generalizing global explainability

        1.  make separate script for regex compression framework, such
            as `compress_explainable_spp`

            1.  develop merging framework where regular expressions are
                generalized from all the best patterns available

            2.  provide options such as rational, brave and mad for
                compression types, indicating how broad they could be

            3.  look into similar regular expressions across patterns
                and if this can be optimized somehow

            4.  produce pretty and compact ensemble of regular
                expressions

            5.  perform some kind of clustering to determine which regex
                patterns to show/save and which ones not to

            6.  compile regular expressions when doing a search over the
                mimic model

            7.  modify `get_nearest_neighbors` to use full embeddings
                and to use biases as well

            8.  keep the existing nearest neighbours functionality to
                deal with similar patterns with different tokens

            9.  add back `k_best` to argument parser in case it is
                needed

            10. think about how to work with unknown tokens on new data
                for mimic model

            11. find tricks which help to increase generalization such
                as defaulting to patterns given nearest neighbours or
                levenstein distance -\> or at least discuss them

        2.  since all transitions are only dependent on the state and
            not on each other, would it be possible to concatenate all
            token spans for each pattern and interweave between them?

            1.  if this is possible, it will hugely benefit global
                explainability

            2.  might not be the best solution, but we can experiment
                how this affects generalization

            3.  perhaps use graphviz for visualization

    2.  Quantification and generalizing global explainability

        1.  compare confusion matrices between oracle and mimic and
            compute euclidean distances over raw softmax predictions

        2.  demonstrate when global explainability aligns well and when
            it does not

        3.  can be done for both the train and test partitions to check
            for extrapolation potential for explainability

    3.  Local explainability as a failsafe

        1.  rename `explain_spp` to something related to globalk
            explainability and mimic model construction, since another
            script will be needed to conduct local explanations

        2.  consider adding softmax to model forward if this is of any
            use

        3.  in cases where global explainability cannot be deciphered,
            we can provide a failsafe of local explainability

        4.  this can be done a per-sample basis with pattern and score
            specification -\> more likely to be useful on the test set

    4.  Generic changes

        1.  re-check token indices in backpointers and ensure they are
            indeed correct -\> do some manual runs and re-thinking

        2.  think whether to allow transition out of end state and how
            this works, or what this means and if it is logical

        3.  look again into document indices and why legacy code
            subtracted padding tokens

        4.  look into legacy visualization and interpretation scripts
            for possible workflows to adopt

        5.  clean out source code with newer and more efficient
            workflows, consistent variable namings and function
            definitions on-the-fly

        6.  precisely type functions and classes on-the-fly -\>
            especially for explainability scripts

        7.  sort out scattered TODOs

2.  Post explainability

    1.  Extension to new data sets

        1.  consider extending workflow to ATIS and/or SNIPS

        2.  re-use preprocessing functions by sending them to utils and
            perhaps make them more general where possible

        3.  both have some papers which could be cited to add some
            relevance

    2.  Explainability execution speed

        1.  add batch size to explainability while computing
            `interim_scores`, but would require that `explain_data` and
            `explain_text` are sorted beforehand in a symmetric manner
            and respecting the fact that sorting requires label to be
            replaced with text, also need to add `batch_size` to
            `arg_parser`

        2.  if applicable, merge efficiencies from
            `visualize_efficiently` such as using `heapq`

    3.  Dedicated modelling

        1.  think again about removing binarizer if it limits freedom of
            model too much

        2.  add option to increase threshold for STE to detect more
            distinct and sparse patterns instead of roughly half -\>
            call this threshold \"tau\"

        3.  attempt to make normalizer dynamically ignore infinities
            instead of expecting fixed sizes -\> could be done with a
            simple normalization routine but would have to work on
            vectoring code

        4.  consider adding back elementwise affine transformations for
            LayerNorm -\> but this could possibly result in dead
            patterns to be activated which is an illogical result -\>
            unless we can guarantee that infinity states will always be
            ignored

        5.  consider using a generic function for batch minima, since
            this could be dependent on the semiring

        6.  encourage learning of wildcards by increasing its scale
            factor default -\> in case there are not enough

        7.  add 2 threads specific arguments to all jarvis shell scripts
            and commit as local optimizations

        8.  change frequency of tensorboard, evaluation and model saving
            to update-level

            1.  update arg parser with new arguments

            2.  update train_inner with new workflow for saving/loading
                information

            3.  update save checkpoint code with required information

            4.  shift torch hooks around to be more sensible

            5.  convert data object to generator

        9.  **extra:** repeat grid-search with multiple random seeds -\>
            do this after all changes

        10. **extra:** use parallelized computations to fill up all GPU
            memory -\> would require reading-up on how to do this safely
            for a single GPU

### Long-term

1.  Performance and explainability

    1.  add check to ensure start, end and pad tokens don\'t occur
        adversarially inside sequence -\> need to have the vocabulary
        object catch such an error

    2.  add predict function for both mimic and oracle model which does
        not need extra data to be loaded -\> can also accept stdin as
        unix pipe

    3.  when comparing model performance with other studies, consider
        only lowercasing and not making unique the test set

    4.  check if packed sequences could be incoporated into model

2.  Re-check potential pitfalls

    1.  add `with torch.no_grad()` scope indicator alongside
        `model.eval()` to perform inference/validation correctly and
        efficiently

    2.  replace all legacy `tensor.data` calls with `tensor.detach()`
        for safety and `tensor.detach().clone()` for cases where data is
        being updated

    3.  check code for `squeeze()` call which can be problematic for dim
        1 tensors

3.  Dependencies, typing and testing

    1.  include basic test code by instantiating class and/or other
        simple methods

    2.  add mypy as an explicit part of testing the source code

    3.  replace Union + None types with Optional type for conciseness

    4.  replace all `dict` types with `Dict` for consistency

    5.  look into cases where List was replaced by Sequential and how
        this can be changed or understood to keep consistency (ie. keep
        everything to List)

4.  Documentation and clean-code

    1.  rename `spp_model` to `spp_torch_model` to fit indents better

    2.  port backpointer concept to model as another function inside
        torch model

    3.  remove cases where variables from argument namespace are
        redefined as local variables, a common example of this is with
        `args.model_log_directory` and `model_log_directory`

    4.  read paper again to get some familiarity with terms and
        algorithms

    5.  find better naming for mimic/oracle models which is based on
        research terminology

    6.  GPU/CPU runs not always reproducible depending on
        multi-threading, see:
        <https://pytorch.org/docs/stable/notes/randomness.html#reproducibility>

    7.  consider renaming `soft_patterns_pp` to more elegant name
        without special symbols such as `spp` or better -\> be useful to
        think of this before registering topic

    8.  reduce source code lines, chunking and comments -\> pretty sort
        python code and function/class orders perhaps by length

    9.  add a comment to each code chunk which explains inner mechanisms
        better

    10. update metadata eg. with comprehensive python/shell help
        scripts, comments describing functionality and readme
        descriptions for git hooks

    11. add information on best model downloads and preparation

    12. add pydocstrings to all functions and improve argparse
        documentation

    13. provide description of data structures (eg. data, labels)
        required for training processes

    14. test download and all other scripts to ensure they work

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

### Administrative

1.  Timeline

    1.  ~~Initial thesis document: **15.09.2020**~~

    2.  ~~Topic proposal draft: **06.11.2020**~~

    3.  ~~Topic proposal final: **15.11.2020**~~

    4.  ~~Topic registration: **01.02.2021**~~

    5.  Manuscript submission: **31.03.2021**

2.  Manuscript notes

    1.  Text-related feedback

        1.  20-90 pages thesis length -\> try to keep ideas
            well-motivated yet succinct

        2.  make abstract more specific in terms of \"highly
            performant\"

        3.  fix absolute terms such as \"automated reasoning\", or quote
            directly from paper

        4.  re-consider reference to Transformers for dynamic sub-word
            level word-embeddings

        5.  improve capitalization with braces in bibtex file

    2.  Concept-related feedback

        1.  clarify meaning and concept of \"occlusion\" as
            leave-one-out perturbation analysis

        2.  cite and explain straight-through estimation (STE) with
            Heaviside variant

        3.  improve arbitrary vs. contrained oracle phrasing -\> perhaps
            black-box vs. white-box but more specific

        4.  expound on trade-off between performance and explainability
            and process of mimic extraction

        5.  add more information on what competitive performance means
            (eg. within few F_1 points)

        6.  how to evaluate improved explainability -\> make hierarchy
            for local vs. global explainability -\> also explainability
            is only relevant if the oracle and mimic models both
            **perform competitively and have similar confusion matrix
            profiles** (both conditions must be satisfied)

        7.  how does binarizing help with explainability?

        8.  how does this new framework improve explainability over the
            previous baseline? explain either via hierarchies, metrics
            or tangible task-based insights

        9.  how can a user make use of the mimic model and what benefits
            are there for the user in terms of security/safety/etc?

        10. think more about points to include or mention weakly instead
            of strongly in paper such as something is better or worse

        11. further work: porting this technique to a transformer where
            possible

    3.  Visualizations

        1.  add visualization of in-depth computational graph in paper
            for clarity -\> this can be automated

        2.  use graphical TikZ editor for creating graphs -\> produce
            pretty graph to show processing

        3.  produce visualization of training performance using python
            frameworks rather than R

    4.  Self-thoughts

        1.  compare oracle performance with those from other papers

        2.  semirings, abstract algebra and how they are used for
            finite-state machines in Forward and Viterbi algorithms -\>
            go deeper into this to get some background

        3.  use more appropriate and generalized semiring terminology
            from Peng et al. 2019 -\> more generalized compared to SoPa
            paper

        4.  Chomsky hierarchy of languages -\> might be relevant
            especially relating to CFGs

        5.  FSA/WFSAs -\> input theoretical CS, mathematics background
            to describe these

        6.  ANN\'s historical literature -\> describe how ANNs
            approximate symbolic representations

        7.  extension/recommendations -\> transducer for seq2seq tasks

## Completed

**DONE** modify normalizer to ignore calculation of all
infinities via minimal value replacement

**CLOSED:** *\[2021-01-27 Wed 19:19\]*

**DONE** remove both epsilon/self-loops -\> use only simple
transitions and hard wild cards

**CLOSED:** *\[2021-01-27 Wed 15:01\]*

**DONE** defaults from paper: semiring -\> max-product,
batch-size -\> 128 (cpu), epochs -\> 200, patience -\> 30, word_dim -\>
300

**CLOSED:** *\[2021-01-02 Sat 14:23\]*

**DONE** reduce circum-padding token count to 1 instead of
length of longest pattern

**CLOSED:** *\[2020-12-31 Thu 13:03\]*

**DONE** test out to see if scheduler works and if its state
gets incremented -\> need to train single model for long period of time
and analyze state_dict of scheduler to see what has been recorded -\> it
works well when clip threshold is set to zero and patience is observed

**CLOSED:** *\[2020-12-31 Thu 13:01\]*

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
