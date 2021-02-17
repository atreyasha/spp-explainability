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

1.  Tasks pre-writing

    **DEADLINE:** *\<2021-02-21 Sun\>*

    1.  Dedicated modelling

        1.  **TODO** re-make usage scripts for github readme

        2.  **TODO** make custom normalizer module to
            dynamically ignore infinities instead of expecting fixed
            sizes -\> perhaps create a new torch Module for this

        3.  **TODO** consider using a generic function for
            batch minima, since this could be dependent on the semiring

        4.  change frequency of tensorboard, evaluation and model saving
            to update-level

            1.  update arg parser with new arguments

            2.  update train_inner with new workflow for saving/loading
                information

            3.  update save checkpoint code with required information

            4.  shift torch evaluation/autograd hooks around to be more
                sensible and to respect more frequent evaluations

            5.  vary tau argument in grid search and use values in range
                from 0.00-0.75 -\> use a greater than zero tau value to
                depend on fewer patterns

            6.  repeat grid-search with multiple random seeds -\> do
                this after all changes and run code multiple times to
                maximize GPU memory usage

2.  Tasks intra-writing

    1.  Visualization and summary-statistics

        1.  Cross-model comparisons

            1.  visualize examples where regex and neural model align
                and misalign, eg. with a confusion matrix

            2.  compute statistics over inter-model comparisons such as
                average distance, misalignment, activation frequency and
                other useful metrics that can elucidate on-the-ground
                processes

        2.  Regex OOP

            1.  add visualization of regex ensemble with graphviz -\>
                can specify which pattern to visualize and how to make
                large scale visualizations

            2.  visualize patterns as dots with internal pie charts
                which show magnitudes of each class; displayed as a
                colour with legend, will show relative importance of
                each binary neuron and can help us segment their
                purposes

        3.  Model runs

            1.  add visualizations of model runs using dedicated
                functions, preferably all using python libraries

        4.  SoPa++ computational graph

            1.  add visualization of computational graph function
                directly to sopa torch model

    2.  Model diversity

        1.  run above explainability and comparison workflow on a larger
            model as well, just to check extrapolation to different
            model sizes

    3.  Speed of explainability execution

        1.  attempt to multi-thread all regex-related scripts

        2.  find out if individual compiled regex or full
            bracket-compiled regex is better

### Long-term

1.  Explore activation generalization methods

    1.  improve baseline simplification and rational compression method

        1.  handle **UNK** tokens on new data for either in regex OOP or
            during simplification/compression -\> perhaps look for best
            possible match given context -\> **might be well-enough
            handled by wildcards**

        2.  EITHER needs more features from simplification such as
            nearest neighbours OR generate them with access to the model
            again -\> use comparison scripts to determine which
            improvements are necessary -\> this should go into the
            SoPa++ neural model below trace functions -\> look into
            legacy code for some hints -\> \*might be well enough
            handled by looking into enough training samples

    2.  think of taking tokens in a regex group and finding their
        **K-nearest-neighbours** in transition space to expand on them
        if possible -\> only do this if there are few samples and if
        their neighbours have very close scores (within eps), see:
        <https://discuss.pytorch.org/t/k-nearest-neighbor-in-pytorch/59695/2>

        1.  would require extra neural class function to compute all
            transition matrices

        2.  hard to justify these as compression techniques, more closer
            to simplificiation -\> but perhaps this is just a
            technicality which can be addressed later on

        3.  might not help too much since regex model appears
            over-activated at the binary layer compared to the neural
            model -\> these compression generalizations will just
            increase activations; where we would rather expect sparsity
            instead

    3.  think of semantic clustering with digits or time or other means
        -\> if there are no wildcards present -\> would require external
        ontology such as WordNet -\> would be slightly more work
        intensive and is perhaps better to leave this for further work

2.  Performance and explainability

    1.  rename `explain_spp` to something related to global
        explainability and mimic model construction, since another
        script will be needed to conduct local explanations

    2.  add check to ensure start, end and pad tokens don\'t occur
        adversarially inside sequence -\> `nltk.word_tokenize` already
        breaks these up

    3.  add predict function for both mimic and oracle model which does
        not need extra data to be loaded -\> can also accept stdin as
        unix pipe

    4.  when comparing model performance with other studies, consider
        only lowercasing and not making unique the test set

    5.  check if packed sequences could be incoporated into model

    6.  ensure final published model has all new model parameters such
        as `tau_threshold` and `bias_scale` specified

3.  Re-check potential pitfalls

    1.  add `with torch.no_grad()` scope indicator alongside
        `model.eval()` to perform inference/validation correctly and
        efficiently

    2.  replace all legacy `tensor.data` calls with `tensor.detach()`
        for safety and `tensor.detach().clone()` for cases where data is
        being updated

    3.  check code for `squeeze()` call which can be problematic for dim
        1 tensors

4.  Dependencies, typing and testing

    1.  precisely type functions and classes on-the-fly -\> especially
        for explainability scripts

    2.  include basic test code by instantiating class and/or other
        simple methods

    3.  add mypy as an explicit part of testing the source code

    4.  replace Union + None types with Optional type for conciseness

    5.  look into cases where List was replaced by Sequential and how
        this can be changed or understood to keep consistency (ie. keep
        everything to List with overloads)

5.  Documentation and clean-code

    1.  look again into argument parser which have `None` type defaults
        -\> they should be justified to be exceptional cases such as
        dynamic constants

    2.  fix up filenames later on to keep things consistent, eg. `spp`
        vs. `regex_spp` vs. `spp_regex` etc.

    3.  clean out source code with newer and more efficient workflows,
        consistent variable namings and function definitions on-the-fly

    4.  add different usages for different types of models

    5.  settle argument parser examples where some defaults are `None`,
        perhaps we could use another way to specify what default values
        will be chosen, or perhaps only leave arguments to `None` when
        there is no other semantic choice

    6.  remove cases where variables from argument namespace are
        redefined as local variables, a common example of this is with
        `args.model_log_directory` and `model_log_directory`

    7.  find better naming for mimic/oracle models which is based on
        research terminology -\> right now mix of neural and regex is
        being used; it would be good to have something more firm

    8.  GPU/CPU runs not always reproducible depending on
        multi-threading, see:
        <https://pytorch.org/docs/stable/notes/randomness.html#reproducibility>

    9.  reduce source code lines, chunking and comments -\> pretty sort
        python code and function/class orders perhaps by length

    10. add a comment to each code chunk which explains inner mechanisms
        better

    11. update metadata eg. with comprehensive python/shell help
        scripts, comments describing functionality and readme
        descriptions for git hooks

    12. add information on best model downloads and preparation -\> add
        these to Google Drive later on

    13. add pydocstrings to all functions and improve argparse
        documentation

    14. provide description of data structures (eg. data, labels)
        required for training processes

    15. test download and all other scripts to ensure they work

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

4.  Extension to new data sets

    1.  could extend workflow to ATIS and/or SNIPS since all other code
        is established

5.  Constraints

    1.  work with RNNs only

    2.  seq2cls tasks -\> eg. NLU/NLI/semantic tasks, try to work with
        simpler single (vs. double) sequence classification task

    3.  base main ideas off peer-reviewed articles

6.  Research questions

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

    1.  Paper length

        1.  20-90 pages thesis length -\> try to keep ideas
            well-motivated yet succinct

    2.  Feedback-based

        1.  General

            1.  make abstract more specific in terms of \"highly
                performant\"

            2.  fix absolute terms such as \"automated reasoning\", or
                quote directly from paper

            3.  re-consider reference to Transformers for dynamic
                sub-word level word-embeddings

            4.  improve capitalization with braces in bibtex file

            5.  clarify meaning and concept of \"occlusion\" as
                leave-one-out perturbation analysis

            6.  improve arbitrary vs. contrained oracle phrasing -\>
                perhaps black-box vs. white-box but more specific

            7.  add more information on what competitive performance
                means (eg. within few F_1 points)

        2.  Visualizations

            1.  add visualization of in-depth computational graph in
                paper for clarity -\> this can be automated

            2.  use graphical TikZ editor for creating graphs -\>
                produce pretty graph to show processing

            3.  produce visualization of training performance using
                python frameworks rather than R

            4.  produce visualizations of regex ensembles which would be
                interesting, and also pattern activations

        3.  Quantization/Binarization

            1.  change STE citation to 2013 paper which is more
                fundamental, use different name depending on context:
                <https://arxiv.org/abs/1308.3432>

            2.  cite and explain straight-through estimation (STE), and
                what benefits this is supposed to bring

            3.  how does binarizing help with explainability, justify
                requirement for it in both training and testing

        4.  Explainability

            1.  qualify what does it mean to be explainable and how to
                define this

            2.  expound on trade-off between performance and
                explainability and process of mimic extraction

            3.  mention that explainability focuses on exposing the
                model\'s logic and not on necessarily creating
                rationality in the model

            4.  how can a user make use of the mimic model and what
                benefits are there for the user in terms of
                security/safety/etc?

            5.  look at correctly classified samples and see if
                explanations make sense in general, also can do the same
                for wrongly classified examples

            6.  make claim that SoPa++ explainability is different from
                that of vanilla SoPa, but don\'t necessarily say it is
                better

                1.  make hierarchy for local vs. global explainability
                    -\> can provide hints of differences here

                2.  explainability is only relevant if the oracle and
                    mimic models both **perform competitively and have
                    similar confusion matrix profiles**

                3.  provide some metrics or tangible task-based insights
                    to show how new explainability works

        5.  Further work

            1.  mention about how it is not easy to evaluate the
                \"quality of explainability\" and to say one
                explainability is better than the other -\> aside from a
                theoretical perspective

            2.  perhaps suggest how this explainability could be
                evaluated via conducting a survey and getting rating
                from people

            3.  porting this technique to a transformer where possible
                -\> but mention limitations of everything being context
                dependent

    3.  Own-thoughts

        1.  run explainability and evaluation comparisons on models
            trained with different and increasing tau values to
            experiment how this affects performance/explainability -\>
            hypothesis is that this might bring regex and neural
            explainabilities closer together -\> if not then think of
            issues with this process

        2.  think about why larger regex model tends to not be as close
            to neural as a smaller regex model -\> can also be
            investigated with different models

        3.  read paper again to get some familiarity with terms and
            algorithms

        4.  database with indexing could help improve regex lookup speed
            -\> further work

        5.  provide evidence for why different forms of compression
            improve explainable model performance

        6.  can map linear to decision tree as extra work

        7.  make comparison of single-threaded sequential speeds of both
            model over test set

        8.  show possibilities of fixing errors on the test set with
            general changes to the regex model which are much easier to
            do compared to the tensor model

        9.  show cases where we could avoid adversarial cases using the
            insight of the regex model

        10. would be interesting to deterministically export which
            patterns for sure lead to which class, could help to
            identify adversarial samples via tinkering

        11. add information on memory compression resulting from regex
            compression methods

        12. compare oracle performance with those from other papers

        13. semirings, abstract algebra and how they are used for
            finite-state machines in Forward and Viterbi algorithms -\>
            go deeper into this to get some background

        14. use more appropriate and generalized semiring terminology
            from Peng et al. 2019 -\> more generalized compared to SoPa
            paper

        15. Chomsky hierarchy of languages -\> might be relevant
            especially relating to CFGs

        16. FSA/WFSAs -\> input theoretical CS, mathematics background
            to describe these

        17. ANN\'s historical literature -\> describe how ANNs
            approximate symbolic representations

        18. extension/recommendations -\> transducer for seq2seq tasks

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
