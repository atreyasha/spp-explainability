## Table of Contents
-   [Tasks](#tasks)
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

Tasks
-----

### Research

1.  Clean code and documentation

    1.  Current

        1.  **TODO** major code refactoring for main model
            with conversion to recent PyTorch (eg. 1.\*) and CUDA
            versions (eg. 10.\*)

            **DEADLINE:** *\<2020-12-03 Thu\>*

            1.  add typing to remaining core python scripts -\> use this
                to gain an understanding

        2.  work on major changes to architecture

            **DEADLINE:** *\<2020-12-24 Thu\>*

            1.  execute code on low batch level with preset arg
                namespace in `soft_patterns.py` and `train.py` -\> use
                this to understand batch level processes and limitations
                -\> correct typing where necessary -\> confirm whether
                `Adam` falls under `torch.optim.Optimizer` when running
                code

            2.  initial issues

                1.  need to understand `nn.Module` functionality before
                    anything else -\> investigate whether `fixed_var`
                    function is indeed necessary or can be removed since
                    `requires_grad` is set to False by default, but
                    coule be some conflict with `nn.Module` default
                    parameter construction with `requires_grad = True`
                    -\> left intact for now and appears to work well

                2.  look into replacing `loss.data` with
                    `loss.detach()`: see
                    <https://pytorch.org/blog/pytorch-0_4_0-migration-guide/#what-about-data>

                3.  model appears to work out-of-box for multiple
                    classes -\> might not be reliable because it might
                    only focus on one class

                4.  look into iteration progress snippet with gold and
                    predicted \"1\'s\" meaning

                5.  replace input arg namespace with explicit arguments

                6.  rename unsemantic functions such as `read_docs` to
                    `read_doc(ument)`

                7.  look into ISO hard encoding when reading files -\>
                    perhaps this can be modified

                8.  replace current status bars with tqdm and
                    torch-infused bar which could help to understand
                    more parameters

                9.  make separate script to pre-processing raw data into
                    readable format for sopa++ -\> consider renaming
                    `preprocess_glove.py` to more appropriate name later
                    -\> add nltk where needed for pre-processing

                10. change argument names later on as this might break
                    things in actual script for testing

                11. improve code quality with unique model logging and
                    tensorboard workflows

            3.  core model changes

                1.  use separate tokenizers such as nltk or
                    sentencepiece tokenizer from Transformers library

                2.  why are `*START*` and `*END*` tokens repeated before
                    and after, and why is `*UNK*` used for padding when
                    a separate `*PAD*` token could be used?

                    1.  review and opine whether this needs to be
                        changed -\> perhaps there is a reason but this
                        needs to be taken into account in the
                        interpretation/visualization phase

                3.  dynamic OR static word/sub-word embeddings

                    1.  unknown, start and end GloVe vector should be
                        learned, not set to zero

                    2.  OR perhaps there is merit in keeping input
                        vectors fixed to prevent overfitting in small
                        subset of them

                    3.  OR perhaps modify such that the `*UNK*` token
                        can be used for learning over the dataset, while
                        `*START*`, `*END*` and `*PAD*` tokens stay as
                        zeroes to complement overall model

                4.  experiment more gracious self-loops and epsilon
                    transitions for improved generalization

                5.  modify final layer to a general additive layer with
                    tree structure or soft logic where possible -\>
                    perform this incrementally for comparison

            4.  core explainability changes

                1.  revert/refactor soft_patterns_rnn, visualization,
                    interpretation and testing scripts from git backlog
                    to repository -\> understand and improve these
                    significantly

            5.  fine-tuning

                1.  fix up python logging verbosity and pass tqdm
                    directly to logger instead of purely to stdout -\>
                    integrate this well with debug argument as per
                    legacy sopa: see
                    <https://github.com/tqdm/tqdm/issues/313>

                2.  possible to make separate argparse Namespace which
                    can be passed to main, this could help with
                    portability

                3.  use `renv` for managing and shipping R dependencies
                    -\> keep just `renv.lock` for easier shipping and
                    ignore other files

                4.  design new and improved test cases using pytest
                    after understanding code completely

                5.  add proper type checking later to flymake, use data
                    processor class from torch later on

                6.  extend workflow to other RASA NLU data sets given
                    time and resources -\> would require new
                    pre-processing scripts

                7.  provide only description of data structures (eg.
                    data, labels) required for training processes

                8.  consider changing default helpers in readme to
                    python helpers instead of those from shell scripts

        3.  run SoPa++ for multiple runs to survey performance -\> run
            on all variants and data-set portions with grid-search to
            get plenty of candidates

            **DEADLINE:** *\<2021-02-01 Mon\>*

        4.  with decent model performance, branch off to improve
            explainability with weighting of patterns -\> do this and
            the previous task simultaneously

            **DEADLINE:** *\<2021-02-01 Mon\>*

    2.  Long-term

        1.  update metadata in scripts later with new workflows, eg.
            with help scripts, comments describing functionality and
            readme descriptions for git hooks

        2.  add pydocstrings to all functions for improved documentation

        3.  add MIT license when made public

        4.  make list of all useful commands for slurm

2.  SoPa++

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

3.  SoPa

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

    3.  issues

        1.  unsure what self-loops and fwd-1s mean in output of
            `visualize.py` -\> GitHub issue made to request for more
            information:
            <https://github.com/Noahs-ARK/soft_patterns/issues/8>

    4.  general: likely higher performance due to direct inference and
        less costly conversion methods

4.  Data sets

    1.  NLU data sets -\> single sequence intent classification,
        typically many classes involved -\> eg. ATIS, Snips,
        AskUbuntuCorpus, FB task oriented dataset (mostly intent
        classifications)

    2.  SOTA scores for NLU can be found on
        <https://github.com/nghuyong/rasa-nlu-benchmark#result>

    3.  vary training data sizes from 10% to 70% for perspective on data
        settings

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

### Admin

1.  Timeline

    1.  ~~Initial thesis document: **15.09.2020**~~

    2.  ~~Topic proposal draft: **06.11.2020**~~

    3.  ~~Topic proposal final: **15.11.2020**~~

    4.  Topic registration: **01.02.2021**

    5.  Manuscript submission: **18.03.2021**

2.  Manuscript notes

    1.  Text-related feedback

        1.  make abstract more specific in terms of \"highly
            performant\"

        2.  fix absolute terms such as \"automated reasoning\", or quote
            directly from paper

        3.  re-consider reference to Transformers for dynamic sub-word
            level word-embeddings

        4.  improve capitalization with braces in bibtex file

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

    3.  Self-thoughts

        1.  semirings, abstract algebra and how they are used for
            finite-state machines in Forward and Viterbi algorithms -\>
            go deeper into this to get some background

        2.  Chomsky hierarchy of languages -\> might be relevant
            especially relating to CFGs

        3.  FSA/WFSAs -\> input theoretical CS, mathematics background
            to describe these

        4.  ANN\'s historical literature -\> describe how ANNs
            approximate symbolic representations

        5.  extension/recommendations -\> transducer for seq2seq tasks

Completed
---------

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

Legacy
------

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
