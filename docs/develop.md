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

    1.  Step-by-step

        1.  **TODO** major code refactoring for main model
            (ignore visualization and interpretation) with conversion to
            recent PyTorch (eg. 1.\*) and CUDA versions (eg. 10.\*)

            **DEADLINE:** *\<2020-11-30 Mon\>*

            1.  use brute-force method of running code with newer
                libraries and fix errors on the fly -\> replace outdated
                functions on the fly and make everything clean/modern
                -\> migrate to soft-patterns-pp and clean from there

            2.  find out whether use_rnn is passed by default and what
                its purpose generally is -\> need to understand all
                facets of code to master it

            3.  code works well with latest CUDA and torch, now through
                code chunk by chunk and execute it to test on data,
                understand everything and clean errors on the fly -\>
                worry about performance in later steps -\> for now focus
                on understanding everything

            4.  replace start and end pad token proxies with real
                `[PAD]` tokens which should be ignored by the RNN

                1.  no need to declare variables with autograd
                    explicitly: see
                    <https://stackoverflow.com/questions/57580202/whats-the-purpose-of-torch-autograd-variable>

                2.  UserWarning: size_average and reduce args will be
                    deprecated, please use reduction=\'sum\' instead

                3.  UserWarning: nn.functional.sigmoid is deprecated.
                    Use torch.sigmoid instead

                4.  UserWarning: Implicit dimension choice for
                    log_softmax has been deprecated. Change the call to
                    include dim=X as an argument

            5.  take note that patterns of length one are not allowed,
                ignore `visualize_efficiently.py` for now

        2.  choose exact data set and set up workflow to download and
            pro-process it -\> prefer to find a nice benchmark which can
            be used for extensive comparisons (like RASA NLU)

            **DEADLINE:** *\<2020-12-03 Thu\>*

        3.  work on changes to architecture

            **DEADLINE:** *\<2020-12-24 Thu\>*

            1.  incoporate useful SOPs such as namespace saving and
                printing

            2.  improve code quality with unique model logging and
                tensorboard workflows

            3.  dynamic word embeddings and experimenting with more
                gracious self-loops and epsilon transitions -\> perform
                this incrementally for comparison

            4.  modify final layer to a general additive layer with tree
                structure or soft logic where possible -\> perform this
                incrementally for comparison

            5.  design new and improved test cases using pytest after
                understanding code completely

        4.  run SoPa++ for multiple runs to survey performance -\> run
            on all variants and data-set portions with grid-search to
            get plenty of candidates

            **DEADLINE:** *\<2021-02-01 Mon\>*

        5.  with decent model performance, branch off to improve
            explainability with weighting of patterns -\> do this and
            the previous task simultaneously

            **DEADLINE:** *\<2021-02-01 Mon\>*

    2.  KIV

        1.  current slurm considerations

            1.  activate cuda-8.0 in sbatch scripts explicitly

            2.  use debug mode and low runtime for quick slurm runs

            3.  make list of all useful commands for slurm

        2.  update metadata in scripts later with new workflows, eg.
            with help scripts, comments describing functionality and
            readme descriptions for git hooks

        3.  add MIT license when made public

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

    1.  ~~Initial thesis document: **15.09.20**~~

    2.  ~~Topic proposal draft: **06.11.20**~~

    3.  ~~Topic proposal final: **15.11.20**~~

    4.  Topic registration: **01.02.20**

    5.  Manuscript submission: **18.03.20**

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
