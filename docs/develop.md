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

1.  Clean-code and documentation

    1.  work on prototyping basic results from SoPa

        1.  unsure what self-loops and fwd-1s mean in patterns.log -\>
            make GitHub issue to ask regarding this

        2.  **TODO** analyze pattern log more closely with
            code on the side to understand what it means -\> can start
            writing early when things start to make sense

        3.  research question could be whether SoPa could extend towards
            global explainability

        4.  think more about local vs. global explainability and how
            SoPa could help achieve this -\> might have issues wrt.
            convergence as mentioned in paper

        5.  **TODO** write proposal with key research
            questions -\> address points directly from step 3 document
            requirements -\> prepare some basic accuracy metrics and
            interpretations from best model

    2.  work on refactoring code and upgrading/porting dependencies to
        more recent versions

        1.  incoporate useful SOPs such as namespace saving and printing

        2.  issue of strange numbers of start and end tokens -\> should
            be replaced with pad tokens

    3.  run tests on s3it with slurm

        1.  activate cuda-8.0 in sbatch scripts explicitly

        2.  use debug mode and low runtime for quick slurm runs

        3.  make list of all useful commands for slurm

    4.  update metadata in scripts later with new workflows, eg. with
        help scripts and comments describing functionality

    5.  add MIT license when made public

2.  Interpretable neural architectures

    1.  Soft patterns (SoPa)

        1.  good: practical new architecture which maps to RNN-CNN mix
            via WFSAs

        2.  good: code quality in PyTorch, lengthy code

        3.  good: contact made with author and could get advice for
            possible extensions

        4.  problematic: global explainability might be a far shot,
            occlusion is still used for documents

        5.  problematic: not clear how this could be linked to a final
            WFSA -\> perhaps it is ultimately not but it is still
            interpretable and explainable

        6.  possible extensions: improve on larger data, learnable word
            embeddings, sub-word pre-processing to leverage morphology,
            increase generalization with wildcards, improve
            interpretability via in-built method instead of occlusion in
            document analysis setting, final additive layer for finding
            relevance of patterns, multi-class tasks instead of only
            binary

    2.  **GIST:** likely higher performance due to direct inference and
        less costly

3.  Data sets

    1.  NLU data sets -\> single sequence intent classification,
        typically many classes involved -\> eg. ATIS, Snips,
        AskUbuntuCorpus, FB task oriented dataset (mostly intent
        classifications)

    2.  NLI data sets -\> two sequences for predicting entailment,
        contradiction, neutral -\> eg. SNLI, MNLI, XNLI

    3.  SOTA scores for NLU can be found on
        <https://github.com/nghuyong/rasa-nlu-benchmark#result>

    4.  SOTA scores for NLI can be found on
        <https://nlp.stanford.edu/projects/snli/>

    5.  **consideration:** use both small and large data sets to get an
        idea of performance

    6.  **GIST:** easier to work with NLU data sets since these only
        involve one data set to start off with

4.  Constraints

    1.  work with RNNs only

    2.  seq2cls tasks -\> eg. NLU/NLI/semantic tasks, try to work with
        simpler single (vs. double) sequence classification task

    3.  base main ideas off peer-reviewed articles

5.  High-level

    1.  **globally explainable** -\> exposes inner mechanisms and global
        biases which could help for ethical and adversarial problem
        detections

    2.  **high-performance** -\> competitive with similar
        non-explainable learning techniques

    3.  **contributions** -\> should add insights which are new and not
        commonly found in research so far

### Admin

1.  General timeline

    1.  ~~Initial thesis document: 15.09.20~~

    2.  Topic proposal draft: 06.11.20

    3.  Topic proposal final: 15.11.20

    4.  Topic registration: 01.02.20

    5.  Manuscript submission: 18.03.20, try to extend if possible

    6.  **Note:** meeting every 3 weeks with Sharid and more regularly
        with Mathias

2.  Manuscript notes

    1.  semirings, abstract algebra and how they are used for
        finite-state machines in Forward and Viterbi algorithms -\> go
        deeper into this to get some background

    2.  Chomsky hierarchy of languages -\> might be relevant especially
        relating to CFGs

    3.  FSA/WFSAs -\> input theoretical CS, mathematics background to
        describe these

    4.  ANN\'s historical literature -\> describe how ANNs approximate
        symbolic representations

    5.  extension/recommendations -\> transducer for seq2seq tasks

Completed
---------

**DONE** add large amounts of binary data for testing with
CPU/GPU -\> requires pre-processing

**DONE** find re-usable code for running grid search -\>
otherwise construct makeshift quick code

CLOSED: \[2020-11-03 Tue 21:07\]

**DONE** test SoPa on sample data in repository to ensure it
works out-of-the-box -\> try this on laptop and s3it

**DONE** make workflow to reproduce virtual environment
cleanly via poetry

**DONE** make workflow to download simple but high-quality
NLU dataset and glove data sets

**DONE** read more into these tasks and find one that has
potential for interpretability -\> likely reduce task to binary case for
easier processing (eg. entailment)

**DONE** search for popular NLI datasets which have existing
RNN models as (almost) SOTAs, possibly use ones that were already tested
for eg. RTC or ones used in papers that may have semantic element

**DONE** explore below frameworks (by preference) and find
most feasible one

**DONE** add org-mode hook to remove startup visibility
headers in org-mode to markdown conversion

**DONE** Set up repo, manuscript and develop log

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

1.  research questions:

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

3.  research questions:

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
