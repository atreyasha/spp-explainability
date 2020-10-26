## Table of Contents
-   [Tasks](#tasks)
    -   [Research](#research)
    -   [Admin](#admin)
-   [Completed](#completed)
-   [Legacy](#legacy)
    -   [Interpretable neural
        architectures](#interpretable-neural-architectures-1)
    -   [Interpretable surrogate
        extraction](#interpretable-surrogate-extraction)
    -   [Neuro-symbolic paradigms](#neuro-symbolic-paradigms)
    -   [Neural decision trees](#neural-decision-trees)
    -   [Inductive logic on NLP search
        spaces](#inductive-logic-on-nlp-search-spaces)

Tasks
-----

### Research

1.  High-level

    1.  **globally explainable** -\> exposes inner mechanisms and global
        biases which could help for ethical and adversarial problem
        detections

    2.  **high-performance** -\> competitive with similar
        non-explainable learnint techniques

    3.  **contributions** -\> should add insights which are new and not
        commonly found in research so far

2.  Data sets

    1.  **TODO** search for popular NLU datasets which have
        existing RNN models as (almost) SOTAs, possibly use ones that
        were already tested for eg. RTC or ones used in papers that may
        have semantic element

    2.  **TODO** read more into these tasks and find one that
        has potential for interpretability -\> likely reduce task to
        binary case for easier processing (eg. entailment)

3.  Clean-code and documentation

    1.  **TODO** write proposal and manuscript with key
        research questions which can be answered

    2.  start populating repository with hooks, data downloads,
        documentation and models

4.  Interpretable neural architectures

    1.  Soft patterns (SoPa)

        1.  good: practical new architecture which maps to RNN-CNN mix
            via WFSAs

        2.  good: code quality in PyTorch, lengthy code

        3.  good: contact made with author and could get advice for
            possible extensions

        4.  problematic: global explainability might be a far shot,
            occlusion is still used for documents

        5.  problematic: not clear how this could be linked to a final
            WFSA -\> perhaps it is ultimately not

        6.  possible extensions: improve on larger data, learnable word
            embeddings, sub-word pre-processing to leverage morphology,
            increase generalization with wildcards, improve
            interpretability via in-built method instead of occlusion in
            document analysis setting, final additive layer for finding
            relevance of patterns

    2.  State-regularized-RNNs (SR-RNNs)

        1.  good: very powerful and easily interpretable architecture
            with extensions to NLP and CV

        2.  good: simple code which can probably be ported to PyTorch
            relatively quickly

        3.  good: contact made with author and could get advice for
            possible extensions

        4.  problematic: code is outdated and written in Theano,
            TensorFlow version likely to be out by end of year

        5.  problematic: DFA extraction from SR-RNNs is clear, but DPDA
            extraction/visualization from SR-LSTMs is not clear probably
            because of no analog for discrete stack symbols from
            continuous cell (memory) states

        6.  possible extensions: port state-regularized RNNs to PyTorch
            (might be simple since code-base is generally simple), final
            conversion to REs for interpretability, global
            explainability for natural language, adding differnet loss
            to ensure words cluster to same centroid as much as possible
            -\> or construct large automata, perhaps pursue sentiment
            analysis from SR-RNNs perspective instead and derive DFAs to
            model these

    3.  **GIST:** likely higher performance due to direct inference and
        less costly

5.  Constraints

    1.  work with RNNs only

    2.  seq2cls tasks -\> eg. NLU/semantic, paraphrase detection

    3.  base main ideas off peer-reviewed articles

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

    1.  FSA/WFSAs -\> input theoretical CS, mathematics background to
        describe these

    2.  ann\'s historical literature -\> describe how ANNs approximate
        symbolic representations

    3.  extension/recommendations -\> transducer for seq2seq tasks

Completed
---------

**DONE** explore below frameworks (by preference) and find
most feasible one

**DONE** add org-mode hook to remove startup visibility
headers in org-mode to markdown conversion

**DONE** Set up repo, manuscript and develop log

Legacy
------

### Interpretable neural architectures {#interpretable-neural-architectures-1}

1.  Rational recurences (RRNNs)

    1.  good: code quality in PyTorch, succinct and short

    2.  good: heavy mathematical background which could lend to more
        interesting mathematical analyses

    3.  problematic: seemingly missing interpretability section in paper
        -\> theoretical and mathematical, which is good for
        understanding

    4.  problematic: hard to draw exact connection to interpretability,
        might take too long to understand everything

2.  Finite-automation-RNNs (FA-RNNs)

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
