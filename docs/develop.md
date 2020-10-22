Tasks
-----

### Research

1.  Interpretable neural architectures

    1.  **TODO** explore below frameworks (by preference) and
        find most feasible one

    2.  SoPA and rational recurrences

        1.  first is a practical approach while second is highly
            theoretical

        2.  provide interpretable architectures which need to be further
            explored

        3.  both implementations have decent code quality -\> could be
            extended

    3.  State-regularized-RNNs

        1.  practical and offers direct interpretability from
            architecture

        2.  code is outdated and written in Theano, TensorFlow version
            likely to be out by end of year

        3.  possible work: port state-regularized RNNs to PyTorch with
            CUDA headers

        4.  final conversion to REs for interpretability

    4.  Finite-automation-RNNs

        1.  source code likely released by November, but still requires
            initial REs which may not be present

        2.  FA-RNNs involving REs and substitutions could be useful
            extensions as finite state transducers for interpretable
            neural machine translation

    5.  ****GIST:**** likely higher performance due to direct inference
        and less costly

2.  Clean-code and documentation

    1.  **TODO** start populating repository with hooks, data
        downloads, documentation and models

    2.  populate manuscript and repository with key research questions
        which can be answered

3.  Data sets

    1.  **TODO** search for popular NLU datasets which have
        existing RNN models as (almost) SOTAs

    2.  **TODO** read more into these tasks and find one that
        has potential for interpretability -\> likely reduce task to
        binary case for easier processing (eg. entailment)

4.  Constraints

    1.  work with RNNs only

    2.  seq2cls tasks -\> eg. NLU/semantic, paraphrase detection

    3.  base main ideas off peer-reviewed articles

5.  High-level

    1.  ****globally explainable**** -\> exposes inner mechanisms and
        global biases which could help for ethical and adversarial
        problem detections

    2.  ****high-performance**** -\> competitive with similar
        non-explainable learnint techniques

    3.  ****contributions**** -\> should add insights which are new and
        not commonly found in research so far

### Admin

1.  General timeline

    1.  ~~Initial thesis document: 15.09.20~~

    2.  Topic proposal draft: 06.11.20

    3.  Topic proposal final: 15.11.20

    4.  Topic registration: 01.02.20

    5.  Manuscript submission: 18.03.20, try to extend if possible

    6.  ****Note:**** meeting every 3 weeks with Sharid and more
        regularly with Mathias

2.  Manuscript notes

    1.  FSA/WFAs -\> input theoretical CS, mathematics background to
        describe these

    2.  ann\'s historical literature -\> describe how ANNs approximate
        symbolic representations

    3.  extension/recommendations -\> transducer for seq2seq tasks

Completed
---------

**DONE** add org-mode hook to remove startup visibility
headers in org-mode to markdown conversion

**DONE** Set up repo, manuscript and develop log

Legacy
------

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
