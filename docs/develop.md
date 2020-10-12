Development
===========

Tasks
-----

### Main research direction

1.  Day-to-day

    1.  **TODO** find applications of FSA/WFAs in NLP -\>
        decipher if they are generally performant and/or explainable

    2.  **TODO** look into algorithms to extract FSA/WFAs
        from RNNs

    3.  **TODO** if all works, finalize seq2cls tasks -\> eg.
        verb-object agreement, NLU, paraphrase detection, or otherwise
        semantics-oriented tasks

    4.  **TODO** read more about turing machines and FSA/WFAs
        to get theoretical background

    5.  constraints: work with RNNs, focus on NLU/semantic seq2cls task

    6.  constraints: base main ideas off peer-reviewed articles

2.  Big-picture

    1.  think about what this research adds that is not present

    2.  high-performance -\> show it is better than basic learning
        method from data -\> implies we would have to use some negative
        examples from the oracle as well

    3.  explainable -\> could potentially expose global bias for ethical
        and adversarial problem detections

    4.  keep probing networks/tasks as a backup option

### Admin

1.  General timeline:

      Task                      Deadline     Details
      ------------------------- ------------ -------------------------------------
      Initial thesis document   `15.09.20`   \-
      Topic proposal draft      06.11.20     Finalize general task and algorithm
      Topic proposal final      15.11.20     \-
      Topic registration        01.02.20     \-
      Manuscript submission     18.03.20     Consider applying for extension

    1.  keep good communication with supervisors -\> every 3 weeks for
        Sharid and more regularly with Mathias

### Manuscript-specifics

1.  ANNs historical literature find all -\> especially focusing on how
    ANNs approximate symbolic representations which would motivate
    overall topic

2.  convergence, universal approximation and generalization are
    satisfied by ANNs to a high degree, semantic relevance in the final
    model is not guaranteed and this needs to be an additional task that
    where symbolic frameworks are needed

3.  limit main experiments on sequence classification but mention
    transducer extension to seq2seq

4.  if possible, bring in theoretical CS and mathematics into paper

Brainstorming
-------------

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

1.  consider using differentiable ILP from DeepMind

2.  can potentially use existing IM models such as paraphrase detector
    for introspection purposes in thesis

3.  n-gram power sets to explore for statistical artefacts -\> ANNs can
    only access the search space of N-gram power sets -\> solution to
    NLP tasks must be a statistical solution within the power sets which
    links back to symbolism

4.  propositional logic only contains atoms while predicate/first-order
    logic contain variables
