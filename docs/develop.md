Tasks
-----

### Main research direction

1.  Current

    1.  **TODO** rank articles on relevance to XAI,
        feasibility, presence of source code or pseudcode, etc.

        1.  methods with new architectures -\> eg. SoPa,
            state-regularized RNNs, FA-RNNs -\> more direct learning
            where interpretability is built into main model -\> might
            make sense to take an existing framework and extend it to
            interesting tasks and probe this for the extent of
            explainability

        2.  methods for FSA/WFSA extraction -\> eg. spectral learning,
            clustering -\> less direct interpretability and more proof
            of performance needed -\> need to show it is better than
            simple data learning, might be more costly

        3.  SR-RNNs and FA-RNNs: both are very practical and relatively
            simpler than the next two studies -\> one possibility could
            be extending state-regularized RNNs (and pytorch/cuda
            porting) to include conversion to REs at the end for best
            interpretability -\> as a bridge between both studies,
            downside is the code is not very well developed

        4.  SoPA and rational recurrences: first is a practical approach
            while second is highly theoretical. Both have good code
            quality

    2.  constraint: work with RNNs only

    3.  constraint: seq2cls tasks -\> eg. NLU/semantic, paraphrase
        detection

    4.  constraint: base main ideas off peer-reviewed articles

2.  Long-term

    1.  develop a feasible, practical and interesting research question,
        writing will be easy when everything else is well defined

    2.  think about what this research adds that is not present -\>
        possible to look at next steps in existing articles or possibly
        extension to new sequence classification tasks

    3.  high-performance -\> show it is better than basic learning
        method from data -\> implies we would have to use some negative
        examples from the oracle as well

    4.  explainable -\> could potentially expose global bias for ethical
        and adversarial problem detections

    5.  keep probing networks/tasks as a backup option

### Admin

1.  keep good communication with supervisors -\> every 3 weeks for
    Sharid and more regularly with Mathias

2.  General timeline

    1.  ~~Initial thesis document: 15.09.20~~

    2.  Topic proposal draft: 06.11.20

    3.  Topic proposal final: 15.11.20

    4.  Topic registration: 01.02.20

    5.  Manuscript submission: 18.03.20, try to extend if possible

### Manuscript-specifics

1.  read more about turing machines and FSA/WFAs to get theoretical
    background

2.  ann\'s historical literature find all -\> especially focusing on how
    ANNs approximate symbolic representations which would motivate
    overall topic

3.  convergence, universal approximation and generalization are
    satisfied by ANNs to a high degree, semantic relevance in the final
    model is not guaranteed and this needs to be an additional task that
    where symbolic frameworks are needed

4.  limit main experiments on sequence classification but mention
    transducer extension to seq2seq

5.  if possible, bring in theoretical CS and mathematics into paper

Legacy
------

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
