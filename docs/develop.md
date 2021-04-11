## Table of Contents
-   [Tasks](#tasks)
    -   [Manuscript](#manuscript)
    -   [Programming](#programming)
-   [Notes](#notes)
    -   [Admin](#admin)
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

### Manuscript

1.  Post-paper iterations

    1.  Iteration 2 (consistency and formatting)

        1.  Final steps

            1.  update everything from Miku\'s comments

                1.  update objective to be clear -\> improve sentence
                    structure

                2.  update background concepts to mention localized and
                    indirect

                3.  need to update methodologies as well with motivation

                4.  add objective to conclusion as well

            2.  perform spell-check of all text -\> reformat
                spell-corrections if need be

                1.  check that all quotes have page/section attribution

                2.  check that all borrowed figures have \"taken from\"

                3.  ensure all text is padded

            3.  re-read paper for flow and sensibility

                1.  check all figure and table captions -\> remove
                    visualization as well since it is redundant

                2.  look out for possible abbreviation errors

        2.  Formalities

            1.  write abstract and mention results and TauSTE with new
                purpose -\> add it into the TOC as well

            2.  add student registration details and latest date on
                bottom

            3.  take note of all other submission criteria such as
                statement of originality, German abstract, digital copy
                and others, see:
                <https://www.uni-potsdam.de/en/studium/studying/organizing-your-exams/final-thesis>

                1.  20-90 pages thesis length

                2.  add these after TOC and list them into TOC

                3.  add remaining features by referring back to master
                    template

### Programming

1.  Clean-code and documentation

    1.  Source-code

        1.  rename RegexSoftPatternClassifier class to RegexProxy if
            possible without breakages

        2.  change light, medium and heavy to small, medium and large in
            all scripts, filenames and log variables consistently

        3.  add a comment above each code chunk which explains inner
            mechanisms better

    2.  Readme

        1.  update readme and usages with finalized antecedent and proxy
            terminologies

        2.  upadte readme and usages with finalized STE/output neurons
            terminologies

        3.  rename all mentions of regex to RE in readme other than
            usage

        4.  add information on best model downloads and preparation -\>
            add these to Google Drive later on

        5.  update metadata eg. with comprehensive python/shell help
            scripts, comments describing functionality and readme
            descriptions for git hooks

        6.  test out all shell-scripts and python code to make sure
            everything works the same after major renamings

        7.  test download and all other scripts to ensure they work

        8.  perform spell-check on readme

## Notes

### Admin

1.  Timeline

    1.  ~~Initial thesis document: **15.09.2020**~~

    2.  ~~Topic proposal draft: **06.11.2020**~~

    3.  ~~Topic proposal final: **15.11.2020**~~

    4.  ~~Topic registration: **01.02.2021**~~

    5.  Offical manuscript submission: **12.04.2021**

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

2.  FA/WFA extraction

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
