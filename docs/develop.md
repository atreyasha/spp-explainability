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

1.  Visualization and summary-statistics

    **DEADLINE:** *\<2021-03-11 Thu\>*

    1.  \[\#A\] Binary neurons and regex\'s

        1.  use python script to pipe softmax values to temporary csv
            file and plot them with `ggplot` in `R`

        2.  add visualization of regex ensemble with graphviz which can
            be done fully in python

        3.  think about file naming again given this situation with
            mixed functionalities

    2.  \[\#B\] SoPa++ computational graph

        1.  add visualization of computational graph function using tikz

    3.  \[\#B\] FMTOD

        1.  add visualisation of data statistics with different
            partitions, perhaps as a stacked bar chart

2.  Methodologies

    1.  FMTOD data set

        1.  provide summary statistics and explanation about the data
            set

        2.  provide a visualization on the data set and its splits

        3.  mention that data set was altered to keep it in good quality
            by removing duplicates, perhaps this can be re-tested
            without such processing or used as a limitation/further-work
            -\> although results appear to show that performance metric
            improve when using original data set because of duplicates
            and overlaps between train and test sets

        4.  talk about lowercasing, upsampling and other important
            information

    2.  Model

        1.  update the WFSA definitions to signify wildcard transitions
            and mention why this is found here and not in the background
            concepts since we could not find literature which defined it
            in a similar way

        2.  add pseudocode for various segments -\> would help to cement
            certain concepts

        3.  look out for misconception between tau and transition matrix
            symbol

        4.  try to find literature-based justification for wildcard
            transition -\> if not use omega symbol to differentiate from
            the Kleene star symbol

        5.  add detailed information on how hard SoPa++ model differs
            from SoPa related to transitions and other simplifications
            -\> motivate them using idea of explainable simplification

        6.  comb through terms and iron out usage of patterns vs.
            substrings vs. WFSAs -\> make these consistent and have them
            only refer to one consistent entity, also do not mix
            colloquial and technical terms

        7.  talk about GloVe embeddings, lowercasing and other important
            things

        8.  neural SoPa++ is a black-box (non-transparent) model, regex
            proxy is a transparent model -\> need justifications from
            background concepts, might need significant text on this
            portion

        9.  SoPa++ uses explanation by simplification (globally) -\>
            need justification from background concepts -\> not much use
            of global in paper, but we can make our own arguments

        10. try to link as much as possible with the background concepts
            for models/explainability concepts

        11. add github link to repo as a footnote

    3.  Explainability

        1.  explain how we make explanations by simplification work
            altogether

        2.  utilize antecendent/proxy model terminologies

        3.  mention tangible metric for simplification -\> how close is
            proxy to antecedent

        4.  untangible metric is how satisfying the proxy model is to a
            target audience -\> future work

        5.  hard to find hierarchies of good vs. not-good
            explainability, but we can argue that we tried a different
            explainability method, i.e. explanation by simplification
            with a global simplified model vs.
            local-explanations/feature-relevance -\> also we can use the
            constrictive argument from the paper and others

        6.  make claim that SoPa++ explainability has a different
            explainability taxonomy from that of vanilla SoPa, but
            don\'t claim it is decisively better

        7.  use the three good explainability criteria to show that our
            technique might be better theoretically, but the real test
            would have to be done with a target audience\'s survey

        8.  mention that the target audience of this explainability
            method is domain experts, since it is still very complicated

        9.  link back to background concepts when we discuss bringing
            neural and regex model as close to each other as possible

    4.  Quantization/Binarization

        1.  explain how and why we modified STE to TauSTE

        2.  how does binarizing help with explainability, justify
            requirement for it in both training and testing

    5.  Training/Evaluation/Explainability-evaluation

        1.  provide extensive details of training setup

        2.  provide extensive details of evaluating neural/regex models

        3.  provide extensive details of evaluating explanations by
            simplification \"metric\" of neural-regex models -\> which
            should make results clearer

        4.  **important:** everything shown in the \"Results\" section
            should be motivated or introduced here

    6.  Visualizations

        1.  add visualization of TauSTE function

        2.  produce manual computational graph using tikz, building from
            what was made earlier

        3.  add visualization of in-depth computational graph in paper
            for clarity -\> this can be automated with PyTorch tools

3.  Results

    1.  Report F_1 scores and performances of neural models

        1.  modify visualize scripts to aggregate and print summary
            stats in script to re-use later in paper with means and
            standard deviations across random seeds

        2.  report parameter counts in scores as well

        3.  compare performance to that of other paper(s)

        4.  mention again about test partition difference due to making
            it unique

        5.  consider making test-partition not unique so this could be
            used to compare with other studies

    2.  Relationship between tau threshold vs. performance vs.
        softmax/binary distances

        1.  compute statistics with random-seed deviations over
            inter-model comparisons such as average distance,
            misalignment, activation frequency and other useful metrics
            that can elucidate on-the-ground processes

    3.  Visualizations

        1.  show visualization of training performance timelines, think
            about how to keep most important information

        2.  show visualization of tau threshold vs. performance vs.
            softmax/binary distances with error bars for random seed
            iterations

        3.  show confusion matrix between regex and neural models to
            show alignment/misalignment, if this is necessary -\> but it
            will show interesting class differences

4.  Discussion

    1.  Expound on trade-off between performance and transparency by
        looking at differently sized models

    2.  Discuss relationship between tau threshold and the
        softmax/binary distances

    3.  Think about why larger regex models tend to show more
        misalignment from neural counterparts compared to smaller models

    4.  Visualizations

        1.  show visualizations of important patterns in a regex model
            -\> something which is small and fits well into a page

        2.  show TikZ visualization of each binary neuron\'s relative
            importance for classes -\> would be interesting to see how
            saturated these are

5.  Conclusions

    1.  Summarize everything in manuscript

    2.  Address research questions

6.  Further work

    1.  Quality of explainability

        1.  this is subjective and a survey from the target audience
            would be good to have

    2.  Modeling

        1.  use packed sequences for fast processing

        2.  use multiple-threads for extracting regular expressions, or
            store them in a database

        3.  more thorough regex lookup since now only the first one
            causes a loop breakage

        4.  add predict function for people to test with arbitrary
            sequences

        5.  consider using nearest-neighbours to expand adjacent tokens
            on already found regex\'s

        6.  consider internal regex UNK token handling cases other than
            wildcard presence

        7.  consider semantic clustering of digits or other objects to
            help achieve generality

        8.  extend to a finite-state transducer for seq2seq tasks

        9.  can map linear to decision tree to get clearer picture of
            possibilities

        10. human intervention inside regex model to monitor/improve
            performance

    3.  Analysis

        1.  extension to more NLU data sets such as SNIPS, ATIS

        2.  analyzing whether patterns can help discover possible
            adversarial patterns

        3.  for the target audience of end-users -\> how can a user make
            use of the regex model

        4.  visualize examples/classes where regex and neural model
            align and misalign, eg. with a confusion matrix

7.  Formatting

    1.  Paper length

        1.  20-90 pages thesis length -\> try to keep ideas
            well-motivated yet succinct

    2.  Points to address towards end

        1.  Introduction

            1.  fine-tune introduction with new details from other
                chapters

            2.  update motivations from Arrieta et al. 2020 \"What for\"
                section

            3.  add C-like reference to explain what SoPa++ means like
                in i++

            4.  add links to chapters in thesis structure, improve
                formatting

        2.  Background concepts

            1.  think about providing an additional definition for
                \"understandability\"

            2.  consider quoting all definitions to further imply that
                they are exactly taken from other studies

            3.  add a Kleene-star operator mention to remark 9.4

            4.  include a section on risks on large NLP models and why
                explainability is necessary with different study

            5.  if possible, try to reduce references to Arrieta et al.
                2020 to reduce perceived over-dependence

            6.  revisit sopa explainability evaluation with three
                guidelines to check if it makes sense after having
                evaluated sopa++ with the same guidelines

            7.  look into antecedent/proxy names and if these can be
                improved

            8.  return to this chapter to add/remove content based on
                requirements of later chapters

        3.  Bibliography

            1.  improve capitalization with braces in bibtex file

            2.  if possible, try to find non-arxiv citations for papers

            3.  remove red link color in table of contents

            4.  fine-tune citation color to be consistent with other
                colors

        4.  Manuscript admin

            1.  consider replacing legacy-sopa figures with pdf extracts
                instead of screenshots

            2.  talk to supervisors about many definitions and if these
                are alright

            3.  always mention \"figure taken from study (year)\" when
                using external figures

            4.  fine tune WFSA to mean either automata or automaton,
                make plural abbreviation clear as well

            5.  remove sub-enumeration for single remarks under a
                definition

            6.  add links to different sections later on once structure
                and content is clear

            7.  sort out all abbreviations and standardize formatting in
                terms of where they are first declared

            8.  change to two sided format before printing, as this
                works well for binding/printing

            9.  add Uni-Potsdam originality declaration, or modify
                current one to fit

            10. add remaining features by referring to master template
                such as abstract (short summarized introduction), list
                of tables/figures/abbreviations, appendices, and all
                others

            11. perform spell-check of everything at the end

### Programming

1.  Dependencies, typing and testing

    1.  if using R, document R dependencies with `sessionInfo()`

    2.  add mypy as an explicit part of testing the source code

    3.  replace Union + None types with Optional type for conciseness

    4.  look into cases where List was replaced by Sequential and how
        this can be changed or understood to keep consistency (ie. keep
        everything to List with overloads)

    5.  include basic test code by instantiating class and/or other
        simple methods

2.  Documentation and clean-code

    1.  Terminology-based modifications post-paper

        1.  if necessary, apply further script renaming using antecedent
            and proxy terminologies -\> update readme and usages

        2.  consider removing `utils` extension from all utils scripts
            since these might be unnecessary

        3.  test out all shell-scripts and python code to make sure
            everything works the same after major renamings

    2.  Others

        1.  GPU/CPU runs not always reproducible depending on
            multi-threading, see:
            <https://pytorch.org/docs/stable/notes/randomness.html#reproducibility>

        2.  reduce source code lines, chunking and comments -\> pretty
            sort python code and function/class orders perhaps by length

        3.  add a comment above each code chunk which explains inner
            mechanisms better

        4.  update metadata eg. with comprehensive python/shell help
            scripts, comments describing functionality and readme
            descriptions for git hooks

        5.  add information on best model downloads and preparation -\>
            add these to Google Drive later on

        6.  add pydocstrings to all functions and improve argparse
            documentation

        7.  provide description of data structures (eg. data, labels)
            required for training processes and lowercasing

        8.  update/remove git hooks depending on which features are
            finally used, eg. remove pre-push hook

        9.  test download and all other scripts to ensure they work

        10. perform spell-check on readme

## Notes

### Admin

1.  Timeline

    1.  ~~Initial thesis document: **15.09.2020**~~

    2.  ~~Topic proposal draft: **06.11.2020**~~

    3.  ~~Topic proposal final: **15.11.2020**~~

    4.  ~~Topic registration: **01.02.2021**~~

    5.  Manuscript draft submission: **31.03.2021**

    6.  Offical manuscript submission: **11.04.2021**

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
