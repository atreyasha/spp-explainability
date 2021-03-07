## Table of Contents
-   [Tasks](#tasks)
    -   [Manuscript](#manuscript)
    -   [Current Programming](#current-programming)
    -   [Future programming](#future-programming)
-   [Notes](#notes)
    -   [Manuscript](#manuscript-1)
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

    1.  \[\#A\] Cross-model comparisons

        1.  use error-bar plot to reflect random seed iterations for
            binary misalignment and softmax norm differences -\> analyze
            relationship with tau threshold vs. performance vs.
            softmax/binary distances

        2.  perhaps change name of visualization scripts to imply it is
            constrained -\> can be done later on to keep names
            consistent with other scripts as well

        3.  keep visualization scripts hard-coded and specific for now,
            they can be made more generalized later on

        4.  change R dependencies later on depending on new json parser

    2.  \[\#A\] Model OOP

        1.  visualize patterns as dots with internal pie charts which
            show magnitudes of each class; displayed as a colour with
            legend, will show relative importance of each binary neuron
            and can help us segment their purposes -\> this should be
            juxtaposed with regex samples to show importance vs. samples

    3.  \[\#A\] Regex OOP

        1.  add visualization of regex ensemble with graphviz -\> can
            specify which pattern to visualize and how to make large
            scale visualizations

    4.  \[\#B\] SoPa++ computational graph

        1.  add visualization of computational graph function using tikz

    5.  \[\#B\] Extra cross-model comparisons

        1.  visualize examples/classes where regex and neural model
            align and misalign, eg. with a confusion matrix

    6.  \[\#B\] FMTOD

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

        3.  try to find literature-based justification for wildcard
            transition -\> if not use omega symbol to differentiate from
            the Kleene star symbol

        4.  add detailed information on how hard SoPa++ model differs
            from SoPa related to transitions and other simplifications
            -\> motivate them using idea of explainable simplification

        5.  comb through terms and iron out usage of patterns vs.
            substrings vs. WFSAs -\> make these consistent and have them
            only refer to one consistent entity, also do not mix
            colloquial and technical terms

        6.  talk about GloVe embeddings, lowercasing and other important
            things

        7.  neural SoPa++ is a black-box (non-transparent) model, regex
            SoPa++ is a transparent model -\> need justifications from
            background concepts, might need significant text on this
            portion

        8.  SoPa++ uses explanation by simplification (globally) -\>
            need justification from background concepts -\> not much use
            of global in paper, but we can make our own arguments

        9.  try to link as much as possible with the background concepts
            for models/explainability concepts

        10. add github link to repo as a footnote

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

        1.  report parameter counts in scores as well

        2.  compare performance to that of other paper(s)

        3.  mention again about test partition difference due to making
            it unique

        4.  consider making test-partition not unique so this could be
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

    2.  Modelling

        1.  extract relevant points from future programming tasks and
            add them here

        2.  extend to a finite-state transducer for seq2seq tasks

        3.  can map linear to decision tree to get clearer picture of
            possibilities

        4.  human intervention inside regex model to monitor/improve
            performance

    3.  Analysis

        1.  analyzing whether patterns can help discover possible
            adversarial patterns

        2.  for the target audience of end-users -\> how can a user make
            use of the regex model

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

            2.  include a section on risks on large NLP models and why
                explainability is necessary with different study

            3.  if possible, try to reduce references to Arrieta et al.
                2020 to reduce perceived over-dependence

            4.  revisit sopa explainability evaluation with three
                guidelines to check if it makes sense after having
                evaluated sopa++ with the same guidelines

        3.  Bibliography

            1.  try to paraphrase as much as possible in background
                concepts otherwise quote statements or definitions to
                signify they are borrowed

            2.  improve capitalization with braces in bibtex file

            3.  if possible, try to find non-arxiv citations for papers

            4.  remove red link color in table of contents

            5.  fine-tune citation color to be consistent with other
                colors

            6.  think about citing Arrieta et al. 2020 in each
                definition, or perhaps it is overkill

            7.  look into antecedent/proxy names and if these can be
                improved

            8.  return to this chapter to add/remove content based on
                requirements of later chapters

        4.  Manuscript admin

            1.  talk to supervisors about many definitions and if these
                are alright

            2.  always mention \"figure taken from study (year)\" when
                using external figures

            3.  fine tune WFSA to mean either automata or automaton,
                make plural abbreviation clear as well

            4.  remove sub-enumeration for single remarks under a
                definition

            5.  add links to different sections later on once structure
                and content is clear

            6.  sort out all abbreviations and standardize formatting in
                terms of where they are first declared

            7.  change to two sided format before printing, as this
                works well for binding/printing

            8.  add Uni-Potsdam originality declaration, or modify
                current one to fit

            9.  add remaining features by referring to master template
                such as abstract (short summarized introduction), list
                of tables/figures/abbreviations, appendices, and all
                others

            10. perform spell-check of everything at the end

### Current Programming

1.  Dependencies, typing and testing

    1.  if using R, document R dependencies with `sessioninfo`

    2.  include basic test code by instantiating class and/or other
        simple methods

    3.  add mypy as an explicit part of testing the source code

    4.  replace Union + None types with Optional type for conciseness

    5.  look into cases where List was replaced by Sequential and how
        this can be changed or understood to keep consistency (ie. keep
        everything to List with overloads)

2.  Documentation and clean-code

    1.  consider removing NLU term and just keep the task as
        intent-detection, as mentioned on the repo

    2.  find better naming for mimic/oracle models which is based on
        research terminology -\> right now mix of neural and regex is
        being used; it would be good to have something more firm

    3.  GPU/CPU runs not always reproducible depending on
        multi-threading, see:
        <https://pytorch.org/docs/stable/notes/randomness.html#reproducibility>

    4.  reduce source code lines, chunking and comments -\> pretty sort
        python code and function/class orders perhaps by length

    5.  add a comment above each code chunk which explains inner
        mechanisms better

    6.  update metadata eg. with comprehensive python/shell help
        scripts, comments describing functionality and readme
        descriptions for git hooks

    7.  add information on best model downloads and preparation -\> add
        these to Google Drive later on

    8.  add pydocstrings to all functions and improve argparse
        documentation

    9.  provide description of data structures (eg. data, labels)
        required for training processes and lowercasing

    10. update/remove git hooks depending on which features are finally
        used, eg. remove pre-push hook

    11. test download and all other scripts to ensure they work

### Future programming

1.  Modelling improvements

    1.  check if packed sequences could be incoporated into model

        1.  might increase efficiency related to batch latency

    2.  find single-threaded ways to speed up regular expression
        searches -\> bottleneck appears to be search method

        1.  multiprocessing with specific chunksize seems to have some
            effect

        2.  might need to have a very large batch size to see any
            improvements with multiprocessing

        3.  database with indexing could help improve regex lookup speed

    3.  consider using finditer for regex lookup with trace, since we
        should return all matches

        1.  make activating text unique in case we return multiple texts
            and not one -\> but then won\'t correspond to activating
            regexes

        2.  might not make a huge difference since we use short
            sentences

        3.  might be better for speed reasons to leave it as a search
            method

    4.  add predict function for both mimic and oracle model which does
        not need extra data to be loaded -\> can also accept stdin as
        unix pipe

2.  Explore activation generalization methods

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

## Notes

### Manuscript

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

    3.  base main ideas off peer-reviewed artics

### Admin

1.  Research questions

    1.  To what extent does SoPa++ contribute to competitive performance
        on NLU tasks?

    2.  To what extent does SoPa++ contribute to explainability by
        simplification?

    3.  What interesting and relevant explanations does SoPa++ provide
        on NLU task(s)?

2.  Timeline

    1.  ~~Initial thesis document: **15.09.2020**~~

    2.  ~~Topic proposal draft: **06.11.2020**~~

    3.  ~~Topic proposal final: **15.11.2020**~~

    4.  ~~Topic registration: **01.02.2021**~~

    5.  Manuscript draft submission: **31.03.2021**

    6.  Offical manuscript submission: **10.04.2021**

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
