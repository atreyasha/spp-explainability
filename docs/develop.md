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

1.  Background concepts

    **DEADLINE:** *\<2021-03-05 Fri\>*

    1.  Explainability

        1.  Basics of transparent/black-box models

            1.  perhaps add some remarks to explain transparency
                categories -\> paraphrase these to keep things simple

            2.  define criteria to classify models into transparent vs.
                black-box (as the opposite of transparent models) -\>
                make these distinctions very clear

            3.  black box models (opaque) vs. transparent models -\>
                give examples of each

            4.  mention contrast between previous AI surge vs. now (more
                black-box techniques) -\> also why it is important for
                XAI to exist with more black-box models (repeat
                introduction points)

        2.  Explainability techniques

            1.  bring up different post-hoc explainability techniques,
                with explanations by simplification definition and other
                ones such as local explanations, feature-relevance and
                examples such as occlusion, LIME and others

            2.  improve phrasing of oracle vs. mimic model names -\>
                maybe antecedent and proxy models

            3.  mention examples of research conducting extraction of
                FSA/WFSA from RNNs, link then to next section on WFSA

        3.  Interesting insights

            1.  bring up concept of trade-off between performance and
                trasparency

            2.  problem of a lack of XAI metrics -\> we could address
                this by our distance metrics to provide some insight but
                this might not be enough

            3.  audience needed to evaluate -\> link to future work -\>
                but some psychological conclusions can be made about
                constrictiveness of explanations which is achieved by
                our model but not the previous one

            4.  possibly explanations are better when constrictive, can
                use other reference to justify this too

            5.  add images/visualizations where possible

    2.  STE layer

        1.  Add historical literature from 2013/2016/2019 papers

            1.  justify forward and backward passes using the 2013/2016
                papers, as well as the 2019 paper if possible

        2.  Add visual of STE function along with single-line (min-max)
            and partial function equation representations

    3.  Legacy SoPa

        1.  Model

            1.  mention epsilons and self-loops

            2.  mention we only use one start and end vectors and not
                all -\> because must start and end there

            3.  mention time complexity differences and other things
                that are different from definitions -\> sparse
                transition matrix

            4.  mention other deviations from main equation and ensemble
                of WFSAs, semirings etc

            5.  talk about performance and other general basics, time
                complexities where relevant

        2.  Explainability

            1.  SoPa is a black-box model -\> justify using previous
                definitions

            2.  SoPa uses local explanations and/or feature relevance
                techniques -\> need evidence from background concepts

            3.  reference topics from explainability to mention current
                taxonomy/hierarchy

        3.  Use SoPa explanation here to talk about limitations and
            fixes in next chapter, which should provide a clean flow to
            the next chapter\'s content

            1.  add images where possible

2.  Visualization and summary-statistics

    **DEADLINE:** *\<2021-03-11 Thu\>*

    1.  Cross-model comparisons

        1.  compute statistics with random-seed deviations over
            inter-model comparisons such as average distance,
            misalignment, activation frequency and other useful metrics
            that can elucidate on-the-ground processes

        2.  use error-bar plot to reflect random seed iterations for
            binary misalignment and softmax norm differences -\> analyze
            relationship with tau threshold vs. performance vs.
            softmax/binary distances

        3.  visualize examples where regex and neural model align and
            misalign, eg. with a confusion matrix

    2.  Regex OOP

        1.  add visualization of regex ensemble with graphviz -\> can
            specify which pattern to visualize and how to make large
            scale visualizations

        2.  visualize patterns as dots with internal pie charts which
            show magnitudes of each class; displayed as a colour with
            legend, will show relative importance of each binary neuron
            and can help us segment their purposes

    3.  Model runs

        1.  add visualizations of model runs using dedicated functions,
            preferably all using python libraries, or otherwise
            defaulting to R libraries

    4.  SoPa++ computational graph

        1.  visualize STE function

        2.  visualize TauSTE function

        3.  add visualization of computational graph function using tikz

        4.  add automated computational graph as well

3.  Methodologies

    1.  FMTOD data set

        1.  provide summary statistics and explanation about the data
            set

        2.  mention that data set was altered to keep it in good quality
            by removing duplicates, perhaps this can be re-tested
            without such processing or used as a limitation/further-work
            -\> although results appear to show that performance metric
            improve when using original data set because of duplicates
            and overlaps between train and test sets

    2.  Model

        1.  motivate structure in terms of RNNs, CNNs and WFSA where
            possible

        2.  add pseudocode for various segments -\> would help to cement
            certain concepts

        3.  add detailed information on how hard SoPa++ model differs
            from SoPa related to transitions and other simplifications
            -\> motivate them using idea of explainable simplification

        4.  neural SoPa++ is a black-box (non-transparent) model, regex
            SoPa++ is a transparent model -\> need justifications from
            background concepts

        5.  SoPa++ uses explanation by simplification (globally) -\>
            need justification from background concepts -\> not much use
            of global in paper, but we can make our own arguments

    3.  Explainability

        1.  explain how we make explanations by simplification work
            altogether

        2.  hard to find hierarchies of good vs. not-good
            explainability, but we can argue that we tried a different
            explainability method, i.e. explanation by simplification
            with a global simplified model vs.
            local-explanations/feature-relevance -\> also we can use the
            constrictive argument from the paper

        3.  make claim that SoPa++ explainability has a different
            explainability taxonomy from that of vanilla SoPa, but
            don\'t claim it is decisively better

        4.  mention that the target audience of this explainability
            method is domain experts, since it is still very complicated

        5.  link back to background concepts when we discuss bringing
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

4.  Results

    1.  Report F_1 scores and performances of neural models

        1.  compare performance to that of other paper(s)

        2.  mention again about test partition difference due to making
            it unique

        3.  consider making test-partition not unique so this could be
            used to compare with other studies

    2.  Relationship between tau threshold vs. performance vs.
        softmax/binary distances

    3.  Visualizations

        1.  show visualization of training performance timelines, think
            about how to keep most important information

        2.  show visualizations of important patterns in a regex model
            -\> something which is small and fits well into a page

        3.  show visualization of tau threshold vs. performance vs.
            softmax/binary distances with error bars for random seed
            iterations

        4.  show confusion matrix between regex and neural models to
            show alignment/misalignment, if this is necessary -\> but it
            will show interesting class differences

5.  Discussion

    1.  Expound on trade-off between performance and transparency by
        looking at differently sized models

    2.  Discuss relationship between tau threshold and the
        softmax/binary distances

    3.  Think about why larger regex models tend to show more
        misalignment from neural counterparts compared to smaller models

    4.  Visualizations

        1.  show TikZ visualization of each binary neuron\'s relative
            importance for classes -\> would be interesting to see how
            saturated these are

6.  Conclusions

    1.  Summarize everything in manuscript

    2.  Address research questions

7.  Further work

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

8.  Formatting

    1.  Paper length

        1.  20-90 pages thesis length -\> try to keep ideas
            well-motivated yet succinct

    2.  Points to address towards end

        1.  Introduction

            1.  add C-like reference to explain what SoPa++ means like
                in i++

            2.  fix introduction with new details from other chapters

            3.  update motivations from Arrieta et al. 2020 \"What for\"
                section

            4.  add links to chapters in thesis structure, improve
                formatting

        2.  Bibliography

            1.  improve capitalization with braces in bibtex file

            2.  if possible, try to find non-arxiv citations for papers

            3.  remove red link color in table of contents

            4.  fine-tune citation color to be consistent with other
                colors

        3.  Manuscript admin

            1.  sort out all abbreviations and standardize formatting in
                terms of where they are first declared

            2.  change to two sided format before printing, as this
                works well for binding/printing

            3.  add Uni-Potsdam originality declaration, or modify
                current one to fit

            4.  add remaining features by referring to master template
                such as abstract (short summarized introduction), list
                of tables/figures/abbreviations, appendices, and all
                others

            5.  perform spell-check of everything at the end

### Current Programming

1.  Dependencies, typing and testing

    1.  if using R, document R dependencies such as package versions
        neatly (avoid `renv`)

    2.  include basic test code by instantiating class and/or other
        simple methods

    3.  add mypy as an explicit part of testing the source code

    4.  replace Union + None types with Optional type for conciseness

    5.  look into cases where List was replaced by Sequential and how
        this can be changed or understood to keep consistency (ie. keep
        everything to List with overloads)

2.  Documentation and clean-code

    1.  find better naming for mimic/oracle models which is based on
        research terminology -\> right now mix of neural and regex is
        being used; it would be good to have something more firm

    2.  GPU/CPU runs not always reproducible depending on
        multi-threading, see:
        <https://pytorch.org/docs/stable/notes/randomness.html#reproducibility>

    3.  reduce source code lines, chunking and comments -\> pretty sort
        python code and function/class orders perhaps by length

    4.  add a comment above each code chunk which explains inner
        mechanisms better

    5.  update metadata eg. with comprehensive python/shell help
        scripts, comments describing functionality and readme
        descriptions for git hooks

    6.  add information on best model downloads and preparation -\> add
        these to Google Drive later on

    7.  add pydocstrings to all functions and improve argparse
        documentation

    8.  provide description of data structures (eg. data, labels)
        required for training processes and lowercasing

    9.  update/remove git hooks depending on which features are finally
        used, eg. remove pre-push hook

    10. test download and all other scripts to ensure they work

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
