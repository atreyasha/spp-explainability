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

1.  Methodologies

    **DEADLINE:** *\<2021-03-21 Sun\>*

    1.  Model

        1.  update the WFSA definitions to signify wildcard transitions
            and mention why this is found here and not in the background
            concepts since we could not find literature which defined it
            in a similar way

        2.  add pseudocode for various segments -\> would help to cement
            certain concepts

        3.  look out for misconception between tau and transition matrix
            symbol -\> perhaps change transition matrix symbol to
            something for distinct

        4.  try to find literature-based justification for wildcard
            transition -\> if not use omega symbol to differentiate from
            the Kleene star symbol -\> use UTF-8 symbol for graphviz
            plots

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

        11. add Github link to repo as a footnote

        12. **new:** produce manual neural computational graph using
            tikz, building from what was made earlier

    2.  Quantization/Binarization

        1.  explain how and why we modified STE to TauSTE

        2.  how does binarizing help with explainability, justify
            requirement for it in both training and testing

        3.  add visualization of TauSTE function

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

        10. **new:** produce manual regex computational graph using
            tikz, building from what was made earlier

    4.  Training/Evaluation/Explainability-evaluation

        1.  talk about upsampling data set during training

        2.  provide extensive details of training setup

        3.  provide extensive details of evaluating neural/regex models

        4.  provide extensive details of evaluating explanations by
            simplification \"metric\" of neural-regex models -\> which
            should make results clearer

        5.  **important:** everything shown in the \"Results\" section
            should be motivated or introduced here, focus harder on
            methodologies so that everything else is very easy to
            explain

2.  Results

    **DEADLINE:** *\<2021-03-28 Sun\>*

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

        2.  go into details on how effective compression algorithm was
            in terms of reducing the memory and number of regex\'s -\>
            can tabulate all of these

    3.  Explain discussion figures very clearly and show the relevance
        to the third research question

    4.  Visualizations

        1.  show visualization of training performance timelines, think
            about how to keep most important information

        2.  show visualization of tau threshold vs. performance vs.
            softmax/binary distances with error bars for random seed
            iterations

        3.  show visualizations of important patterns in a regex model
            -\> something which is small and fits well into a page

            1.  figures must be manually put together later directly in
                latex

            2.  consider removing double-circle for start state, since
                this usually denotes the accepting state

        4.  show TikZ visualization of each binary neuron\'s relative
            importance for classes -\> would be interesting to see how
            saturated these are

3.  Discussion

    1.  Expound on trade-off between performance and transparency by
        looking at differently sized models

    2.  Discuss relationship between tau threshold and the
        softmax/binary distances

    3.  Think about why larger regex models tend to show more
        misalignment from neural counterparts compared to smaller models

    4.  Can talk about neurons responsible for certain decisions, as
        well as distributed representations in neurons where there is
        really no clear neuron responsible for one-thing -\> which is an
        impediment to explainability

    5.  If possible, add a basic human evaluation of explainability
        otherwise leave it to future work

4.  Conclusions

    1.  Summarize everything in manuscript

    2.  Address research questions

5.  Further work

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

6.  Post-paper iteration/formatting

    **DEADLINE:** *\<2021-03-31 Wed\>*

    1.  Paper length

        1.  20-90 pages thesis length -\> try to keep ideas
            well-motivated yet succinct

    2.  Points to address towards end

        1.  Introduction

            1.  abstract and introduction should already mention
                results, and should not leave this to conclusions

            2.  fine-tune introduction with new details from other
                chapters

            3.  update motivations from Arrieta et al. 2020 \"What for\"
                section

            4.  add C-like reference to explain what SoPa++ means like
                in i++

            5.  add links to chapters in thesis structure, improve
                formatting

        2.  Background concepts

            1.  add more background information on linear-chain WFSAs,
                FSAs, regular expressions and conversion processes

            2.  explain vanilla SoPa more clearly to motivate everything
                else -\> perhaps need more information on FSAs with
                starting and accepting states

            3.  EITHER quote + indent sentences directly taken from
                other studies (cite pages and paragraphs) OR paraphrase
                them and leave them in a definition environment

            4.  consider citing pages and sections for Arrieta article
                in all cases since there is a lot of information -\>
                might make citations of same article less redundant
                since there is accompanying information to diversify
                things

            5.  be very clear on what is directly taken from another
                study versus what is paraphrased

            6.  think about providing an additional definition for
                \"understandability\"

            7.  consider quoting all definitions to further imply that
                they are exactly taken from other studies

            8.  add a Kleene-star operator mention to remark 9.4

            9.  include a section on risks on large NLP models and why
                explainability is necessary with different study

            10. if possible, try to reduce references to Arrieta et al.
                2020 to reduce perceived over-dependence

            11. revisit sopa explainability evaluation with three
                guidelines to check if it makes sense after having
                evaluated sopa++ with the same guidelines

            12. look into antecedent/proxy names and if these can be
                improved

            13. return to this chapter to add/remove content based on
                requirements of later chapters

        3.  Bibliography

            1.  look for journal/conference alternative citations for
                current papers

            2.  improve capitalization with braces in bibtex file

            3.  if possible, try to find non-arxiv citations for papers
                -\> look for alternative citations in ACL or other
                conferences instead of arxiv papers

            4.  remove red link color in table of contents

            5.  fine-tune citation color to be consistent with other
                colors

        4.  Methodologies

            1.  think about adding new table with percentage of each
                data class in FMTOD -\> would highlight the imbalance a
                bit better

            2.  consider respelling \"preprocessing\" as
                \"pre-processing\" if necessary

            3.  consider redoing FMTOD table with relative frequencies

            4.  consider reporting token length statistics in a table or
                with a figure

            5.  use same terminology between intent detection and intent
                classification

            6.  think of how to better present accuracies from other
                studies for FMTOD, perhaps with a table in the results
                section

        5.  Appendix

            1.  consider adding separate table in appendix for examples
                of FMTOD data instances by class

            2.  add more information to appendices and link them in the
                appropriate text portions

        6.  Manuscript admin

            1.  read manuscript and ensure there is an easily followable
                narrative for someone who is a non-expert -\> can be
                done by adding more text before or around definition
                environments in order to lead the reader into each
                concept smoothly -\> this can be done post-paper since
                it requires all the information to be present in the
                paper first

            2.  definition/remark structure might need to be revised to
                something more narrative-suited, or simply add
                sufficient lines before and after to keep the narrative
                flowing

            3.  ensure that areas between chapters-sections or
                sections-subsections are filled with some explanatory
                text to give the sense of a flowing narrative -\> use
                links to individual sections in each chapter to describe
                what these describe -\> this will help string everything
                together including for linking background concepts to
                methodologies

            4.  add titles to all figures in the manuscript

            5.  always mention \"figure taken from study (year)\" when
                using external figures

            6.  fine tune WFSA to mean either automata or automaton,
                make plural abbreviation clear as well

            7.  add links to different sections later on once structure
                and content is clear -\> need to read through to catch
                all parts which need links

            8.  sort out all abbreviations and standardize formatting in
                terms of where they are first declared

            9.  change to two sided format before printing, as this
                works well for binding/printing

            10. add Uni-Potsdam originality declaration, or modify
                current one to fit

            11. add student registration details to paper such as
                matriculation number and other details

            12. add remaining features by referring to master template
                such as abstract (short summarized introduction), list
                of tables/figures/abbreviations, appendices, and all
                others

            13. date on bottom of manuscript should be date of
                submission before mailing to Potsdam

            14. take note of all other submission criteria such as
                statement of originality, printing, German abstract,
                digital copy and others, see:
                <https://www.uni-potsdam.de/en/studium/studying/organizing-your-exams/final-thesis>

            15. perform spell-check of everything at the end

### Programming

1.  Dependencies, typing and testing

    1.  if using R, document R dependencies with `sessionInfo()`

    2.  look into cases where List was replaced by Sequential and how
        this can be changed or understood to keep consistency (ie. keep
        everything to List with overloads)

2.  Documentation and clean-code

    1.  if necessary, apply further script renaming using antecedent and
        proxy terminologies -\> update readme and usages

    2.  fix terminology of STE/output neurons consistently after paper

    3.  find a better way of naming visualization pdfs to attribute to
        specific model and make this unique -\> perhaps via timestamp

    4.  GPU/CPU runs not always reproducible depending on
        multi-threading, see:
        <https://pytorch.org/docs/stable/notes/randomness.html#reproducibility>

    5.  add a comment above each code chunk which explains inner
        mechanisms better

    6.  update metadata eg. with comprehensive python/shell help
        scripts, comments describing functionality and readme
        descriptions for git hooks

    7.  add pydocstrings to all functions and improve argparse
        documentation

    8.  add information on best model downloads and preparation -\> add
        these to Google Drive later on

    9.  test out all shell-scripts and python code to make sure
        everything works the same after major renamings

    10. test download and all other scripts to ensure they work

    11. perform spell-check on readme

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
