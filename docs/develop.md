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

    1.  Iteration 1 (content and readability)

        1.  **TODO** Introduction

            1.  consider removing quote and instead describe issues in
                more detail

            2.  add links to chapters in thesis structure

            3.  add meta-text for padding and linking

            4.  improve overall readability

        2.  Background concepts

            1.  add more information on page numbers and sections in all
                Arrieta et al. citations so these seem more
                differentiated

            2.  slightly reword performance-interpretability trade-off
                segment

            3.  mention early on that quantized NNs are useful for
                low-precision computing, but we use it for other reasons
                later on

            4.  add how nfa can be extracted from wfa for non semiring
                zero path scores

            5.  mention how or why SoPa falls into RNN and CNN
                categories

            6.  mention target audience of SoPa explainability

            7.  change SoPa post-hoc explainability methods to mention
                quality of them

            8.  add meta-text for padding and linking

            9.  improve overall readability

        3.  Methodologies

            1.  mention briefly the purpose of the slot filling task

            2.  consider adding background section on NLU overall and
                tasks available

            3.  consider using a better term for FMTOD English language
                intent detection

            4.  mention the purpose of the intent detection task briefly
                on a grander-scheme-of-things

            5.  describe meaning of SoPa++ in C-symbology

            6.  explain what omega transition means in a FA vs. in a
                regex

            7.  add a short segment to mention backward passes since
                this was not mentioned

            8.  make the RE lookup layer algorithm more concise

            9.  mention RE lookup layer can be attributed to TauSTE
                neurons

            10. explain more what regex lookup layer does mention that
                it memorises REs which lead to activations which can
                help in discussion segment

            11. change regex to RE in computational graphs to be
                consistent with paper

            12. emphasize that SoPa++ and RE proxy models come in pairs

            13. add mention that SoPa++ is fast while regex is slow

            14. mention target audience of SoPa++ explainability

            15. add statement on quality of explanations based on three
                criteria in XAI -\> link this to RQ2 and find an
                appropriate place to add it -\> or perhaps put this in
                SoPa vs. SoPa++ section with some motivation

            16. improve table of similarities and differences

            17. change light, medium, heavy to small, medium, large

            18. add detailed information on what small medium large
                means in terms of model specifics

            19. improve notation for metrics as well as their
                definitions with or without overlines

            20. perhaps provide example of how softmax is conducted over
                weights to make this clearer

            21. we can analyze these to see how generalized these are or
                whether there could be strong inductive bias

            22. add meta-text for padding and linking

            23. improve overall readability

        4.  Results

            1.  add small, medium and heavy to training and evaluation
                charts on facet titles

            2.  re-do bolding in tables to be more intuitive and improve
                descriptions in captions

            3.  mention that regex was converted to NFA for visualizatio
                -\> link back to background conceptsn purposes or change
                terminology here

            4.  add a mention of which model with exact specifications
                was used for the neuron and regex sampling for clarity
                -\> explain why we used the light model to keep things
                tractable

            5.  add neuron subscript in captions and add some
                description as well

            6.  add meta-text for padding and linking

            7.  improve overall readability

        5.  Discussion

            1.  expound on trade-off between performance and
                transparency by looking at differently sized models -\>
                and then also looking at other studies which used
                BERTesque models -\> link back to background concepts

            2.  discuss relationship between tau threshold and the
                softmax/binary distances

            3.  use discussions section to bring about more nuanced
                points on results -\> perhaps related to inductive
                biases in the RE lookup layer

            4.  provide examples of RE similarities and types

            5.  add meta-text for padding and linking

            6.  improve overall readability

        6.  Further work

            1.  for the target audience of end-users -\> how can a user
                make use of the regex model

            2.  visualize examples/classes where regex and neural model
                align and misalign, eg. with a confusion matrix

            3.  add meta-text for padding and linking

            4.  improve overall readability

    2.  Iteration 2 (consistency and formatting)

        1.  Terminologies and abbreviations

            1.  consider adding abbreviations directly into definitions

            2.  fine-tune preprocessing vs. pre-processing

            3.  fine-tune antecedent/proxy terminology and synchronize
                everywhere

            4.  fine-tune usage of string length and document length
                while consuming

            5.  fine-tune usage of document vs. input utterance

            6.  fine-tune usage of consume a token and not consume a
                symbol

            7.  fine-tune usage of patterns vs. substrings vs. WFAs

            8.  fine-tune usage of TauSTE neurons and output neurons -\>
                especially for RQ3

            9.  fine-tune string consumption vs. string matching

            10. fine-tune WFA to mean either automata or automaton, make
                plural abbreviation WFAs clear

            11. fine-tune usage of FA vs. NFA and make these consistent
                with abbreviations versus full

            12. fine-tune the slot-filling terminology

            13. fine-tune terminology between intent detection and
                intent classification

            14. fine-tune terminology between data set and partition

            15. fine-tune token length, sentence length and utterance
                length

            16. fine-tune regex vs. RE

            17. fine-tune usage of the RE lookup layer

            18. fine-tune linear vs. linear regression layer

            19. fine-tune FMTOD data set vs. intent detection task

            20. ensure linear-chain and strict linear-chain are added
                before WFAs

            21. define GPU somewhere and add to abbreviation

            22. replace all epsilon words by the symbol where possible

            23. make number of decimal places consistent wherever they
                are used such as in tables with tau

            24. sort out all abbreviations and standardize formatting in
                terms of where they are first declared -\> perhaps add a
                page number on list of abbreviations to indicate first
                declaration

        2.  Formatting

            1.  ensure that areas between chapters-sections and/or
                sections-subsections are filled with explanatory text to
                provide a narrative -\> use links to/from individual
                sections/chapters to string everything together -\> no
                area between title and next sub-title or environment
                should be empty -\> an example is adding text before WFA
                definitions

            2.  EITHER quote + indent sentences directly taken from
                other studies with page and section OR paraphrase them
                and leave them in a definition environment

            3.  check that all borrowed figures have an explicit
                attribution such as \"taken from paper et al (year)\"

            4.  change red link color in table of contents and modify
                color of URLs

        3.  Formalities

            1.  20-90 pages thesis length -\> well-motivated yet
                succinct

            2.  abstract and introduction should already mention key
                findings -\> synthesize these

            3.  add remaining features by referring to master template
                such as abstract (short summarized introduction), list
                of tables/figures/abbreviations, appendices, etc; see
                master document for examples

            4.  date on bottom of manuscript should be date of
                submission before mailing to Potsdam

            5.  add student registration details to paper such as
                matriculation number and other details

            6.  update title page date to current submission date

            7.  take note of all other submission criteria such as
                statement of originality, German abstract, digital copy
                and others, see:
                <https://www.uni-potsdam.de/en/studium/studying/organizing-your-exams/final-thesis>

        4.  Bibliography

            1.  improve capitalization with braces in bibtex file

            2.  find alternative journal/conference citations for
                current arxiv papers

        5.  Final steps

            1.  perform spell-check of all text

            2.  re-read paper for flow and sensibility

### Programming

1.  Clean-code and documentation

    1.  Source-code

        1.  rename RegexSoftPatternClassifier class to RegexProxy if
            possible without breakages

        2.  change light, medium and heavy to small, medium and large in
            all scripts, filenames and log variables consistently

        3.  remove all neural sopa from readme and everywhere else since
            spp already implies neural -\> consider changing parser
            arguments as well if possible for consistency from
            --neural-model-checkpoint to --spp-model-checkpoint

        4.  add pydocstrings to all functions and improve argparse
            documentation

        5.  add a comment above each code chunk which explains inner
            mechanisms better

    2.  Readme

        1.  update readme and usages with finalized antecedent and proxy
            terminologies

        2.  upadte readme and usages with finalized STE/output neurons
            terminologies

        3.  add information on best model downloads and preparation -\>
            add these to Google Drive later on

        4.  update metadata eg. with comprehensive python/shell help
            scripts, comments describing functionality and readme
            descriptions for git hooks

        5.  test out all shell-scripts and python code to make sure
            everything works the same after major renamings

        6.  test download and all other scripts to ensure they work

        7.  perform spell-check on readme

2.  Dependencies, typing and testing

    1.  if using R, document R dependencies with `sessionInfo()`

    2.  look into cases where List was replaced by Sequential and how
        this can be changed or understood to keep consistency (ie. keep
        everything to List with overloads)

    3.  GPU/CPU runs not always reproducible depending on
        multi-threading, see:
        <https://pytorch.org/docs/stable/notes/randomness.html#reproducibility>

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
