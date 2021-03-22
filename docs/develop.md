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

    **DEADLINE:** *\<2021-03-24 Wed\>*

    1.  Regex proxy

        1.  Describe removal of lower model and upper model

            1.  link this to explanations by simplification and regular
                expressions, as well as conversion from WFAs to FAs/REs
                -\> should be introduced in the background concepts

            2.  describe importance of TauSTE layer for explainability
                and how discrete layers help

            3.  link back to background concepts when we discuss
                bringing neural and regex model as close to each other
                as possible

        2.  Add pseudocode for regex proxy model construction and
            compression

        3.  Add pseudocode for regex proxy modl forward pass

            1.  think about whether SoPa++ forward pseudocode must be
                modified with text-level instructions, this might be too
                complicated with too many details for now -\> or maybe
                change pseudocode to traversal code

            2.  mention that pseudocode everywhere has been simplified
                and they should refer to source code for full
                implementation details

            3.  consider changing figure softmax to soft+argmax

            4.  report time complexities of the simplification process
                as well as compression where possible

            5.  make sure to state that the regex\'s represent substring
                matches to the borders are generic -\> or otherwise
                explicitly add these to the visualizations -\> make this
                clear both here and for the results to questions three

            6.  alternatively consider different notation to imply this
                is a regex match with arbitrary borders

            7.  show TikZ visualization of regex computational graph and
                tweak colors/labels to make explanations easier if need
                be

            8.  replace FAs in visualization with regular expression
                matrix where possible

            9.  talk about regex proxy possibly being a transparent
                model and motivate arguments for/against this, say this
                is only theoretical but we come back to this in the
                discussion segment

        4.  Add table with key differences from SoPa model

            1.  recycle this table later and add it to the SoPa model
                section

            2.  talk about SoPa++ still being a black-box model, compare
                all features

    2.  SoPa++ training setup

        1.  talk about GloVe embeddings, lowercasing and other important
            things

        2.  talk about upsampling data set during training

        3.  mention about PyTorch for parallel computation and autograd
            for backward pass computation

        4.  mention about NLL loss and log softmax for numerical
            stability

        5.  add sufficient information on grid training and
            hyperparameter setup

    3.  Performance evaluation

        1.  explain usage of test set

        2.  explain how we evaluate performance using accuracy on both
            SoPa++ and its regex proxy

        3.  mention checking if the performance falls in the competitie
            range

    4.  Explanations by simplification evaluation

        1.  mention we are trying to evaluate how close the models are,
            and we delegate the quality of explainability based on a
            target audience to future work since this would be
            out-of-scope for us

        2.  mention in general that the target audience for this method
            is still experts

        3.  explain usage of test set in this case

        4.  motivate all the different metrics such as softmax
            difference norm, binary misalignment rate and also
            performance scores

        5.  mention variation of the tau threshold since this would
            limit how often activation occurs, which we hypothesize
            could have an effect in bringing both models closer together

    5.  Insightful explanations insight

        1.  mention how we analyze the output neurons with legitimacy
            given that it is now a linear layer to find how STE neurons
            distribute importances

        2.  in connection with this, we can analyze which textual
            regular expressions are responsible for each STE neuron
            activation -\> which gives us an insight into which regular
            expressions are important for classifications

        3.  we can analyze these to see how generalized these are or
            whether there could be strong inductive bias

2.  Results

    **DEADLINE:** *\<2021-03-28 Sun\>*

    1.  **Important:** everything shown in the \"Results\" section
        should have been well-motivated and should answer all three
        research questions

    2.  1\. Evaluation of test accuracy scores of SoPa++ and regex proxy
        models

        1.  show visualization of training process for posterity

        2.  provide a table of summarized results w.r.t. random seed
            deviations

            1.  modify visualize scripts to aggregate and print summary
                stats in script to re-use later in paper with means and
                standard deviations across random seeds

            2.  report parameter counts in scores as well

        3.  use the plus-minus sign to give the score ranges for all
            model types

        4.  make statement on the competitiveness of these results

    3.  2\. Evaluation of explanations by simplification

        1.  add table with information on performances and distance
            metrics averaged over random seeds with plus-minus deviation
            format

        2.  show plot of how these trends work and that increasing the
            tau threshold brings models closer together

        3.  make statement that explanations by simplifications appears
            to effective on the unseen evaluation set with similar
            scores; also a relationship can be observed with respect to
            the tau threshold

    4.  3\. Interesting and insightful explanations on FMTOD

        1.  show plot of neuron importance distributions -\> consider
            removing inter-neuron importance with alpha levels since
            this might be irrelevant

        2.  show relevant plots of regular expressions with neurons to
            see what kinds of textual patterns are imporant for the
            SoPa++ model and its regex proxy

            1.  figures must be manually put together later directly in
                latex

            2.  label the states consistently with \"q\" instead of only
                numbers

            3.  legend in neurons can be shifted below, perhaps to have
                just one copy for all neurons shown

            4.  tweak relative importances between neurons as well, if
                this is of use otherwise drop it -\> need to add color
                to legend or otherwise fixed size with color gradient

            5.  consider adding subscripts to states

3.  Discussion

    1.  Performance

        1.  mention issue of being unsure whether other studies removed
            duplicates -\> but in our case using the same test set seems
            to improve evaluation performance

    2.  Explainability

        1.  discuss how transparent the regex proxy really is given how
            many regular expressions are picked up -\> link to how
            rules-based models in Arrieta et al 2020 paper can also
            become black-boxes -\> provide numbers of regular
            expressions that get captured and stored

        2.  add segment on how useful this might be to a target audience
            based on the three criteria, but acknowledge that this would
            need to be consulted with a target audience

            1.  describe how a basic human evaluation of explainability
                could be done

            2.  mention visualization resources needed for this and it
                would have to be done with a much simpler and smaller
                model

    3.  Interesting and insightful observations

        1.  inductive biases might be possible to find by looking into
            regular expressions

        2.  mention distributed representations in neurons where there
            is really no clear neuron responsible for one-thing -\>
            which is an impediment to explainability since attribution
            and causal links are difficult to identify

    4.  Other discussion points

        1.  Discuss relationship between tau threshold and the
            softmax/binary distances

        2.  Expound on trade-off between performance and transparency by
            looking at differently sized models -\> and then also
            looking at other studies which used BERTesque models

4.  Conclusions

    1.  Summarize everything in manuscript

    2.  Address research questions

5.  Further work

    1.  Modeling

        1.  use multiple-threads for extracting regular expressions, or
            store them in a database with indexing for faster regex
            lookups

        2.  extend to a finite-state transducer for seq2seq tasks

    2.  Explainability generalization/evaluation

        1.  can map linear to decision tree to get clearer picture of
            possibilities

        2.  use nearest-neighbours to expand adjacent tokens

        3.  semantic clustering of common patterns for increased
            generalization

        4.  this is subjective and a survey from the target audience
            would be good to have -\> would require an interactive
            interface where we visualize explanations

        5.  UNK token handling workflow for regex model where UNK has to
            representation other than indirect wildcards

        6.  human intervention inside regex model to monitor/improve
            performance

    3.  Analysis

        1.  extension to more NLU data sets such as SNIPS, ATIS

        2.  analyzing whether patterns can help discover possible
            adversarial patterns or inductive biases

        3.  for the target audience of end-users -\> how can a user make
            use of the regex model

        4.  visualize examples/classes where regex and neural model
            align and misalign, eg. with a confusion matrix

6.  Post-paper iteration/formatting

    **DEADLINE:** *\<2021-03-31 Wed\>*

    1.  Introduction

        1.  abstract and introduction should already mention key
            findings

        2.  update motivations from Arrieta et al. 2020 \"What for\"
            section

        3.  add links to chapters in thesis structure

    2.  Background concepts

        1.  think more about how to improve (W)FA definitions

        2.  consider adding definitions for main transitions if they are
            useful

        3.  pad definition environments with more explanatory text for
            flow

        4.  add more information on page numbers and sections in all
            Arrieta et al. citations so these seem more differentiated

        5.  explain vanilla SoPa in a more prose format using a table to
            explain important features -\> this table can then be
            compared directly with new SoPa++ features

        6.  mention early on that quantized NNs are useful for
            low-precision computing, but we use it for other reasons
            later on

    3.  Methodologies

        1.  add Github link to repo as a footnote when introducing this
            chapter

        2.  consider adding background section on NLU overall and tasks
            available

        3.  mention the purpose of the intent detection task briefly on
            a grander-scheme-of-things

        4.  mention briefly the purpose of the slot filling task

    4.  Terminologies and abbreviations

        1.  sort out all abbreviations and standardize formatting in
            terms of where they are first declared -\> perhaps add a
            page number on list of abbreviations to indicate first
            declaration

        2.  consider adding abbreviations directly into definitions

        3.  consider respelling \"preprocessing\" as \"pre-processing\"
            if necessary, fine-tune their usage

        4.  fine-tune antecedent/proxy terminology and synchronize
            everywhere

        5.  fine-tune usage of patterns vs. substrings vs. WFAs

        6.  fine-tune WFA to mean either automata or automaton, make
            plural abbreviation WFAs clear

        7.  fine-tune the slot-filling terminology

        8.  fine-tune terminology between intent detection and intent
            classification

        9.  fine-tune terminology between data set and partition

        10. fine-tune token length, sentence length and utterance length

    5.  Bibliography

        1.  improve capitalization with braces in bibtex file

        2.  find alternative journal/conference citations for current
            arxiv papers

    6.  Manuscript admin

        1.  Text-related

            1.  ensure that areas between chapters-sections and/or
                sections-subsections are filled with explanatory text to
                provide a narrative -\> use links to/from individual
                sections/chapters to string everything together -\> no
                area between title and next sub-title or environment
                should be empty -\> an example is adding text before WFA
                definitions

            2.  replace all epsilon words by the symbol where possible

            3.  add remaining features by referring to master template
                such as abstract (short summarized introduction), list
                of tables/figures/abbreviations, appendices, etc; see
                master document for examples

            4.  change to two sided format before printing, as this
                works well for binding/printing

            5.  EITHER quote + indent sentences directly taken from
                other studies with page and section OR paraphrase them
                and leave them in a definition environment

            6.  check that all borrowed figures have an explicit
                attribution such as \"taken from paper et al (year)\"

            7.  perform spell-check of all text

            8.  remove red link color in table of contents

            9.  always mention \"figure taken from study (year)\" when
                using external figures

        2.  UP-related

            1.  20-90 pages thesis length -\> well-motivated yet
                succinct

            2.  date on bottom of manuscript should be date of
                submission before mailing to Potsdam

            3.  add student registration details to paper such as
                matriculation number and other details

            4.  take note of all other submission criteria such as
                statement of originality, German abstract, digital copy
                and others, see:
                <https://www.uni-potsdam.de/en/studium/studying/organizing-your-exams/final-thesis>

### Programming

1.  Dependencies, typing and testing

    1.  if using R, document R dependencies with `sessionInfo()`

    2.  look into cases where List was replaced by Sequential and how
        this can be changed or understood to keep consistency (ie. keep
        everything to List with overloads)

2.  Documentation and clean-code

    1.  update readme and usages with finalized antecedent and proxy
        terminologies

    2.  upadte readme and usages with finalized STE/output neurons
        terminologies

    3.  find attributable naming standards for PDFs produced with
        timestamp, perhaps dump a json file

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

    11. perform formatting on latex code

    12. ensure all label names and figure names are consitent

    13. perform spell-check on readme

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
