## Table of Contents
-   [Current tasks](#current-tasks)
    -   [Manuscript](#manuscript)
    -   [Programming](#programming)
-   [Future tasks](#future-tasks)
    -   [Programming](#programming-1)
-   [Notes](#notes)
    -   [Manuscript](#manuscript-1)
    -   [Admin](#admin)
-   [Completed](#completed)
-   [Legacy](#legacy)
    -   [Interpretable RNN
        architectures](#interpretable-rnn-architectures)
    -   [Interpretable surrogate
        extraction](#interpretable-surrogate-extraction)
    -   [Neuro-symbolic paradigms](#neuro-symbolic-paradigms)
    -   [Neural decision trees](#neural-decision-trees)
    -   [Inductive logic on NLP search
        spaces](#inductive-logic-on-nlp-search-spaces)

## Current tasks

### Manuscript

1.  Structured content

    1.  Introduction

        1.  add research questions to introduction

        2.  make \"highly performant\" more specific

        3.  fix absolute terms such as \"automated reasoning\", or quote
            directly from paper

        4.  add more information on what competitive performance means
            (eg. within few F_1 points)

        5.  re-consider reference to Transformers for dynamic sub-word
            level word-embeddings

        6.  make claim that SoPa++ explainability has a different
            explainability taxonomy from that of vanilla SoPa, but
            don\'t claim it is decisively better

    2.  Background concepts

        1.  ANN\'s historical literature -\> describe how ANNs
            approximate symbolic representations

        2.  Explainability

            1.  qualify what does it mean to be explainable and how to
                define this

            2.  mention that explainability focuses on exposing the
                model\'s logic and not on necessarily creating
                rationality in the model

            3.  make hierarchy for local vs. global explainability -\>
                can provide hints of differences here

            4.  explainability is only relevant if the oracle and mimic
                models both **perform competitively and have similar
                confusion matrix profiles**

            5.  provide some metrics or tangible task-based insights to
                show how new explainability works

            6.  clarify meaning and concept of \"occlusion\" as
                leave-one-out perturbation analysis

            7.  improve arbitrary vs. contrained oracle phrasing -\>
                perhaps black-box vs. white-box but more specific

        3.  FSA/WFSAs -\> input theoretical CS, mathematics background
            to describe these

            1.  use more appropriate and generalized semiring
                terminology from Peng et al. 2019 -\> more generalized
                compared to SoPa paper

            2.  semirings, abstract algebra and how they are used for
                finite-state machines in Forward and Viterbi algorithms
                -\> go deeper into this to get some background

    3.  Methodologies

        1.  Quantization/Binarization

            1.  change STE citation to 2013 paper which is more
                fundamental, use different name depending on context:
                <https://arxiv.org/abs/1308.3432>

            2.  cite and explain straight-through estimation (STE), and
                what benefits this is supposed to bring

            3.  how does binarizing help with explainability, justify
                requirement for it in both training and testing

        2.  Visualizations

            1.  add visualization of in-depth computational graph in
                paper for clarity -\> this can be automated

            2.  produce manual computational graph using tikz, building
                from what was made earlier

    4.  Results

        1.  Visualizations

            1.  produce visualization of training performance using
                python frameworks rather than R

            2.  produce visualizations of regex ensembles which would be
                interesting, and also pattern activations

        2.  Compare results to that of other paper(s)

    5.  Discussion

        1.  addresss interesting observations and their implications

        2.  expound on trade-off between performance and explainability
            and process of mimic extraction

        3.  how can a user make use of the mimic model and what benefits
            are there for the user in terms of security/safety/etc -\>
            or could add this to further work

        4.  mention possibilities of fixing errors on the test set with
            general changes to the regex model which are much easier to
            do compared to the tensor model

        5.  show cases where we could avoid adversarial cases using the
            insight of the regex model

        6.  run explainability and evaluation comparisons on models
            trained with different and increasing tau values to
            experiment how this affects performance/explainability -\>
            hypothesis is that this might bring regex and neural
            explainabilities closer together -\> if not then think of
            issues with this process -\> would be very interesting to
            explore this relationship on both smmall and large models
            -\> binaries are saturated so maybe tau might help with this

    6.  Conclusions

        1.  summarize everything in manuscript and address research
            questions

    7.  Further work

        1.  look into future programming tasks and add their content
            here -\> there are many points to consider

        2.  mention about how it is not easy to evaluate the \"quality
            of explainability\" and to say one explainability is better
            than the other -\> aside from a theoretical perspective

        3.  perhaps suggest how this explainability could be evaluated
            via conducting a survey and getting rating from people

        4.  porting this technique to a transformer where possible -\>
            but mention limitations of everything being context
            dependent

        5.  extension/recommendations -\> transducer for seq2seq tasks

        6.  can map linear to decision tree to get clearer picture of
            possibilities

        7.  would be interesting to deterministically export which
            patterns for sure lead to which class, could help to
            identify adversarial samples via tinkering

        8.  database with indexing could help improve regex lookup speed
            -\> further work

        9.  think about why larger regex model tends to not be as close
            to neural as a smaller regex model -\> can also be
            investigated with different models

2.  Formatting

    1.  Paper length

        1.  20-90 pages thesis length -\> try to keep ideas
            well-motivated yet succinct

    2.  Points to address towards end

        1.  improve capitalization with braces in bibtex file

        2.  remove red link color in table of contents

        3.  fine-tune citation color to be consistent with other colors

        4.  change to two sided format before printing, as this works
            well for binding/printing

        5.  add Uni-Potsdam originality declaration, or modify current
            one to fit

        6.  add remaining features by referring to master template such
            as abstract, list of tables/figures/abbreviations,
            appendices, and all others

        7.  perform spell-check of everything at the end

### Programming

1.  Visualization and summary-statistics

    1.  Cross-model comparisons

        1.  visualize examples where regex and neural model align and
            misalign, eg. with a confusion matrix

        2.  compute statistics with random-seed deviations over
            inter-model comparisons such as average distance,
            misalignment, activation frequency and other useful metrics
            that can elucidate on-the-ground processes

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

        1.  add visualization of computational graph function directly
            to sopa torch model

2.  Dependencies, typing and testing

    1.  if using R, document R dependencies such as package versions
        neatly (avoid `renv`)

    2.  include basic test code by instantiating class and/or other
        simple methods

    3.  add mypy as an explicit part of testing the source code

    4.  replace Union + None types with Optional type for conciseness

    5.  look into cases where List was replaced by Sequential and how
        this can be changed or understood to keep consistency (ie. keep
        everything to List with overloads)

3.  Documentation and clean-code

    1.  find better naming for mimic/oracle models which is based on
        research terminology -\> right now mix of neural and regex is
        being used; it would be good to have something more firm

    2.  GPU/CPU runs not always reproducible depending on
        multi-threading, see:
        <https://pytorch.org/docs/stable/notes/randomness.html#reproducibility>

    3.  reduce source code lines, chunking and comments -\> pretty sort
        python code and function/class orders perhaps by length

    4.  add a comment to each code chunk which explains inner mechanisms
        better

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

## Future tasks

### Programming

1.  Model diversity

    1.  run above explainability and comparison workflow on larger
        models after efficiency improvements

2.  Modelling improvements

    1.  check if packed sequences could be incoporated into model

        1.  might increase efficiency related to batch latency

    2.  find single-threaded ways to speed up regular expression
        searches -\> bottleneck appears to be search method

        1.  multiprocessing with specific chunksize seems to have some
            effect

        2.  might need to have a very large batch size to see any
            improvements with multiprocessing

    3.  consider using finditer for regex lookup with trace, since we
        should return all matches

        1.  make activating text unique in case we return multiple texts
            and not one -\> but then won\'t correspond to activating
            regexes

        2.  might not make a huge difference since we use short
            sentences

        3.  might be better for speed reasons to leave it as a search
            method

3.  Model features

    1.  add check to ensure start, end and pad tokens don\'t occur
        adversarially inside sequence -\> `nltk.word_tokenize` already
        breaks these up

    2.  add predict function for both mimic and oracle model which does
        not need extra data to be loaded -\> can also accept stdin as
        unix pipe

4.  Explore activation generalization methods

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

    2.  To what extent does SoPa++ contribute to improved explainability
        by simplification?

    3.  What interesting and relevant explanations does SoPa++ provide
        on NLU task(s)?

2.  Timeline

    1.  ~~Initial thesis document: **15.09.2020**~~

    2.  ~~Topic proposal draft: **06.11.2020**~~

    3.  ~~Topic proposal final: **15.11.2020**~~

    4.  ~~Topic registration: **01.02.2021**~~

    5.  Projected manuscript completion: **31.03.2021**

    6.  Offical manuscript submission: **10.04.2021**

## Completed

**DONE** modify normalizer to ignore calculation of all
infinities via minimal value replacement

**CLOSED:** *\[2021-01-27 Wed 19:19\]*

**DONE** remove both epsilon/self-loops -\> use only simple
transitions and hard wild cards

**CLOSED:** *\[2021-01-27 Wed 15:01\]*

**DONE** defaults from paper: semiring -\> max-product,
batch-size -\> 128 (cpu), epochs -\> 200, patience -\> 30, word_dim -\>
300

**CLOSED:** *\[2021-01-02 Sat 14:23\]*

**DONE** reduce circum-padding token count to 1 instead of
length of longest pattern

**CLOSED:** *\[2020-12-31 Thu 13:03\]*

**DONE** test out to see if scheduler works and if its state
gets incremented -\> need to train single model for long period of time
and analyze state_dict of scheduler to see what has been recorded -\> it
works well when clip threshold is set to zero and patience is observed

**CLOSED:** *\[2020-12-31 Thu 13:01\]*

**DONE** log model metrics with intra/inter-epoch frequency
which can be shared with tqdm for displaying -\> would require some
recoding with modulos -\> how to manage updates with batch vs. epochs
conflict and how to continue training as well, think about whether to
recompute accuracy as well on a batch-basis

**CLOSED:** *\[2020-12-22 Tue 12:22\]*

**DONE** add argparse option of how often to update tqdm
metrics in training -\> should be shared parameter for tensorboard
logging

**CLOSED:** *\[2020-12-22 Tue 12:22\]*

**DONE** make consistent use of `validation` versus `dev`
throughout all source code -\> redo all log messages and also file
naming especially related to inputs, preprocessing and argparse -\> will
require time and effort

**CLOSED:** *\[2020-12-20 Sun 17:49\]*

**DONE** remove `rnn` option from code altogether -\> keep
things simple for now

**CLOSED:** *\[2020-12-19 Sat 02:33\]*

**DONE** change argparse variable names within train script
to reflect parser and make this consistent throughout, including in
other auxiliary scripts

**CLOSED:** *\[2020-12-19 Sat 01:33\]*

**DONE** need to understand `nn.Module` functionality before
anything else -\> investigate whether `fixed_var` function is indeed
necessary or can be removed since `requires_grad` is set to False by
default, but could be some conflict with `nn.Module` default parameter
construction with `requires_grad = True` -\> left intact for now and
appears to work well

**CLOSED:** *\[2020-12-12 Sat 12:28\]*

**DONE** look through `train.py` and make comments on general
processes -\> fix minor issues where present such as variable naming,
formatting etc.

**CLOSED:** *\[2020-12-08 Tue 18:38\]*

**DONE** major code refactoring for main model with
conversion to recent PyTorch (eg. 1.\*) and CUDA versions (eg. 10.\*)

**CLOSED:** *\[2020-12-05 Sat 18:47\]* **DEADLINE:** *\<2020-12-06
Sun\>*

**DONE** add tensorboard to explicit dependencies to view
relevant logs during training

**CLOSED:** *\[2020-12-03 Thu 14:40\]*

**DONE** replace all Variable calls with simple Tensors and
add `requires_grad` argument directly to tensors where this is
necessary: see
<https://stackoverflow.com/questions/57580202/whats-the-purpose-of-torch-autograd-variable>

**CLOSED:** *\[2020-12-02 Wed 21:50\]*

**DONE** UserWarning: Implicit dimension choice for
log_softmax has been deprecated. Change the call to include dim=X as an
argument

**CLOSED:** *\[2020-12-02 Wed 18:57\]*

**DONE** UserWarning: size_average and reduce args will be
deprecated, please use reduction=\'sum\' instead

**CLOSED:** *\[2020-12-02 Wed 18:39\]*

**DONE** make workflow to download Facebook Multilingual Task
Oriented Dataset and pre-process to sopa-ready format -\> text data and
labels with dictionary mapping as to what the labels mean

**CLOSED:** *\[2020-12-01 Tue 20:29\]* **DEADLINE:** *\<2020-12-03
Thu\>*

**DONE** fixed: UserWarning: nn.functional.sigmoid is
deprecated. Use torch.sigmoid instead

**CLOSED:** *\[2020-11-30 Mon 18:16\]*

**DONE** sort CLI arguments into proper groups, sort them
alphabetically for easier reading

**CLOSED:** *\[2020-11-30 Mon 18:07\]*

**DONE** add types to `parser_utils.py` script internals

**CLOSED:** *\[2020-11-30 Mon 18:07\]*

**DONE** separate extras in `soft_patterns.py` into
`utils.py` -\> test out how batch is utilized -\> fix batch issue, then
move on to other steps -\> batch mini-vocab appears to be a hack to
create a meta-vocabulary for indices -\> try to push with this again
another time -\> consider reverting Vocab index/token defaults in case
this was wrong

**CLOSED:** *\[2020-11-30 Mon 18:07\]*

**DONE** appears to be major bug in Batch class, try to
verify if it is indeed a bug and how it can be fixed

**CLOSED:** *\[2020-11-30 Mon 18:07\]*

**DONE** extract all arg parser chunks and place in dedicated
file

**CLOSED:** *\[2020-11-30 Mon 18:07\]*

**DONE** clean preprocessing script for GloVe vectors and
understand inner mechanisms

**CLOSED:** *\[2020-11-28 Sat 17:02\]*

**DONE** find better location to place code from `util.py`

**CLOSED:** *\[2020-11-27 Fri 19:38\]*

**DONE** migrate to soft-patterns-pp and clean from there

**CLOSED:** *\[2020-11-26 Thu 20:11\]*

**DONE** update proposal with comments from supervisors -\>
update same information here

**CLOSED:** *\[2020-11-17 Tue 14:52\]* **DEADLINE:** *\<2020-11-17
Tue\>*

**DONE** write proposal with key research questions -\>
address points directly from step 3 document requirements -\> prepare
some basic accuracy metrics and interpretations from best model

**CLOSED:** *\[2020-11-10 Tue 18:45\]* **DEADLINE:** *\<2020-11-06
Fri\>*

**DONE** analyze pattern log more closely with code on the
side to understand what it means -\> can start writing early when things
start to make sense

**CLOSED:** *\[2020-11-10 Tue 18:44\]* **DEADLINE:** *\<2020-11-05
Thu\>*

**DONE** add large amounts of binary data for testing with
CPU/GPU -\> requires pre-processing

**CLOSED:** *\[2020-11-10 Tue 18:21\]*

**DONE** find re-usable code for running grid search -\>
otherwise construct makeshift quick code

**CLOSED:** *\[2020-11-05 Thu 20:38\]*

**DONE** test SoPa on sample data in repository to ensure it
works out-of-the-box -\> try this on laptop and s3it

**CLOSED:** *\[2020-11-02 Mon 16:40\]*

**DONE** make workflow to reproduce virtual environment
cleanly via poetry

**CLOSED:** *\[2020-11-02 Mon 16:34\]*

**DONE** make workflow to download simple but high-quality
NLU dataset and glove data sets

**CLOSED:** *\[2020-11-01 Sun 20:15\]* **DEADLINE:** *\<2020-11-01
Sun\>*

**DONE** read more into these tasks and find one that has
potential for interpretability -\> likely reduce task to binary case for
easier processing (eg. entailment)

**CLOSED:** *\[2020-10-28 Wed 15:32\]* **DEADLINE:** *\<2020-10-28
Wed\>*

**DONE** search for popular NLI datasets which have existing
RNN models as (almost) SOTAs, possibly use ones that were already tested
for eg. RTC or ones used in papers that may have semantic element

**CLOSED:** *\[2020-10-26 Mon 17:57\]* **DEADLINE:** *\<2020-10-28
Wed\>*

**DONE** explore below frameworks (by preference) and find
most feasible one

**CLOSED:** *\[2020-10-26 Mon 14:28\]* **DEADLINE:** *\<2020-10-26
Mon\>*

**DONE** add org-mode hook to remove startup visibility
headers in org-mode to markdown conversion

**CLOSED:** *\[2020-10-22 Thu 13:28\]*

**DONE** Set up repo, manuscript and develop log

**CLOSED:** *\[2020-10-22 Thu 12:36\]*

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
