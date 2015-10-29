# ActiveTM

`ActiveT(opic)M(odeling)` is a framework for building experiments to measure the
effects of active learning on predictive topic models.

## Prerequisites

The Python code relies on ctypes to get reasonable computation speeds for
sampling based methods.  To compile the C code, make sure you have the following
programs and libraries installed:

* `gcc`
* `make`
* `LAPACK`
* `LAPACK` development headers

Then run `make` in the `activetm/models/sampler` directory.

To run the Python code, you will need to install the following modules:

* [`ankura`](https://github.com/jlund3/ankura)
* `numpy`
* `sklearn`

It is possible to install these modules by running

```
pip install -r requirements.txt
```

in the `ActiveTM` directory.

Note that in order for the code to work, it is necessary to tweak `ankura`
slightly.  Change the return value of `ankura.topic.predict_topics` from `z` to
`counts`.

## Settings

Running the code requires a settings file to define the arguments and parameters
used in a given run.  Blank lines and comment lines are ignored.  A comment line
begins with `#`.  Lines specifying settings take the form:

```
[key]<WHITESPACE>[val]
```

where `[key]` is the name of a setting, `<WHITESPACE>` is some whitespace
character, excluding the newline character, and `[val]` is the value for
`[key]`.  Depending on certain settings, more settings may be required to be
defined.  To help users with these intricacies, a description follows of the
`[key]`s available and/or expected and their valid `[val]`s.

Note also that there is an example settings file:  `example.settings`.

### Universally Required Settings

These settings are required for every run.

* `pickle` takes a string which names the pickle file created for the data set
* `corpus` takes one of two string.  One is a glob pattern specifying where the
  corpus can be found.  Check the `ankura.pipeline.read_glob` documentation for
  further details of this use.  The other is the path to a file containing
  documents all in one file.  Check the `ankura.pipeline.read_file` for futher
  details of this use.
* `labels` takes the path to a file containing the labels for the documents
  imported from `corpus`.  It takes the form:

    ```
    [id]<WHITESPACE>[label]
    ```

  where `[id]` is the identification of a given document according to `ankura`'s
  import pipeline and `[label]` is some floating point number representing that
  document's label.
* `stopwords` takes the path to a file containing stopwords to ignore during
  corpus import.  The form of the file is merely one word per line.
* `rare` takes an integer as the parameter to `ankura.filter_rarewords`.
* `common` takes an integer as the parameter to `ankura.filter_commonwords`.
* `smalldoc` takes an integer as the parameter to `ankura.filter_smalldocs`.
* `pregenerate` takes any string value.  When the value is `YES`, the import
  process will also invoke `ankura.pregenerate_doc_tokens`.
* `group` takes a string value.  This value is used to specify a subdirectory
  within an output directory (specified when a script gets called) where a file
  containing results of the run are written.
* `seed` takes an integer value for seeding the random number generator used in
  the experiment.
* `select` takes a string value specifying which selection method to use when
  deciding which documents to have labeled for the next training run.  The
  string value options are keys of `active.select.factory`.
* `testsize` takes an integer value specifying the number of documents to use as
  a test set (for evaluating the learner).
* `startlabeled` takes an integer value specifying the number of documents to
  include in the initial labeled set
* `endlabeled` takes an integer value specifying the number of documents that
  will be in the labeled set on the final training run of the experiment.
* `increment` takes an integer value specfiying the number of documents to add
  to the labeled set between training runs
* `candsize` takes an integer value specifying the number of documents from the
  unlabeled set to consider during selection.  Whereas `increment` specifies how
  many documents will be added to the labeled set between training runs,
  `candsize` specifies how many documents become candidates to become part of
  the labeled set.
* `model` takes a string value specifying the type of model to use for training.
  The options are the keys of `models.models.factory`.

### Model Specific Settings

Each model requires slightly different settings to be specified.

#### Settings for `slda`

This model is an implementation of supervised latent Dirichlet allocation, using
Gibbs sampling to perform inference.

* `cseed` takes an integer value for seeding the random number generator used
  for Gibbs sampling.
* `numtopics` takes an integer value specifying the number of topics the model
  considers.
* `alpha` takes a floating point value specifying the value of the
  hyper-parameter over the topic distributions of the documents, as a symmetric
  Dirichlet distribution.
* `beta` takes a floating point value specifying the value of the
  hyper-parameter over the word distributions of the topics, as a symmetric
  Dirichlet distribution.
* `var` takes a floating point value specifying the value of the variance
  hyper-parameter over the label distribution of the documents.
* `numtrain` takes an integer value specifying the number of training chains to
  run Gibbs sampling on.
* `numsamples train` takes an integer value specifying the number of states to
  keep from each training chain.
* `trainburn` takes an integer value specifying the number of states to sample
  before saving the state.  This burn value gets used only once per training
  chain, the first time it is trained.
* `trainlag` takes an integer value specifying the number of states to sample
  between saved states.
* `numsamplespredict` takes an integer value specifying the number of states to
  take per saved training states when running Gibbs sampling during prediction.
* `predictburn` takes an integer value specifying the number of states to sample
  before saving the state for prediction.
* `predictlag` takes an integer value specifying the number of states to sample
  between saved states for prediction.

#### Settings for `sup_anchor`

This model is an implementation of supervised anchor words.

* `numtopics` takes an integer value specifying the number of topics the model
  considers.
* `numtrain` takes an integer value specifying the number of anchor words
  instances to train during on training iteration.
