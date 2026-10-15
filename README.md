# Public Comment Classification

Research code for public comment classification and model evaluation using the NEPABench BinAssign task.

## Background

Organizing public input into issue categories is one component of comment review. Automated classification can assist that organization, but useful evaluation should establish more than a single overall score: it should show which categories and document groups support the reported results.

The [Permitting Technology Action Plan (PTAP)](https://permitting.innovation.gov/resources/action-plan/functional-requirements/) identifies automated comment compilation and analysis as a capability for modernizing federal environmental review and permitting. This project examines the classification and evaluation component through independent benchmark research, focusing on how model comparisons depend on the composition of the test data.

## Task and approach

The task is to assign a prepared comment excerpt to one of the predefined categories for its project, then compare the prediction with the benchmark's reference label. The code evaluates category assignment, rather than comment summarization or generation of agency responses.

It compares **lexical**, **semantic**, and **combined** representations: features derived from wording, sentence embeddings, and their combination. Text-grouped comparisons are run separately for each project. The document-grouped experiments focus on the benchmark's migratory-bird permitting project and require document-group information alongside the prepared excerpts.

The evaluation supports analysis of performance on held-out groups, coverage across repeated tests, and the category-level contributions to an overall difference between methods. Related material is kept together according to the selected grouping protocol.

## What the code produces

With the required inputs, the experiment scripts write machine-readable predictions and evaluation summaries.

| Output | What it helps examine |
| --- | --- |
| Predicted category labels | How evaluated excerpts are assigned to the project's categories. |
| Overall and category-level scores | Performance across categories, individual excerpts, and the groups defined for the experiment. |
| Coverage and comparison diagnostics | Which groups were tested, how categories contribute to model differences, and where methods make different errors. |

Metrics include macro-F1, which weights evaluated categories equally, excerpt accuracy, and group-averaged accuracy.

These outputs support comparisons of classification methods while making the role of category support and test-group coverage explicit.
