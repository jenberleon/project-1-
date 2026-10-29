# Project 1 – Machine Learning (CS-433)

This repository contains our code for Project 1 of the Machine Learning course at EPFL. It includes our implementations of the six required ML methods, the data preprocessing pipeline, our experiments, and the script that produces our final submission.

<!-- Add a short description of the task and your best result here, e.g. the dataset, the metric, and your final score on AIcrowd. -->

## Team

- Mahmoud Y. Boukouis  
- Zakarias Hofstetter
- Jeniffer P. Bermudez

## Repository structure

```
project-1-/
├── implementations.py       The six required ML methods (see below)
├── helpers.py               Helper functions (data loading, CSV submission)
├── preprocessing.py         Data cleaning and feature processing
├── baseline.py              Baseline models
├── boosting.py              Boosting model
├── boosting_parameters.py   Hyperparameters for the boosting model
├── compare_models.py        Comparison of the different models
├── experiments.py           Experiments and hyperparameter tuning
├── make_figures.py          Generates the figures in figures/
├── run.py                   Produces the final submission
├── tuning_results.json      Results of hyperparameter tuning
├── figures/                 Plots used in the report
├── results/                 Saved experiment results
├── grading_tests/           Public tests provided by the course
└── README.md
```

## Requirements

The code only uses Python and NumPy (plus Matplotlib for the figures). The easiest way to get the right versions is the course environment:

```bash
cd grading_tests
conda env create --file=environment.yml --name=project1-grading
conda activate project1-grading
```

## Data

Download the dataset from the course page and place the files in the `dataset/` folder. The data is not included in the repository because of its size.

## Reproducing the submission

From the main project folder, run:

```bash
python run.py
```

This loads the data, applies the preprocessing, trains the final model, and writes the predictions to `submission.csv`.

To regenerate the figures:

```bash
python make_figures.py
```

## Implemented methods

All functions are in `implementations.py` and return `(w, loss)`, where `w` is the final weight vector and `loss` is the corresponding loss. For the regularized methods, the returned loss does not include the penalty term.

| Function | Description |
|---|---|
| `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)` | Linear regression using gradient descent |
| `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)` | Linear regression using stochastic gradient descent (batch size 1) |
| `least_squares(y, tx)` | Least squares using the normal equations |
| `ridge_regression(y, tx, lambda_)` | Ridge regression using the normal equations |
| `logistic_regression(y, tx, initial_w, max_iters, gamma)` | Logistic regression using gradient descent |
| `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` | Regularized logistic regression using gradient descent |

## Running the grading tests

The `grading_tests/` folder contains the public tests provided by the course to check the functions in `implementations.py`.

After creating and activating the environment (see Requirements), run:

```bash
cd grading_tests
pytest --github_link https://github.com/jenberleon/project-1-/tree/$(git rev-parse HEAD) .
```

The tests require a link to a specific commit, not just to the repository. The command above builds it automatically from the latest local commit, so make sure all changes are pushed with `git push` before running it.

The link can also be found in the browser: open the repository page on GitHub and press the `y` key. The address bar then shows a link of the form `https://github.com/jenberleon/project-1-/tree/<commit-id>`.

To test the local code without pushing, use:

```bash
pytest --github_link .. .
```
