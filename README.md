# Project 1 – Machine Learning (CS-433)

This repository contains our code for Project 1 of the Machine Learning course at EPFL. The task is to predict a binary health outcome from survey data (BRFSS). It includes our implementations of the six required ML methods, the data preprocessing pipeline, a baseline based on regularized logistic regression, a gradient boosting model written from scratch in NumPy, and the experiments and figures used in our report.

## Team

-Mahmoud Y. Boukouis
-Zakarias Hofstetter
-Jeniffer P. Bermudez

## Repository structure

```
project-1-/
├── README.md
├── run.py                      Runs the six required methods
├── implementations.py          The six required ML methods (see below)
├── .gitignore
│
├── src/                        Core code
│   ├── helpers.py              Data loading and CSV submission (provided by the course)
│   ├── preprocessing.py        Data cleaning and feature processing
│   ├── baseline.py             Regularized logistic regression pipeline
│   ├── boosting.py             Gradient boosting model
│   └── boosting_parameters.py  Random search over the boosting hyperparameters
│
├── experiments/                Scripts for the report
│   ├── compare_models.py       Cross-validated comparison of all linear models
│   ├── experiments.py          Experiments (regularization, step size, threshold, ablation, timing)
│   └── make_figures.py         Figures and tables from the experiment results
│
├── results/                    Saved results, figures and tables
├── figures/                    Figures used in the report
├── submissions/                Prediction files
├── grading_tests/              Public tests provided by the course
└── dataset/                    Data files (not tracked, see below)
```

## Requirements

The code only uses Python, NumPy and Matplotlib (for the figures). The easiest way to get the right versions is the course environment:

```bash
cd grading_tests
conda env create --file=environment.yml --name=project1-grading
conda activate project1-grading
```

## Data

Download `x_train.csv`, `y_train.csv` and `x_test.csv` from the course page and place them in the `dataset/` folder. The data is not included in the repository because of its size. On the first run, the data is loaded from the CSV files and saved as `dataset/cache.npz`, which makes later runs much faster.

## How to run the code

All commands must be run from the main project folder. The scripts inside `src/` and `experiments/` are run as modules with `python -m`, so that they can import each other.

Most scripts accept `--quick`, which uses only the first 50,000 training rows and writes no submission. This is useful for testing.

### Required methods

```bash
python run.py
```

This runs each of the six functions in `implementations.py` on a small example and prints the resulting weights and loss.

### Baseline: regularized logistic regression

```bash
python -m src.baseline
```

This runs 5-fold cross-validation, chooses the decision threshold that maximizes F1, then retrains on all training data and writes `submissions/submission.csv`. Additional preprocessing options can be combined:

```bash
python -m src.baseline --quick              # first 50,000 rows, no submission
python -m src.baseline --one-hot            # one-hot encode categorical features
python -m src.baseline --none-codes         # turn BRFSS "none" codes (88/888) into 0
python -m src.baseline --missing-ind        # add "was missing" indicator columns
```

### Gradient boosting

```bash
python -m src.boosting
```

This runs 5-fold cross-validation to choose the number of trees (up to 250) and the decision threshold, then retrains on all training data and writes `submissions/submission_boosting.csv`. Use `--quick` for a fast test.

### Hyperparameter tuning for boosting

```bash
python -m src.boosting_parameters 10 100000
```

The first argument is the number of random trials and the second the number of training rows used. Each trial is evaluated with 3-fold cross-validation. The search uses a fixed seed, so it always tries the same settings in the same order. The results are saved in `results/tuning_results.json`.

### Experiments and figures

`make_figures` reads the results saved by `experiments`, so run them in this order:

```bash
python -m experiments.compare_models
python -m experiments.experiments
python -m experiments.make_figures
```

`experiments` saves its results in `results/results.json`, and `make_figures` writes the figures and the table into `results/`. With `--quick`, they use `results/results_quick.json` and `results/quick/` instead.

## Implemented methods

All functions are in `implementations.py` and return `(w, loss)`, where `w` is the final weight vector and `loss` the corresponding loss. For the regularized methods, the returned loss does not include the penalty term.

| Function | Description |
|---|---|
| `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)` | Linear regression using gradient descent |
| `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)` | Linear regression using stochastic gradient descent (batch size 1) |
| `least_squares(y, tx)` | Least squares using the normal equations |
| `ridge_regression(y, tx, lambda_)` | Ridge regression using the normal equations |
| `logistic_regression(y, tx, initial_w, max_iters, gamma)` | Logistic regression using gradient descent |
| `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` | Regularized logistic regression using gradient descent |

## Evaluation

The classes are strongly imbalanced (about 9% positives), so a model that always predicts "no" reaches about 91% accuracy but an F1 score of 0. We therefore evaluate all models with the F1 score, using k-fold cross-validation with a fixed seed (42). For each model, the decision threshold is chosen to maximize the mean F1 over the folds; the best thresholds are around 0.2 rather than the default 0.5. The preprocessing is fitted on the training folds only, so the validation fold never influences it.

## Boosting hyperparameters

The boosting hyperparameters in `src/boosting.py` were chosen by random search over 10 configurations, using 3-fold cross-validation on 100,000 training rows (`src/boosting_parameters.py`). The selected setting is:

| Parameter | Value |
|---|---|
| `MAX_DEPTH` | 5 |
| `LEARNING_RATE` | 0.1 |
| `REG_LAMBDA` | 0.1 |
| `MIN_CHILD_HESS` | 2.0 |
| `ROW_SUBSAMPLE` | 0.9 |
| `COL_SUBSAMPLE` | 0.5 |

It reached a cross-validated F1 of 0.4385 ± 0.0017, compared to 0.4344 for our initial settings. The best configurations scored within about 0.002 of each other, which is below the variation between folds. In all trials, the validation F1 peaked between about 75 and 125 trees while the training loss kept decreasing, a sign of overfitting beyond that point. The number of trees is therefore not fixed: `boosting.py` selects it by cross-validation, up to a maximum of 250.

## Running the grading tests

The `grading_tests/` folder contains the public tests provided by the course to check the functions in `implementations.py`. After creating and activating the environment (see Requirements), run:

```bash
cd grading_tests
pytest --github_link https://github.com/jenberleon/project-1-/tree/$(git rev-parse HEAD) .
```

The tests require a link to a specific commit, not just to the repository. The command above builds it automatically from the latest local commit, so make sure all changes are pushed with `git push` before running it. The link can also be found in the browser: open the repository page on GitHub and press the `y` key.

To test the local code without pushing, use:

```bash
pytest --github_link .. .
```