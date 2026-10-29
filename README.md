# EPFL CS-433 Project 1: Predicting Coronary Heart Disease (MICHD)

Mahmoud Y. Boukouis, Zakarias Hofstetter, Jeniffer P. Bermudez

We predict whether a person has coronary heart disease (MICHD) from 321 variables
of the BRFSS health survey. We compare the linear methods seen in class, test
whether feature engineering helps them, and then implement gradient-boosted
decision trees from scratch in NumPy.

Only the Python standard library, NumPy and matplotlib (for figures only) are used.
The data files are not included in this repository.

## Results (5-fold cross-validation, F1 at the best threshold)

| Model | CV F1 | Time / fold |
|---|---|---|
| Least squares | 0.359 ± 0.103 | 14 s |
| Ridge, λ = 1e-5 | 0.4185 ± 0.0032 | 14 s |
| Logistic, γ = 0.5 | 0.4250 ± 0.0014 | 20 s |
| Reg. logistic, λ = 1e-4 | 0.4250 ± 0.0016 | 19 s |
| Reg. logistic + one-hot | 0.4282 ± 0.0010 | 19 s |
| **Boosted trees, 200 trees** | **0.4431 ± 0.0013** | 7.7 min |

Times measured on a 13-inch MacBook Air.

**Competition test set (AIcrowd):** our submission (boosted trees) reaches
**F1 = 0.454** and **accuracy = 0.874**.

## Setup

Put the three data files from the AIcrowd competition in a folder `dataset/`
next to the code, without renaming them:

```
dataset/x_train.csv
dataset/y_train.csv
dataset/x_test.csv
```

The first run reads the CSVs (a few minutes) and saves a fast cache
`dataset/cache.npz`, which later runs reuse.

## How to run

| Command | What it does | Time |
|---|---|---|
| `python run.py` | **Reproduces our best submission**: trains the boosted trees (200 trees, threshold 0.20) on all training data and writes `submission.csv` | ~10 min |
| `python experiments.py` | Runs all experiments of the report (linear models, ablation, step size, boosting) and saves the numbers to `results/results.json` | ~70 min |
| `python make_figures.py` | Makes the report figures and the results table (`results/*.pdf`, `results/table.tex`) from `results/results.json` | seconds |
| `python experiments.py --quick` | Same as above on the first 50,000 rows, to test that everything works | ~30 min |
| `python experiments.py --stages boosting` | Runs only some stages (`models`, `ablation`, `gamma`, `boosting`) | |

`run.py` uses fixed seeds, so it produces exactly the same `submission.csv` every time.

## Files

| File | Content |
|---|---|
| `implementations.py` | The 6 required functions: `mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression` |
| `helpers.py` | Course helpers: loading the CSVs, writing the submission |
| `preprocessing.py` | Data cleaning for the linear models (see below and the docstring at the top of the file) |
| `baseline.py` | Shared settings (seed 42, 5 folds, threshold grid, λ, γ) and a single reg. logistic pipeline |
| `compare_models.py` | Earlier script comparing the linear models (superseded by `experiments.py`) |
| `README.md` | This file |
| `boosting.py` | Gradient-boosted decision trees in pure NumPy |
| `experiments.py` | All experiments of the report, with timing |
| `make_figures.py` | Figures and LaTeX table |
| `run.py` | Reproduces the submission |

## Data preparation

BRFSS stores non-answers as numeric codes (see the
[codebook](https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf)):
7/77 = "don't know", 9/99 = "refused", 88 = "none", BLANK = "not asked or missing".
"Refused" and "don't know" carry no information about the true value and would be
read as real amounts (e.g. 99 days of poor health out of 30), so we turn them into
missing values. They are detected automatically from each column's maximum; this is
a heuristic that fits most questions but not all (for example, in `EMPLOY1`
the value 7 means "retired").

For the linear models, we then drop columns with more than 80% missing values,
fill the remaining gaps with the column median, standardize, drop constant columns
and add a bias (200 features). All statistics are computed on the training folds
only. The ablation tests three optional steps: "none" codes set to 0, missing
indicators, and one-hot encoding. Details are in the docstring of `preprocessing.py`.

## Why we moved from linear models to trees

Among linear models, the loss mattered most (logistic beats ridge by +0.0065 F1),
while regularization did not help and feature engineering added at most +0.003
(one-hot encoding). Missing indicators added only +0.0004, so missing values do
not seem to be the main limit.

This suggests that the linear form itself may limit performance. A linear model
computes `w1*x1 + w2*x2 + ...`: each feature has one fixed weight, so

- it cannot capture **curves**: age contributes along a straight line, while risk
  may rise slowly and then quickly. Capturing this requires adding features such
  as age² or age groups by hand;
- it cannot capture **interactions**: smoking adds the same amount at every age,
  while being old *and* a smoker may be riskier than the sum of both effects.
  Capturing this requires adding product features such as age × smoker by hand.

Decision trees can learn both on their own (several splits on age give a curve;
a split on age followed by a split on smoking gives an interaction). Boosted trees
gain +0.015 F1 over the best linear model without any feature engineering, which
is consistent with this explanation. We did not test which of the two effects
matters most; training boosting with depth 1 (curves only, no interactions) and
comparing it to depth 5 would answer this. The same explanation is in the docstring of `boosting.py`.

## Use of AI tools

Claude (Anthropic) helped with grammar, wording and drafting parts of the report
and of the experiment and plotting scripts. We ran all experiments and checked all
code, results and text.
