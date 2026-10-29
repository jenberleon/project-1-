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


