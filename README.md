# Predicting MICHD from BRFSS health survey data

This project predicts whether a person has **MICHD** (coronary heart disease or
myocardial infarction) from their answers to the BRFSS health survey. The
labels are in `{-1, 1}`, and the classes are strongly imbalanced, so models are
evaluated with the **F1 score** rather than accuracy.

Everything is written in pure **NumPy**: no scikit-learn, pandas or other ML
libraries are used.

We compare two families of models:

- **Linear models** (least squares, ridge regression, logistic and regularized
  logistic regression), implemented from scratch in `implementations.py`.
- **Gradient-boosted decision trees** (`boosting.py`), also from scratch,
  which can learn non-linear effects ("risk rises faster after 60") and
  interactions ("old AND smoker") that a linear model cannot.

---

## Repository structure

| File | Purpose |
|---|---|
| `implementations.py` | The six required ML methods (GD, SGD, least squares, ridge, logistic, regularized logistic) and their loss/gradient helpers. |
| `preprocessing.py` | Data loading (`load_data`), BRFSS missing-code handling, and the preprocessing pipeline (`fit_preprocessing`, `transform`). |
| `helpers.py` | `create_csv_submission`, writes predictions in the submission format. |
| `baseline.py` | Main linear pipeline: 5-fold CV of regularized logistic regression, threshold tuning, final `submission.csv`. Also defines the shared settings (seed, folds, thresholds, metrics). |
| `compare_models.py` | Compares all linear models under identical conditions (same preprocessing, folds and threshold search). |
| `boosting.py` | Gradient-boosted trees: CV over the number of trees and threshold, final `submission_boosting.csv`. |
| `experiments.py` | Runs every experiment used in the report and saves the numbers to `results/results.json`. |
| `make_figures.py` | Builds the report figures from `results/results.json`. |
| `run.py` | Produces the final submission with the selected settings. |

---

## Requirements

- Python 3.8+
- NumPy
- Matplotlib (only for `make_figures.py`)

```bash
pip install numpy matplotlib
```

## Data

The dataset is **not included** in this repository: the CSV files are several
hundred MB, too large for GitHub.

To run the code:

1. Download the dataset from **[TODO: link to the course / AIcrowd page]**.
2. Unzip it and place the CSV files in a `dataset/` folder at the root of the
   repository:

```
dataset/
├── x_train.csv
├── y_train.csv
└── x_test.csv
```

All scripts read the data from `dataset/`, so no path needs to be changed.
The `dataset/` folder is listed in `.gitignore` so it is never committed by
accident.

---

## Quick start

Reproduce the final submission:

```bash
git clone <repository-url>
cd <repository-folder>
# download the data into dataset/ (see "Data" above)
python run.py
```

Every script accepts `--quick`, which uses only the first 50,000 training rows
and writes no submission. Use it to check that everything runs before
launching a full (slow) run:

```bash
python baseline.py --quick
python boosting.py --quick
python experiments.py --quick
```

---

## Scripts in detail

### `baseline.py`: regularized logistic regression

```bash
python baseline.py                     # full data, writes submission.csv
python baseline.py --one-hot           # one-hot encode categorical features
python baseline.py --none-codes        # turn BRFSS "none" codes (88/888) into 0
python baseline.py --missing-ind       # add "was missing" indicator columns
python baseline.py --quick --one-hot   # options can be combined
```

1. Loads and preprocesses the data.
2. Runs 5-fold cross-validation of regularized logistic regression and picks
   the decision threshold that maximizes the mean validation F1.
3. Retrains on all training data, predicts the test set and writes
   `submission.csv`.

### `compare_models.py`: fair comparison of linear models

```bash
python compare_models.py            # full data
python compare_models.py --quick    # first 50,000 rows
python compare_models.py --one-hot  # same preprocessing flags as baseline.py
```

On every fold, preprocessing is fitted **once** on the training part and shared
by all models, so they are compared on exactly the same features. Models
compared: least squares, ridge regression and regularized logistic regression
for each λ in `{1e-5, 1e-4, 1e-3, 1e-2}`, and unregularized logistic
regression. Each model is reported at its own best threshold, next to the
trivial "always predict no" baseline.

If a model hits a singular matrix (e.g. least squares on strongly correlated
features), it is skipped on that fold instead of crashing the run.

### `boosting.py`: gradient-boosted trees

```bash
python boosting.py          # CV, then trains on all data, writes submission_boosting.csv
python boosting.py --quick  # first 50,000 rows only, no submission
```

How it works:

- Each feature is cut into at most 32 bins using quantiles of the **training
  fold only**; bin 0 is reserved for missing values.
- Boosting starts from a constant prediction (the overall log-odds of MICHD)
  and adds small trees one at a time. Each tree is fitted to the gradient and
  Hessian of the logistic loss at the current prediction (the XGBoost idea).
- At each split, missing values are sent to whichever side gives the larger
  gain, so no imputation is needed.
- Each leaf outputs a log-odds correction; the predicted probability is
  `sigmoid(base log-odds + sum of all tree corrections)`.
- Cross-validation chooses both the number of trees (checked every 50 trees)
  and the decision threshold.

### `experiments.py`: all numbers for the report

```bash
python experiments.py --quick                    # 50,000 rows, to test first
python experiments.py                            # full data, all stages (slow)
python experiments.py --stages models,ablation   # only some stages
python make_figures.py                           # figures (add --quick for quick results)
```

| Stage | What it does | Used for |
|---|---|---|
| `models` | All linear models with plain preprocessing, for every λ | Method comparison, choice of λ |
| `ablation` | Reg. logistic regression (best λ) with each preprocessing option switched on | Ablation table |
| `gamma` | Training loss per iteration for γ ∈ `{0.05, 0.1, 0.5, 1, 2}` on one fold, for 600 iterations | Choice of γ and of the number of iterations |
| `boosting` | CV F1 and validation log-loss every 25 trees | Choice of the number of trees |

Results are saved to `results/results.json` (`results/results_quick.json`
with `--quick`). Results from earlier runs are loaded and kept, so stages can
be run separately (e.g. on different days) and still end up in one file.
Training time is recorded for every model.

---

## Methods

### Preprocessing

Preprocessing is always **fitted on the training fold only** and then applied
to the validation fold (and, for the final model, to the test set), so no
information leaks from validation data.

- **Missing codes**: BRFSS encodes answers such as "don't know" or "refused"
  with special values (e.g. 7/9, 77/99). Columns using such codes are detected
  automatically and those values are treated as missing.
- **Optional steps** (switched on by command-line flags, tested in the
  ablation):
  - `--none-codes`: BRFSS "none" codes (88/888) become 0, since they mean a
    count of zero rather than a missing answer.
  - `--missing-ind`: adds a column per feature telling whether the value was
    missing; missingness itself can be informative.
  - `--one-hot`: one-hot encodes categorical features, so the model no longer
    treats category codes as ordered numbers.

### Models (`implementations.py`)

| Function | Method |
|---|---|
| `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)` | Linear regression, gradient descent |
| `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)` | Linear regression, SGD (batch size 1) |
| `least_squares(y, tx)` | Normal equations |
| `ridge_regression(y, tx, lambda_)` | Normal equations with L2 penalty |
| `logistic_regression(y, tx, initial_w, max_iters, gamma)` | Gradient descent, `y ∈ {0, 1}` |
| `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` | Gradient descent with L2 penalty, `y ∈ {0, 1}` |

All functions return `(w, loss)`. Conventions:

- The MSE loss is `(1/2N) Σ (y − Xw)²`; the logistic loss is the **mean**
  binary cross-entropy.
- Ridge minimizes `(1/2N)‖y − Xw‖² + λ‖w‖²`, which gives
  `(XᵀX + 2Nλ I) w = Xᵀy`.
- Regularized logistic regression minimizes the mean logistic loss plus
  `λ‖w‖²`, so the penalty adds `2λw` to the gradient.
- The returned loss **never includes** the regularization term.

Regression models are fitted directly on the 0/1 labels, so `tx @ w` is used
as a score around [0, 1]; logistic models pass it through a sigmoid.

### Evaluation

- **5-fold cross-validation** with a fixed seed (42). All scripts build the
  folds the same way, so results from different scripts are directly
  comparable.
- **Threshold tuning**: the default threshold of 0.5 predicts far too few
  positives on imbalanced data. Every model is evaluated at thresholds from
  0.05 to 0.95 (step 0.01), and we keep the threshold with the best **mean F1
  over the folds**.
- We report the mean F1 ± its standard deviation over folds, and the accuracy
  at the chosen threshold. Accuracy alone is misleading here: always
  predicting "no" already gets high accuracy with an F1 of 0.

---

## Hyperparameters

**Linear models** (`baseline.py`)

| Setting | Value |
|---|---|
| λ (L2 strength) | `1e-4` |
| γ (step size) | `0.5` |
| Iterations | `300` |

**Gradient boosting** (`boosting.py`)

| Setting | Value | Meaning |
|---|---|---|
| `N_BINS` | 32 | Bins per feature (bin 0 = missing) |
| `N_TREES` | 250 | Maximum number of trees |
| `MAX_DEPTH` | 5 | Depth of each tree |
| `LEARNING_RATE` | 0.1 | Shrinks each tree's correction |
| `REG_LAMBDA` | 1.0 | L2 penalty on leaf values |
| `MIN_CHILD_HESS` | 10.0 | Minimum Hessian sum per leaf (≈ minimum leaf size) |
| `ROW_SUBSAMPLE` | 0.5 | Fraction of rows used per tree |
| `COL_SUBSAMPLE` | 0.5 | Fraction of features considered per tree |

---

## Results

5-fold cross-validation on the full training set:

| Model | CV F1 | Accuracy | Threshold |
|---|---|---|---|
| Always predict "no" | 0.0000 | _TODO_ | – |
| Least squares | _TODO_ | _TODO_ | _TODO_ |
| Ridge regression (best λ) | _TODO_ | _TODO_ | _TODO_ |
| Logistic regression | _TODO_ | _TODO_ | _TODO_ |
| Reg. logistic regression (best λ) | _TODO_ | _TODO_ | _TODO_ |
| Gradient-boosted trees | _TODO_ | _TODO_ | _TODO_ |

Fill these in from the output of `python experiments.py`.

---

## Reproducibility

- All random choices (fold split, row and column subsampling in boosting) use
  the seed `42`, so every run gives the same numbers.
- `mean_squared_error_sgd` uses `np.random` without a seed; it is not used in
  the experiments.

## Authors

- _Name 1_
- _Name 2_
- _Name 3_