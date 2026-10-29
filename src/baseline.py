'''
Usage (run from the main project folder, expects the CSVs in dataset/):
  python -m src.baseline                     # full data, writes the submission
  python -m src.baseline --quick             # first 50,000 rows, no submission
  python -m src.baseline --one-hot           # one-hot encode categorical features
  python -m src.baseline --none-codes        # turn BRFSS "none" codes (88/888) into 0
  python -m src.baseline --missing-ind       # add "was missing" indicator columns
  python -m src.baseline --quick --one-hot   # options can be combined

'''
import sys
import numpy as np
from src.helpers import create_csv_submission
from implementations import reg_logistic_regression
from src.preprocessing import load_data, fit_preprocessing, transform

SEED = 42
K_FOLDS = 5
LAMBDA = 1e-4      # L2 regularization strength
GAMMA = 0.5        # step size (assumes your loss is the MEAN over samples)
MAX_ITERS = 300
THRESHOLDS = np.linspace(0.05, 0.95, 91)

QUICK = "--quick" in sys.argv      # fast experiment on 50,000 rows
ONE_HOT = "--one-hot" in sys.argv  # one-hot encode categorical features
NONE_CODES = "--none-codes" in sys.argv    # 88/888 ("none") -> 0
MISSING_IND = "--missing-ind" in sys.argv  # add "was missing" columns
PREPROCESSING_OPTIONS = dict(one_hot=ONE_HOT, none_codes=NONE_CODES,
                             missing_ind=MISSING_IND)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def f1_score(y_true, y_pred):
    """F1 for labels in {0, 1}."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0


def accuracy(y_true, y_pred):
    """Fraction of correct predictions (misleading on imbalanced data)."""
    return np.mean(y_true == y_pred)


def train(x_raw, y01):
    """Fit preprocessing and the model on (x_raw, y01)."""
    params = fit_preprocessing(x_raw, **PREPROCESSING_OPTIONS)
    tx = transform(x_raw, params)
    w, _ = reg_logistic_regression(y01, tx, LAMBDA, np.zeros(tx.shape[1]),
                                   MAX_ITERS, GAMMA)
    return params, w


def predict_proba(x_raw, params, w):
    return sigmoid(transform(x_raw, params) @ w)


def cross_validate(x, y01):
    """K-fold CV. Returns best threshold, its mean/std F1 and mean accuracy."""
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(len(y01)), K_FOLDS)
    scores = np.zeros((K_FOLDS, len(THRESHOLDS)))  # F1 per fold and threshold
    accs = np.zeros((K_FOLDS, len(THRESHOLDS)))    # accuracy per fold and threshold

    for k, val_idx in enumerate(folds):
        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != k])
        params, w = train(x[train_idx], y01[train_idx])
        proba = predict_proba(x[val_idx], params, w)
        for t, thr in enumerate(THRESHOLDS):
            pred = (proba >= thr).astype(int)
            scores[k, t] = f1_score(y01[val_idx], pred)
            accs[k, t] = accuracy(y01[val_idx], pred)
        print(f"fold {k + 1}/{K_FOLDS}: best F1 = {scores[k].max():.4f}")

    mean_f1 = scores.mean(axis=0)
    best = np.argmax(mean_f1)
    return THRESHOLDS[best], mean_f1[best], scores[:, best].std(), accs[:, best].mean()


if __name__ == "__main__":
    x_train, x_test, y_train, train_ids, test_ids = load_data("dataset/")
    if QUICK:
        x_train, y_train = x_train[:50000], y_train[:50000]
    print(f"Settings: quick={QUICK}, {PREPROCESSING_OPTIONS}")
    y01 = (y_train == 1).astype(float)  # labels {-1, 1} -> {0, 1}

    threshold, f1_mean, f1_std, acc_mean = cross_validate(x_train, y01)
    print(f"CV F1 = {f1_mean:.4f} +/- {f1_std:.4f} at threshold {threshold:.2f}")
    print(f"CV accuracy at that threshold = {acc_mean:.4f}")
    print(f"(always predicting 'no' would give accuracy {1 - y01.mean():.4f}, F1 0)")

    if QUICK:
        print("Quick mode: no submission written (run without --quick for that)")
    else:
        params, w = train(x_train, y01)
        y_pred = np.where(predict_proba(x_test, params, w) >= threshold, 1, -1)
        create_csv_submission(test_ids, y_pred, "submissions/submission.csv")
        print("Wrote submissions/submission.csv")