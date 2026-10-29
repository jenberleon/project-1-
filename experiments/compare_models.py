import sys
import numpy as np
from implementations import (
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)
from src.preprocessing import load_data, fit_preprocessing, transform
from src.baseline import (
    SEED,
    K_FOLDS,
    PREPROCESSING_OPTIONS,
    GAMMA,
    MAX_ITERS,
    THRESHOLDS,
    sigmoid,
    f1_score,
    accuracy,
)

LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2]

# Each model: name -> function(y01, tx) returning (w, outputs_probability).
# Regression models fit the 0/1 labels directly, so tx @ w is already a score
# roughly between 0 and 1; logistic models need the sigmoid.
MODELS = {
    "least squares": lambda y, tx: (least_squares(y, tx)[0], False),
}
for lam in LAMBDAS:
    MODELS[f"ridge lambda={lam:g}"] = lambda y, tx, lam=lam: (
        ridge_regression(y, tx, lam)[0],
        False,
    )
MODELS["logistic (no reg.)"] = lambda y, tx: (
    logistic_regression(y, tx, np.zeros(tx.shape[1]), MAX_ITERS, GAMMA)[0],
    True,
)
for lam in LAMBDAS:
    MODELS[f"reg. logistic lambda={lam:g}"] = lambda y, tx, lam=lam: (
        reg_logistic_regression(y, tx, lam, np.zeros(tx.shape[1]), MAX_ITERS, GAMMA)[0],
        True,
    )


def compare(x, y01):
    """Return F1 and accuracy arrays (folds x thresholds) for every model."""
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(len(y01)), K_FOLDS)
    f1 = {name: np.full((K_FOLDS, len(THRESHOLDS)), np.nan) for name in MODELS}
    acc = {name: np.full((K_FOLDS, len(THRESHOLDS)), np.nan) for name in MODELS}

    for k, val_idx in enumerate(folds):
        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != k])
        x_tr = x[train_idx]
        params = fit_preprocessing(x_tr, **PREPROCESSING_OPTIONS)
        tx_tr, tx_val = transform(x_tr, params), transform(x[val_idx], params)
        y_tr, y_val = y01[train_idx], y01[val_idx]
        del x_tr

        for name, fit in MODELS.items():
            try:
                w, is_prob = fit(y_tr, tx_tr)
            except np.linalg.LinAlgError:
                print(f"  {name}: singular matrix (correlated features), skipped")
                continue
            score = tx_val @ w
            if is_prob:
                score = sigmoid(score)
            for t, thr in enumerate(THRESHOLDS):
                pred = (score >= thr).astype(int)
                f1[name][k, t] = f1_score(y_val, pred)
                acc[name][k, t] = accuracy(y_val, pred)
        print(f"fold {k + 1}/{K_FOLDS} done")
    return f1, acc


def print_table(f1, acc, y01):
    """Print one row per model at its best threshold."""
    print(f"\n{'model':<30}{'CV F1':>18}{'accuracy':>10}{'threshold':>11}")
    print(f"{'always predict no':<30}{'0.0000':>18}{1 - y01.mean():>10.4f}{'-':>11}")
    for name in MODELS:
        mean_f1 = f1[name].mean(axis=0)
        if np.all(np.isnan(mean_f1)):
            print(f"{name:<30}{'failed':>18}")
            continue
        b = np.nanargmax(mean_f1)
        f1_text = f"{mean_f1[b]:.4f} +/- {f1[name][:, b].std():.4f}"
        print(
            f"{name:<30}{f1_text:>18}{acc[name][:, b].mean():>10.4f}"
            f"{THRESHOLDS[b]:>11.2f}"
        )


if __name__ == "__main__":
    x_train, _, y_train, _, _ = load_data("dataset/")
    if "--quick" in sys.argv:
        x_train, y_train = x_train[:50000], y_train[:50000]
        print("Quick mode: using the first 50,000 rows")
    print(f"Settings: {PREPROCESSING_OPTIONS}")
    y01 = (y_train == 1).astype(float)  # labels {-1, 1} -> {0, 1}

    f1, acc = compare(x_train, y01)
    print_table(f1, acc, y01)
