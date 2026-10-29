import json
import os
import sys
import time

import numpy as np

from src import boosting
from src.baseline import (
    SEED,
    K_FOLDS,
    LAMBDA,
    GAMMA,
    MAX_ITERS,
    THRESHOLDS,
    sigmoid,
    f1_score,
    accuracy,
)
from implementations import (
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)
from src.preprocessing import load_data, fit_preprocessing, transform

QUICK = "--quick" in sys.argv
ALL_STAGES = ["models", "ablation", "gamma", "boosting"]
LAMBDAS = [1e-5, 1e-4, 1e-3, 1e-2]
GAMMAS = [0.05, 0.1, 0.5, 1.0, 2.0]
GAMMA_ITERS = (
    2 * MAX_ITERS
)  # run longer than MAX_ITERS to see whether the loss has flattened
TREE_STEP = 25  # evaluate boosting every 25 trees
ABLATION = {
    "plain": {},
    "+ none codes": {"none_codes": True},
    "+ missing indicators": {"missing_ind": True},
    "+ one-hot": {"one_hot": True},
    "all three": {"none_codes": True, "missing_ind": True, "one_hot": True},
}
RESULTS_FILE = os.path.join(
    "results", "results_quick.json" if QUICK else "results.json"
)


# ---------------------------------------------------------------- helpers
def parse_stages():
    if "--stages" not in sys.argv:
        return ALL_STAGES
    stages = sys.argv[sys.argv.index("--stages") + 1].split(",")
    unknown = set(stages) - set(ALL_STAGES)
    if unknown:
        raise SystemExit(f"Unknown stage(s) {unknown}; choose from {ALL_STAGES}")
    return stages


def load_results():
    """Previous results, so stages run on different days end up in one file."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f)
    print(f"  saved to {RESULTS_FILE}")


def make_folds(n):
    """Same folds as baseline.py, compare_models.py and boosting.py."""
    rng = np.random.default_rng(SEED)
    return np.array_split(rng.permutation(n), K_FOLDS)


def split(folds, k):
    """(training indices, validation indices) for fold k."""
    return np.concatenate([f for i, f in enumerate(folds) if i != k]), folds[k]


def threshold_scores(y, score):
    """F1 and accuracy at every threshold in THRESHOLDS (two lists)."""
    f1, acc = [], []
    for thr in THRESHOLDS:
        pred = (score >= thr).astype(int)
        f1.append(float(f1_score(y, pred)))
        acc.append(float(accuracy(y, pred)))
    return f1, acc


def summarize(f1, acc):
    """Scores at the threshold with the best mean F1 over folds.

    f1 and acc have shape (folds, thresholds). Returns None if every fold failed.
    """
    f1, acc = np.array(f1, dtype=float), np.array(acc, dtype=float)
    mean = f1.mean(axis=0)
    if np.all(np.isnan(mean)):
        return None
    b = int(np.nanargmax(mean))
    return {
        "f1": float(mean[b]),
        "f1_std": float(f1[:, b].std()),
        "acc": float(acc[:, b].mean()),
        "threshold": float(THRESHOLDS[b]),
    }


def fmt_time(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.1f} s" if seconds < 60 else f"{seconds / 60:.1f} min"


def print_table(rows):
    print(
        f"\n  {'model':<30}{'CV F1':>17}{'accuracy':>10}{'threshold':>11}{'time/fold':>11}"
    )
    for name, r in rows.items():
        s = r["summary"]
        if s is None:
            print(f"  {name:<30}{'failed':>17}")
            continue
        f1_text = f"{s['f1']:.4f} +/- {s['f1_std']:.4f}"
        print(
            f"  {name:<30}{f1_text:>17}{s['acc']:>10.4f}{s['threshold']:>11.2f}"
            f"{fmt_time(np.nanmean(r['time'])):>11}"
        )


# ---------------------------------------------------------------- linear models
def linear_models(
    lambdas, families=("least squares", "ridge", "logistic", "reg. logistic")
):
    """name -> (fit, family, lambda). fit(y, tx) returns (w, outputs_probability).

    Regression models fit the 0/1 labels directly, so tx @ w is already a score
    around [0, 1]; logistic models need the sigmoid.
    """

    def zeros(tx):
        return np.zeros(tx.shape[1])

    models = {}
    if "least squares" in families:
        models["least squares"] = (
            lambda y, tx: (least_squares(y, tx)[0], False),
            "least squares",
            None,
        )
    if "ridge" in families:
        for lam in lambdas:
            models[f"ridge lambda={lam:g}"] = (
                lambda y, tx, lam=lam: (ridge_regression(y, tx, lam)[0], False),
                "ridge",
                lam,
            )
    if "logistic" in families:
        models["logistic"] = (
            lambda y, tx: (
                logistic_regression(y, tx, zeros(tx), MAX_ITERS, GAMMA)[0],
                True,
            ),
            "logistic",
            None,
        )
    if "reg. logistic" in families:
        for lam in lambdas:
            models[f"reg. logistic lambda={lam:g}"] = (
                lambda y, tx, lam=lam: (
                    reg_logistic_regression(y, tx, lam, zeros(tx), MAX_ITERS, GAMMA)[0],
                    True,
                ),
                "reg. logistic",
                lam,
            )
    return models


def cv_linear(x, y, folds, models, options):
    """K-fold CV of the given models with one preprocessing setting.

    Preprocessing is fitted once per fold on the training part and shared by
    all models, so they are compared on exactly the same features.
    """
    out = {
        name: {"family": fam, "lambda": lam, "f1": [], "acc": [], "time": []}
        for name, (_, fam, lam) in models.items()
    }
    prep_time, n_features = [], 0
    for k in range(K_FOLDS):
        tr, va = split(folds, k)
        t0 = time.perf_counter()
        x_tr = x[tr]
        params = fit_preprocessing(x_tr, **options)
        tx_tr, tx_va = transform(x_tr, params), transform(x[va], params)
        del x_tr
        prep_time.append(time.perf_counter() - t0)
        n_features = tx_tr.shape[1] - 1  # without the bias column
        for name, (fit, _, _) in models.items():
            r = out[name]
            t0 = time.perf_counter()
            try:
                with np.errstate(over="ignore"):
                    w, is_prob = fit(y[tr], tx_tr)
            except np.linalg.LinAlgError:
                print(f"    {name}: singular matrix, skipped on this fold")
                r["f1"].append([float("nan")] * len(THRESHOLDS))
                r["acc"].append([float("nan")] * len(THRESHOLDS))
                r["time"].append(float("nan"))
                continue
            r["time"].append(time.perf_counter() - t0)
            score = tx_va @ w
            if is_prob:
                score = sigmoid(score)
            f1, acc = threshold_scores(y[va], score)
            r["f1"].append(f1)
            r["acc"].append(acc)
        print(f"  fold {k + 1}/{K_FOLDS} done ({n_features} features)")
    for r in out.values():
        r["summary"] = summarize(r["f1"], r["acc"])
        r["n_features"] = n_features
        r["prep_time"] = prep_time
    return out


def best_lambda(results):
    """Lambda of the best reg. logistic model in the 'models' stage."""
    if "models" not in results:
        print(
            f"  ('models' stage not run yet: using LAMBDA={LAMBDA:g} from baseline.py)"
        )
        return LAMBDA
    rows = [
        r
        for r in results["models"]["rows"].values()
        if r["family"] == "reg. logistic" and r["summary"]
    ]
    return max(rows, key=lambda r: r["summary"]["f1"])["lambda"] if rows else LAMBDA


# ---------------------------------------------------------------- stages
def stage_models(x, y, folds, results):
    print("\n=== models: all linear models, plain preprocessing ===")
    rows = cv_linear(x, y, folds, linear_models(LAMBDAS), {})
    print_table(rows)
    results["models"] = {"rows": rows}


def stage_ablation(x, y, folds, results):
    lam = best_lambda(results)
    print(
        f"\n=== ablation: reg. logistic, lambda={lam:g}, each preprocessing option ==="
    )
    rows = {}
    for setting, options in ABLATION.items():
        name = f"reg. logistic lambda={lam:g}"
        if (
            setting == "plain"
            and "models" in results
            and name in results["models"]["rows"]
        ):
            rows[setting] = results["models"]["rows"][name]  # same folds: reuse
            print(f"\n  setting '{setting}': reused from the 'models' stage")
            continue
        print(f"\n  setting '{setting}'")
        model = linear_models([lam], families=("reg. logistic",))
        rows[setting] = cv_linear(x, y, folds, model, options)[name]
    print_table(rows)
    results["ablation"] = {"lambda": lam, "rows": rows}


def stage_gamma(x, y, folds, results):
    """Loss curves, using logistic_regression from implementations.py one step
    at a time (max_iters=1, starting from the previous w)."""
    print(f"\n=== gamma: training loss over {GAMMA_ITERS} iterations, fold 1 ===")
    tr, va = split(folds, 0)
    x_tr = x[tr]
    params = fit_preprocessing(x_tr)
    tx_tr, tx_va = transform(x_tr, params), transform(x[va], params)
    del x_tr
    curves = {}
    for gamma in GAMMAS:
        w, losses = np.zeros(tx_tr.shape[1]), []
        t0 = time.perf_counter()
        with np.errstate(all="ignore"):  # large gammas may diverge: that is the point
            for _ in range(GAMMA_ITERS):
                w, loss = logistic_regression(y[tr], tx_tr, w, 1, gamma)
                losses.append(float(loss))
            f1, _ = threshold_scores(y[va], sigmoid(tx_va @ w))
        curves[f"{gamma:g}"] = {
            "loss": losses,
            "val_f1": max(f1),
            "time": time.perf_counter() - t0,
        }
        print(
            f"  gamma={gamma:g}: loss after {MAX_ITERS} it. {losses[MAX_ITERS - 1]:.4f}, "
            f"after {GAMMA_ITERS} it. {losses[-1]:.4f}, validation F1 {max(f1):.4f}"
        )
    results["gamma"] = {"max_iters": MAX_ITERS, "curves": curves}


def stage_boosting(x, y, folds, results):
    print(
        f"\n=== boosting: up to {boosting.N_TREES} trees, depth {boosting.MAX_DEPTH}, "
        f"learning rate {boosting.LEARNING_RATE} ==="
    )
    boosting.CHECK_EVERY = TREE_STEP  # fit_boosting reads this module setting
    checkpoints = list(range(TREE_STEP, boosting.N_TREES + 1, TREE_STEP))
    f1 = np.zeros((K_FOLDS, len(checkpoints), len(THRESHOLDS)))
    acc = np.zeros_like(f1)
    logloss = np.zeros((K_FOLDS, len(checkpoints)))
    times = []
    for k in range(K_FOLDS):
        print(f"  fold {k + 1}/{K_FOLDS}")
        tr, va = split(folds, k)
        t0 = time.perf_counter()
        xb_tr, xb_va = boosting.prepare(x[tr], [x[va]])
        _, _, probas = boosting.fit_boosting(xb_tr, y[tr], boosting.N_TREES, xb_va)
        times.append(time.perf_counter() - t0)
        y_va = y[va]
        for c, m in enumerate(checkpoints):
            p = np.clip(probas[m], 1e-12, 1 - 1e-12)
            f1[k, c], acc[k, c] = threshold_scores(y_va, p)
            logloss[k, c] = -np.mean(y_va * np.log(p) + (1 - y_va) * np.log(1 - p))
        print(f"  fold {k + 1}/{K_FOLDS} took {fmt_time(times[-1])}")

    per_trees = [summarize(f1[:, c], acc[:, c]) for c in range(len(checkpoints))]
    best = max(range(len(checkpoints)), key=lambda c: per_trees[c]["f1"])
    print("\n  trees   CV F1             val. log-loss  threshold")
    for c, m in enumerate(checkpoints):
        s = per_trees[c]
        print(
            f"  {m:>5}   {s['f1']:.4f} +/- {s['f1_std']:.4f}   {logloss[:, c].mean():.4f}"
            f"         {s['threshold']:.2f}"
        )
    s = per_trees[best]
    print(
        f"\n  best: {checkpoints[best]} trees, threshold {s['threshold']:.2f}, "
        f"CV F1 {s['f1']:.4f} -> put these two numbers in run.py"
    )
    results["boosting"] = {
        "checkpoints": checkpoints,
        "f1": f1.tolist(),
        "acc": acc.tolist(),
        "val_logloss": logloss.tolist(),
        "time": times,
        "per_trees": per_trees,
        "best_trees": checkpoints[best],
        "summary": s,
        "settings": {
            "n_bins": boosting.N_BINS,
            "max_depth": boosting.MAX_DEPTH,
            "learning_rate": boosting.LEARNING_RATE,
            "reg_lambda": boosting.REG_LAMBDA,
            "min_child_hess": boosting.MIN_CHILD_HESS,
            "row_subsample": boosting.ROW_SUBSAMPLE,
            "col_subsample": boosting.COL_SUBSAMPLE,
        },
    }


STAGES = {
    "models": stage_models,
    "ablation": stage_ablation,
    "gamma": stage_gamma,
    "boosting": stage_boosting,
}

if __name__ == "__main__":
    stages = parse_stages()
    x, x_test, y_pm, _, _ = load_data("dataset/")
    del x_test  # not needed here; frees memory
    if QUICK:
        x, y_pm = x[:50000], y_pm[:50000]
    y = (y_pm == 1).astype(float)  # labels {-1, 1} -> {0, 1}
    folds = make_folds(len(y))
    print(
        f"{len(y)} rows, {x.shape[1]} raw features, {100 * y.mean():.1f}% positive, "
        f"stages: {stages}"
    )

    results = load_results()
    results["info"] = {
        "quick": QUICK,
        "n_rows": len(y),
        "n_raw_features": x.shape[1],
        "positive_rate": float(y.mean()),
        "thresholds": THRESHOLDS.tolist(),
        "k_folds": K_FOLDS,
        "seed": SEED,
        "gamma": GAMMA,
        "max_iters": MAX_ITERS,
    }
    for stage in stages:
        t0 = time.perf_counter()
        STAGES[stage](x, y, folds, results)
        results[stage]["stage_time"] = time.perf_counter() - t0
        print(f"  stage '{stage}' took {fmt_time(results[stage]['stage_time'])}")
        save_results(results)
