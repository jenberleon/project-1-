"""
Usage:
  python -m src.boosting_parameters                 # 20 trials on 100,000 rows, 3 folds
  python tune.py 10 100000       # 10 trials on 100,000 rows
"""
import sys
import json
import time
import numpy as np
from src import boosting
from src.preprocessing import load_data

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 100_000
FOLDS = 3  # fewer folds than the final run, to save time
TREE_BUDGET = 25.0  # trees = TREE_BUDGET / learning rate (0.1 -> 250 trees)

SPACE = {
    "MAX_DEPTH": [3, 4, 5],
    "LEARNING_RATE": [0.05, 0.1, 0.2],
    "REG_LAMBDA": [0.1, 1.0, 5.0, 20.0],
    "MIN_CHILD_HESS": [2.0, 5.0, 10.0, 30.0, 100.0],
    "ROW_SUBSAMPLE": [0.5, 0.7, 0.9],
    "COL_SUBSAMPLE": [0.3, 0.5, 0.8],
}

# Your current setting, run first as the reference point.
DEFAULT = {
    "MAX_DEPTH": 5,
    "LEARNING_RATE": 0.1,
    "REG_LAMBDA": 1.0,
    "MIN_CHILD_HESS": 10.0,
    "ROW_SUBSAMPLE": 0.5,
    "COL_SUBSAMPLE": 0.5,
}


def apply(params):
    for name, value in params.items():
        setattr(boosting, name, value)
    # A smaller learning rate needs more trees to reach the same fit.
    boosting.N_TREES = int(round(TREE_BUDGET / params["LEARNING_RATE"]))
    boosting.CHECK_EVERY = 25
    boosting.K_FOLDS = FOLDS


if __name__ == "__main__":
    x, _, y, _, _ = load_data("dataset/")
    rng = np.random.default_rng(0)
    if N_ROWS < len(y):
        keep = rng.choice(len(y), size=N_ROWS, replace=False)
        x, y = x[keep], y[keep]
    y01 = (y == 1).astype(float)

    results = []
    for trial in range(N_TRIALS):
        params = (
            DEFAULT
            if trial == 0
            else {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
        )
        params = {
            k: (int(v) if k == "MAX_DEPTH" else float(v)) for k, v in params.items()
        }
        apply(params)
        print(
            f"\n=== trial {trial + 1}/{N_TRIALS}: {params}, max trees {boosting.N_TREES}"
        )
        start = time.time()
        n_trees, thr, f1, f1_std, acc = boosting.cross_validate(x, y01)
        results.append(
            {
                **params,
                "n_trees": int(n_trees),
                "threshold": float(thr),
                "f1": float(f1),
                "f1_std": float(f1_std),
                "accuracy": float(acc),
                "hit_tree_limit": bool(n_trees == boosting.N_TREES),
                "seconds": round(time.time() - start),
            }
        )
        with open("results/tuning_results.json", "w") as f:
            json.dump(results, f, indent=2)

    results.sort(key=lambda r: -r["f1"])
    print("\nTop 5 settings:")
    for r in results[:5]:
        print(
            f"  F1 {r['f1']:.4f} +/- {r['f1_std']:.4f} | depth {r['MAX_DEPTH']}, "
            f"lr {r['LEARNING_RATE']}, lambda {r['REG_LAMBDA']}, "
            f"min_hess {r['MIN_CHILD_HESS']}, rows {r['ROW_SUBSAMPLE']}, "
            f"cols {r['COL_SUBSAMPLE']} | {r['n_trees']} trees, thr {r['threshold']:.2f}"
            + ("  (hit tree limit)" if r["hit_tree_limit"] else "")
        )
