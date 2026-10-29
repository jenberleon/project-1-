"""
Usage:
  python -m src.boosting          # CV, then trains on all data, writes submission_boosting.csv
  python -m src.boosting --quick  # first 50,000 rows only, no submission
"""
import sys
import numpy as np
from src.helpers import create_csv_submission
from src.preprocessing import load_data, find_code_cols, replace_missing_codes
from src.baseline import SEED, K_FOLDS, THRESHOLDS, sigmoid, f1_score, accuracy

N_BINS = 32  # bins per feature, bin 0 = missing
N_TREES = 250  # maximum number of trees
CHECK_EVERY = 50  # evaluate on the validation fold every 50 trees
MAX_DEPTH = 5  # depth of each tree
LEARNING_RATE = 0.1  # shrinks each tree's correction (smaller = more careful)
REG_LAMBDA = 0.1  # L2 penalty on leaf values
MIN_CHILD_HESS = 2.0  # minimum Hessian sum per leaf (roughly a minimum leaf size)
ROW_SUBSAMPLE = 0.9  # fraction of rows used to build each tree
COL_SUBSAMPLE = 0.5  # fraction of features considered by each tree
QUICK = "--quick" in sys.argv


# ---------------------------------------------------------------- binning
def fit_bins(x):
    """
    Quantile bin edges for each feature, from training data only.

    """
    quantiles = np.linspace(0, 1, N_BINS)[1:-1]
    edges = []
    for j in range(x.shape[1]):
        col = x[~np.isnan(x[:, j]), j]
        edges.append(
            np.unique(np.quantile(col, quantiles)) if col.size else np.array([])
        )
    return edges


def apply_bins(x, edges):
    """Replace values by bin indices (uint8). Column-major for fast column access."""
    xb = np.zeros(x.shape, dtype=np.uint8, order="F")
    for j, e in enumerate(edges):
        col = x[:, j]
        ok = ~np.isnan(col)
        xb[ok, j] = 1 + np.searchsorted(e, col[ok], side="right")
    return xb


def prepare(x_train, others):
    """Learn missing codes and bins on x_train; bin x_train and each x in others."""
    code_cols = find_code_cols(x_train)
    x_train = replace_missing_codes(x_train, code_cols)
    edges = fit_bins(x_train)
    binned = [apply_bins(x_train, edges)]
    for x in others:
        binned.append(apply_bins(replace_missing_codes(x, code_cols), edges))
    return binned


# ---------------------------------------------------------------- one tree
def best_split(xb, g, h, idx, features, G, H):
    """Best (feature, threshold bin, missing goes left) for the rows idx, or None.

    Rows with bin <= threshold go left; missing values (bin 0) go to the side
    that gives the larger gain. The gain is the usual second-order one:
    GL^2/(HL+lambda) + GR^2/(HR+lambda) - G^2/(H+lambda).
    """
    gi, hi = g[idx], h[idx]
    parent = G * G / (H + REG_LAMBDA)
    best_gain, best = 0.0, None
    for j in features:
        b = xb[idx, j]
        Gb = np.bincount(b, weights=gi, minlength=N_BINS)
        Hb = np.bincount(b, weights=hi, minlength=N_BINS)
        GL0, HL0 = np.cumsum(Gb[1:-1]), np.cumsum(Hb[1:-1])  # thresholds 1..N_BINS-2
        for missing_left in (False, True):
            gl = GL0 + Gb[0] * missing_left
            hl = HL0 + Hb[0] * missing_left
            gr, hr = G - gl, H - hl
            ok = (hl >= MIN_CHILD_HESS) & (hr >= MIN_CHILD_HESS)
            if not ok.any():
                continue
            gain = np.where(
                ok,
                gl**2 / (hl + REG_LAMBDA) + gr**2 / (hr + REG_LAMBDA) - parent,
                -np.inf,
            )
            t = np.argmax(gain)
            if gain[t] > best_gain:
                best_gain, best = gain[t], (j, t + 1, missing_left)
    return best


def build_tree(xb, g, h, rows, features):
    """Grow one tree level by level. Returns it as a dict of arrays."""
    tree = {
        "feat": [],
        "thr": [],
        "miss_left": [],
        "left": [],
        "right": [],
        "value": [],
    }

    def new_node():
        for key, default in (
            ("feat", -1),
            ("thr", 0),
            ("miss_left", False),
            ("left", -1),
            ("right", -1),
            ("value", 0.0),
        ):
            tree[key].append(default)
        return len(tree["feat"]) - 1

    frontier = [(new_node(), rows)]
    for depth in range(MAX_DEPTH + 1):
        next_frontier = []
        for node, idx in frontier:
            G, H = g[idx].sum(), h[idx].sum()
            tree["value"][node] = -G / (H + REG_LAMBDA)  # optimal leaf value
            if depth == MAX_DEPTH:
                continue
            split = best_split(xb, g, h, idx, features, G, H)
            if split is None:
                continue
            j, t, missing_left = split
            b = xb[idx, j]
            go_left = np.where(b == 0, missing_left, b <= t)
            left, right = new_node(), new_node()
            tree["feat"][node], tree["thr"][node] = j, t
            tree["miss_left"][node] = missing_left
            tree["left"][node], tree["right"][node] = left, right
            next_frontier += [(left, idx[go_left]), (right, idx[~go_left])]
        frontier = next_frontier
        if not frontier:
            break
    return {key: np.array(values) for key, values in tree.items()}


def predict_tree(tree, xb):
    """Log-odds correction given by one tree for every row of xb."""
    node = np.zeros(len(xb), dtype=int)
    rows = np.arange(len(xb))
    for _ in range(MAX_DEPTH):
        feat = tree["feat"][node]
        internal = feat >= 0
        if not internal.any():
            break
        b = xb[rows, np.maximum(feat, 0)]
        go_left = np.where(b == 0, tree["miss_left"][node], b <= tree["thr"][node])
        child = np.where(go_left, tree["left"][node], tree["right"][node])
        node = np.where(internal, child, node)
    return tree["value"][node]


# ---------------------------------------------------------------- boosting
def fit_boosting(xb, y, n_trees, xb_val=None):
    """Train n_trees trees on (xb, y), y in {0, 1}.

    If xb_val is given, also returns the validation probabilities after every
    CHECK_EVERY trees, so CV can choose the best number of trees.
    """
    rng = np.random.default_rng(SEED)
    base = np.log(y.mean() / (1 - y.mean()))  # starting log-odds
    F = np.full(len(y), base)
    F_val = None if xb_val is None else np.full(len(xb_val), base)
    n_cols = max(1, int(COL_SUBSAMPLE * xb.shape[1]))
    trees, val_probas = [], {}

    for m in range(1, n_trees + 1):
        p = sigmoid(F)
        g, h = p - y, p * (1 - p)  # gradient and Hessian of the logistic loss
        rows = np.flatnonzero(rng.random(len(y)) < ROW_SUBSAMPLE)
        features = rng.choice(xb.shape[1], size=n_cols, replace=False)
        tree = build_tree(xb, g, h, rows, features)
        tree["value"] = tree["value"] * LEARNING_RATE
        trees.append(tree)
        F += predict_tree(tree, xb)
        if F_val is not None:
            F_val += predict_tree(tree, xb_val)
        if m % CHECK_EVERY == 0:
            p = np.clip(sigmoid(F), 1e-12, 1 - 1e-12)
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            print(f"    {m} trees, training loss {loss:.4f}")
            if F_val is not None:
                val_probas[m] = sigmoid(F_val)
    return base, trees, val_probas


def predict_proba(base, trees, xb):
    F = np.full(len(xb), base)
    for tree in trees:
        F += predict_tree(tree, xb)
    return sigmoid(F)


def cross_validate(x, y01):
    """K-fold CV over the number of trees and the threshold (best mean F1)."""
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(len(y01)), K_FOLDS)
    checkpoints = list(range(CHECK_EVERY, N_TREES + 1, CHECK_EVERY))
    f1 = np.zeros((K_FOLDS, len(checkpoints), len(THRESHOLDS)))
    acc = np.zeros_like(f1)

    for k, val_idx in enumerate(folds):
        print(f"fold {k + 1}/{K_FOLDS}")
        train_idx = np.concatenate([f for i, f in enumerate(folds) if i != k])
        xb_tr, xb_val = prepare(x[train_idx], [x[val_idx]])
        _, _, val_probas = fit_boosting(xb_tr, y01[train_idx], N_TREES, xb_val)
        y_val = y01[val_idx]
        for c, m in enumerate(checkpoints):
            for t, thr in enumerate(THRESHOLDS):
                pred = (val_probas[m] >= thr).astype(int)
                f1[k, c, t] = f1_score(y_val, pred)
                acc[k, c, t] = accuracy(y_val, pred)
        print(f"fold {k + 1}/{K_FOLDS}: best F1 = {f1[k].max():.4f}")

    mean_f1 = f1.mean(axis=0)
    c, t = np.unravel_index(np.argmax(mean_f1), mean_f1.shape)
    print("\nCV F1 (best threshold) by number of trees:")
    for i, m in enumerate(checkpoints):
        print(f"  {m:>4} trees: {mean_f1[i].max():.4f}")
    return (
        checkpoints[c],
        THRESHOLDS[t],
        mean_f1[c, t],
        f1[:, c, t].std(),
        acc[:, c, t].mean(),
    )


if __name__ == "__main__":
    x_train, x_test, y_train, train_ids, test_ids = load_data("dataset/")
    if QUICK:
        x_train, y_train = x_train[:50000], y_train[:50000]
    print(
        f"Settings: quick={QUICK}, trees<={N_TREES}, depth={MAX_DEPTH}, "
        f"learning rate={LEARNING_RATE}"
    )
    y01 = (y_train == 1).astype(float)  # labels {-1, 1} -> {0, 1}

    n_trees, threshold, f1_mean, f1_std, acc_mean = cross_validate(x_train, y01)
    print(
        f"\nCV F1 = {f1_mean:.4f} +/- {f1_std:.4f} with {n_trees} trees, "
        f"threshold {threshold:.2f}"
    )
    print(f"CV accuracy at that threshold = {acc_mean:.4f}")
    print(f"(always predicting 'no' would give accuracy {1 - y01.mean():.4f}, F1 0)")

    if QUICK:
        print("Quick mode: no submission written (run without --quick for that)")
    else:
        print(f"\nTraining final model on all data with {n_trees} trees")
        xb_train, xb_test = prepare(x_train, [x_test])
        base, trees, _ = fit_boosting(xb_train, y01, n_trees)
        y_pred = np.where(predict_proba(base, trees, xb_test) >= threshold, 1, -1)
        create_csv_submission(test_ids, y_pred, "submissions/submission_boosting.csv")
        print("Wrote submissions/submission_boosting.csv")
