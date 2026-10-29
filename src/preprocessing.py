import os
import numpy as np
import matplotlib.pyplot as plt
from src.helpers import load_csv_data


MISSING_CODES = {9: 7, 99: 77, 999: 777, 9999: 7777, 99999: 77777}
# In questions coded up to 99 (or 999), 88 (or 888) usually means "none".
NONE_CODES = {99: 88, 999: 888}


def load_data(data_path="dataset/"):
    """Load the CSVs once, then reuse a fast .npz cache on later runs."""
    cache = os.path.join(data_path, "cache.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["x_train"], d["x_test"], d["y_train"], d["train_ids"], d["test_ids"]
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
    np.savez(cache, x_train=x_train, x_test=x_test, y_train=y_train,
             train_ids=train_ids, test_ids=test_ids)
    return x_train, x_test, y_train, train_ids, test_ids


def find_code_cols(x):
    """Find columns whose maximum is a BRFSS 'refused' code (9, 99, 999, ...).

    Returns {column index: (refused code, don't-know code)}.
    """
    code_cols = {}
    for j in range(x.shape[1]):
        values = x[~np.isnan(x[:, j]), j]
        if values.size and values.max() in MISSING_CODES:
            code_cols[j] = (values.max(), MISSING_CODES[values.max()])
    return code_cols


def replace_missing_codes(x, code_cols):
    """Return a copy of x where the 'refused'/'don't know' codes become NaN."""
    x = x.copy()
    for j, (refused, dont_know) in code_cols.items():
        col = x[:, j]  # a view, so editing col edits x
        col[(col == refused) | (col == dont_know)] = np.nan
    return x


def replace_none_codes(x, none_cols):
    """In place: replace the 'none' code (88/888) by 0 in the given columns."""
    for j, none in none_cols.items():
        col = x[:, j]
        col[col == none] = 0


def find_categories(x, max_categories, min_freq):
    """Find categorical columns and their frequent values.

    A column counts as categorical if it has between 3 and `max_categories`
    distinct values (binary columns need no one-hot: standardizing them is
    equivalent). Only values present in at least `min_freq` of the rows get
    their own column, to keep the number of new columns small.
    Returns {column index: array of frequent values}.
    """
    cats = {}
    for j in range(x.shape[1]):
        col = x[:, j]
        values, counts = np.unique(col[~np.isnan(col)], return_counts=True)
        if 3 <= len(values) <= max_categories:
            frequent = values[counts >= min_freq * len(col)]
            if len(frequent) > 0:
                cats[j] = frequent
    return cats


def expand_features(x, cats, ind_cols):
    """Build the feature matrix: numeric columns, one-hot columns, indicators.

    Numeric (non-categorical) columns come first, unchanged (NaN stays NaN).
    Each categorical column is replaced by one 0/1 column per frequent value;
    a missing or rare value gives 0 in all of them. Finally, one 0/1 column
    per index in `ind_cols` says whether that value was missing.
    """
    num_cols = [j for j in range(x.shape[1]) if j not in cats]
    n_new = sum(len(values) for values in cats.values()) + len(ind_cols)
    out = np.empty((x.shape[0], len(num_cols) + n_new))
    out[:, :len(num_cols)] = x[:, num_cols]
    c = len(num_cols)
    for j, values in cats.items():
        for v in values:
            out[:, c] = x[:, j] == v  # NaN == v is False, so missing -> 0
            c += 1
    for j in ind_cols:
        out[:, c] = np.isnan(x[:, j])
        c += 1
    return out


def fill_missing(x, medians):
    """Replace NaN by the column median, in place (saves memory)."""
    for j in range(x.shape[1]):
        col = x[:, j]
        col[np.isnan(col)] = medians[j]


def fit_preprocessing(x, max_missing=0.8, one_hot=False, none_codes=False,
                      missing_ind=False, max_categories=10, min_freq=0.01,
                      min_missing_ind=0.05):
    """Learn all preprocessing parameters from training data x.

    Steps: (1) detect columns using BRFSS missing codes, optionally turn the
    'none' codes (88/888) into 0, (2) drop columns with more than
    `max_missing` missing values, (3) optionally one-hot encode categorical
    columns and add "was missing" indicators for non-categorical columns with
    at least `min_missing_ind` missing values, (4) store medians for
    imputation, (5) store mean/std for standardization, drop constant columns.
    """
    # 1. Columns whose maximum value is a "refused" code.
    code_cols = find_code_cols(x)
    none_cols = {}
    if none_codes:
        for j, (refused, _) in code_cols.items():
            if refused in NONE_CODES and np.any(x[:, j] == NONE_CODES[refused]):
                none_cols[j] = NONE_CODES[refused]
    x = replace_missing_codes(x, code_cols)
    replace_none_codes(x, none_cols)

    # 2. Drop mostly-missing columns.
    keep = np.isnan(x).mean(axis=0) <= max_missing
    x = x[:, keep]

    # 3. Optional one-hot encoding and "was missing" indicators.
    cats = find_categories(x, max_categories, min_freq) if one_hot else {}
    ind_cols = []
    if missing_ind:
        frac = np.isnan(x).mean(axis=0)
        ind_cols = [j for j in range(x.shape[1])
                    if j not in cats and frac[j] >= min_missing_ind]
    if cats or ind_cols:
        x = expand_features(x, cats, ind_cols)

    # 4. Median imputation.
    medians = np.nanmedian(x, axis=0)
    fill_missing(x, medians)

    # 5. Standardization; constant columns carry no information.
    std = x.std(axis=0)
    nonconst = std > 0
    return {
        "code_cols": code_cols,
        "none_cols": none_cols,
        "keep": keep,
        "cats": cats,
        "ind_cols": ind_cols,
        "medians": medians,
        "nonconst": nonconst,
        "mean": x[:, nonconst].mean(axis=0),
        "std": std[nonconst],
    }


def transform(x, params):
    """Apply learned preprocessing to x and add a bias (intercept) column."""
    x = replace_missing_codes(x, params["code_cols"])
    replace_none_codes(x, params["none_cols"])
    x = x[:, params["keep"]]
    if params["cats"] or params["ind_cols"]:
        x = expand_features(x, params["cats"], params["ind_cols"])
    fill_missing(x, params["medians"])
    if not params["nonconst"].all():
        x = x[:, params["nonconst"]]
    x -= params["mean"]
    x /= params["std"]
    return np.c_[np.ones(len(x)), x]