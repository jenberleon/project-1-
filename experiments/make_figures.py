"""Make the report figures and the LaTeX results table from experiments.py.

Usage:
  python -m experiments.make_figures           # reads results/results.json, writes into results/
  python -m experiments.make_figures --quick   # reads results/results_quick.json, writes into results/quick/
Only the figures whose stages have been run are made. Figures are sized for one
column of the two-column LaTeX template: \\includegraphics[width=\\columnwidth]{...}
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no window needed, just files
import matplotlib.pyplot as plt

QUICK = "--quick" in sys.argv
RESULTS_FILE = os.path.join("results", "results_quick.json" if QUICK else "results.json")
OUT = os.path.join("results", "quick") if QUICK else "results"
plt.rcParams.update({"font.size": 8, "figure.figsize": (3.4, 2.2),
                     "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def fmt_time(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.1f} s" if seconds < 60 else f"{seconds / 60:.1f} min"


def lam_tex(lam):
    return f"$\\lambda=10^{{{round(np.log10(lam))}}}$"


def best_of(rows, family):
    """(name, row) of the best model of a family, or None."""
    cands = [(n, r) for n, r in rows.items() if r["family"] == family and r["summary"]]
    return max(cands, key=lambda nr: nr[1]["summary"]["f1"]) if cands else None


def best_linear_f1(res):
    f1s = [r["summary"]["f1"] for stage in ("models", "ablation") if stage in res
           for r in res[stage]["rows"].values() if r["summary"]]
    return max(f1s) if f1s else None


# ---------------------------------------------------------------- figures
def fig_lambda(res):
    """CV F1 against lambda: how lambda was chosen."""
    rows = res["models"]["rows"]
    fig, ax = plt.subplots()
    for family, marker in (("ridge", "s"), ("reg. logistic", "o")):
        pts = sorted((r["lambda"], r["summary"]["f1"], r["summary"]["f1_std"])
                     for r in rows.values() if r["family"] == family and r["summary"])
        if pts:
            lam, f1, std = zip(*pts)
            ax.errorbar(lam, f1, yerr=std, marker=marker, capsize=2, label=family)
    for name, style in (("least squares", ":"), ("logistic", "--")):
        r = rows.get(name)
        if r and r["summary"]:
            ax.axhline(r["summary"]["f1"], ls=style, color="gray", lw=1,
                       label=f"{name} (no $\\lambda$)")
    ax.set_xscale("log")
    ax.set_xlabel("regularization $\\lambda$")
    ax.set_ylabel("CV F1 (best threshold)")
    ax.legend(fontsize=6, loc="best")
    ax.margins(y=0.25)  # room for the legend
    save(fig, "fig_lambda.pdf")


def fig_threshold(res, thresholds):
    """Mean CV F1 against the decision threshold: why 0.5 is a bad default."""
    curves = []
    if "models" in res:
        rows = res["models"]["rows"]
        for family in ("least squares", "ridge", "logistic", "reg. logistic"):
            b = best_of(rows, family)
            if b:
                curves.append((b[0], np.nanmean(np.array(b[1]["f1"], dtype=float), axis=0)))
    if "boosting" in res:
        b = res["boosting"]
        c = b["checkpoints"].index(b["best_trees"])
        curves.append((f"boosting ({b['best_trees']} trees)",
                       np.array(b["f1"])[:, c].mean(axis=0)))
    if not curves:
        return
    fig, ax = plt.subplots()
    for name, f1 in curves:
        ax.plot(thresholds, f1, label=name, lw=1.2)
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.text(0.51, ax.get_ylim()[1] * 0.95, "default 0.5", fontsize=6, color="gray")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("CV F1")
    ax.legend(fontsize=6)
    save(fig, "fig_threshold.pdf")


def fig_ablation(res):
    """CV F1 of reg. logistic for each preprocessing setting."""
    a = res["ablation"]
    items = [(f"{n} ({r['n_features']} feat.)", r["summary"]["f1"], r["summary"]["f1_std"])
             for n, r in a["rows"].items() if r["summary"]]
    if not items:
        return
    names, f1, std = zip(*items)
    pos = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(3.4, 0.28 * len(names) + 0.7))
    ax.errorbar(f1, pos, xerr=std, fmt="o", capsize=2)
    for p, v in zip(pos, f1):
        ax.annotate(f"{v:.3f}", (v, p), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=6)
    if "boosting" in res:
        v = res["boosting"]["summary"]["f1"]
        ax.axvline(v, ls="--", color="tab:red", lw=1)
        ax.text(v, len(names) - 0.4, " boosting", color="tab:red", fontsize=6, va="bottom")
    ax.set_yticks(pos)
    ax.set_yticklabels(names)
    ax.set_ylim(-0.7, len(names) - 0.1)
    ax.set_xlabel(f"CV F1 of reg. logistic, {lam_tex(a['lambda'])} (best threshold)")
    save(fig, "fig_ablation.pdf")


def fig_gamma(res):
    """Training loss per iteration for each step size: how gamma was chosen."""
    g = res["gamma"]
    fig, ax = plt.subplots()
    finite = []
    for gamma, c in g["curves"].items():
        loss = np.array(c["loss"], dtype=float)
        ax.plot(np.arange(1, len(loss) + 1), loss, lw=1.2,
                label=f"$\\gamma$={gamma} (val. F1 {c['val_f1']:.3f})")
        finite += list(loss[np.isfinite(loss)])
    ax.axvline(g["max_iters"], color="gray", ls=":", lw=1)
    ax.text(g["max_iters"] * 1.02, np.log(2), f"{g['max_iters']} it. used", fontsize=6,
            color="gray", va="top")
    if finite:  # diverging step sizes would otherwise squash the plot
        ax.set_ylim(min(finite) - 0.01, np.log(2) + 0.03)
    ax.set_xlabel("iteration")
    ax.set_ylabel("training log-loss")
    ax.legend(fontsize=6)
    save(fig, "fig_gamma.pdf")


def fig_boosting(res):
    """CV F1 and validation log-loss against the number of trees."""
    b = res["boosting"]
    m = b["checkpoints"]
    f1 = [s["f1"] for s in b["per_trees"]]
    std = [s["f1_std"] for s in b["per_trees"]]
    ll = np.array(b["val_logloss"])
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(3.4, 3.0))
    a1.errorbar(m, f1, yerr=std, marker="o", ms=3, capsize=2, label="boosting")
    lin = best_linear_f1(res)
    if lin is not None:
        a1.axhline(lin, ls="--", color="gray", lw=1, label="best linear model")
    a1.axvline(b["best_trees"], ls=":", color="tab:red", lw=1)
    a1.set_ylabel("CV F1")
    a1.legend(fontsize=6)
    a2.errorbar(m, ll.mean(axis=0), yerr=ll.std(axis=0), marker="o", ms=3, capsize=2)
    a2.set_ylabel("validation log-loss")
    a2.set_xlabel("number of trees")
    save(fig, "fig_boosting.pdf")


def fig_time(res):
    """Training time per fold, log scale."""
    items = []
    if "models" in res:
        rows = res["models"]["rows"]
        first = next(iter(rows.values()))
        items.append(("preprocessing", np.mean(first["prep_time"])))
        for family in ("least squares", "ridge", "logistic", "reg. logistic"):
            t = [np.nanmean(r["time"]) for r in rows.values() if r["family"] == family]
            if t:
                items.append((family, float(np.mean(t))))
    if "boosting" in res:
        items.append(("boosting (incl. binning)", float(np.mean(res["boosting"]["time"]))))
    if not items:
        return
    names, secs = zip(*items)
    pos = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(3.4, 0.28 * len(names) + 0.7))
    ax.barh(pos, secs, color="tab:blue")
    for p, s in zip(pos, secs):
        ax.text(s * 1.1, p, fmt_time(s), va="center", fontsize=6)
    ax.set_xscale("log")
    ax.set_xlim(min(secs) / 2, max(secs) * 8)
    ax.set_yticks(pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("training time per fold (s, log scale)")
    save(fig, "fig_time.pdf")


# ---------------------------------------------------------------- table
def write_table(res):
    """Main results table as LaTeX, one row per method (best lambda only)."""
    lines = [r"\begin{tabular}{lcccc}", r"\hline",
             r"Model & CV F1 & Acc. & Thr. & Time/fold \\", r"\hline"]

    def row(label, r=None, s=None, seconds=None):
        s = s or (r["summary"] if r else None)
        if s is None:
            return
        if seconds is None and r is not None:
            seconds = np.nanmean(r["time"])
        t = fmt_time(seconds) if seconds is not None else "--"
        lines.append(f"{label} & {s['f1']:.3f} $\\pm$ {s['f1_std']:.3f} & {s['acc']:.3f} "
                     f"& {s['threshold']:.2f} & {t} \\\\")

    rate = res["info"]["positive_rate"]
    lines.append(f"Always ``no'' & 0 & {1 - rate:.3f} & -- & -- \\\\")
    if "models" in res:
        rows = res["models"]["rows"]
        row("Least squares", rows.get("least squares"))
        b = best_of(rows, "ridge")
        if b:
            row(f"Ridge, {lam_tex(b[1]['lambda'])}", b[1])
        row(f"Logistic, $\\gamma={res['info']['gamma']:g}$", rows.get("logistic"))
        b = best_of(rows, "reg. logistic")
        if b:
            row(f"Reg. logistic, {lam_tex(b[1]['lambda'])}", b[1])
    if "ablation" in res:
        lines.append(r"\hline")
        for name, r in res["ablation"]["rows"].items():
            if name != "plain":
                row(f"\\quad Reg. logistic {name}", r)
    if "boosting" in res:
        lines.append(r"\hline")
        b = res["boosting"]
        row(f"Boosted trees, {b['best_trees']} trees", s=b["summary"],
            seconds=float(np.mean(b["time"])))
    lines += [r"\hline", r"\end{tabular}"]
    path = os.path.join(OUT, "table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    if not os.path.exists(RESULTS_FILE):
        raise SystemExit(f"{RESULTS_FILE} not found: run experiments.py first")
    with open(RESULTS_FILE) as f:
        res = json.load(f)
    os.makedirs(OUT, exist_ok=True)
    thresholds = np.array(res["info"]["thresholds"])
    print(f"{RESULTS_FILE}: stages {[s for s in ('models', 'ablation', 'gamma', 'boosting') if s in res]}")
    if "models" in res:
        fig_lambda(res)
    fig_threshold(res, thresholds)
    if "ablation" in res:
        fig_ablation(res)
    if "gamma" in res:
        fig_gamma(res)
    if "boosting" in res:
        fig_boosting(res)
    fig_time(res)
    write_table(res)