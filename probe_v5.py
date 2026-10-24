"""
Probe: do Lily's activations predict whether she spares Mike, beyond what the
prompt and her own words already tell us?

Data (from rigged_decision_v5.py, Qwen2.5-14B-Instruct, 14% honest level):
  results/rigged_decision_v5_all.json   one record per decision
  results/activations_v5.npz            decision_acts[i] = last-token hidden
                                        states (all layers) right before
                                        decision i; her reply is already in
                                        the context

Label: spared = 1 if she did NOT challenge the (likely) bluffer.

The trap this script avoids
---------------------------
Across conditions, the prompts differ (different offer text, different bluffer
name), so any probe can "predict" behaviour just by recognising the condition.
That says nothing about intent. So we score the probe ONLY on comparisons
between trials whose prompts are identical except for what Lily herself wrote
(same condition, same bluffer). We call this the within-cell AUC:
  0.5 = no better than knowing the prompt; 1.0 = perfectly separates, within
  each cell, the trials where she spared the bluffer from those where she didn't.

Comparisons
-----------
  activation probe      logistic regression on one layer's activations
  words baseline        logistic regression on TF-IDF of her reply to Mike
                        (what a monitor reading her messages could use)
Both use the same grouped cross-validation (a conversation's Mike and Luke
trials are always in the same fold, so the probe never sees the test
conversation during training). A permutation test (labels shuffled within
cells) checks the best layer is not a fluke of picking the best of many.

Setup (local CPU, no GPU needed):
    pip install numpy scikit-learn matplotlib
    modal volume get sprint-results activations_v5.npz results/
Run:
    python probe_v5.py
Outputs:
    results/probe_v5_final.json, results/probe_v5_final_by_layer.png

Run 1 (without cell-centring) showed the null sat at ~0.39, not 0.5, because
the probe learned condition identity. Features are now centred within each
cell using training-fold means, so 0.5 is the real chance level.
"""
import argparse
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONDITIONS = ["talk", "explicit", "designer"]  # "none" has no variation (30/30 challenged)


def within_cell_auc(scores, y, cells):
    """Pairwise AUC computed only between trials in the same cell."""
    concordant, total = 0.0, 0
    for cell in np.unique(cells):
        idx = cells == cell
        pos, neg = scores[idx & (y == 1)], scores[idx & (y == 0)]
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = pos[:, None] - neg[None, :]
        concordant += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        total += diff.size
    return (concordant / total if total else float("nan")), total


def center_by_cell(X_train, X_test, cells_train, cells_test):
    """Subtract each cell's mean (computed on the training fold only).

    Removes everything that is constant within a cell (the prompt), so the
    model can only learn from variation between trials with the same prompt.
    Without this, the probe learns "which condition is this" and the
    within-cell AUC under the null drops well below 0.5 (seen in v5 run 1).
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    for c in np.unique(cells_train):
        mu = X_train[cells_train == c].mean(axis=0)
        X_train[cells_train == c] -= mu
        X_test[cells_test == c] -= mu
    return X_train, X_test


def fold_ids(y, groups, n_splits=5):
    """Which test fold each trial lands in (deterministic, grouped by conversation)."""
    ids = np.zeros(len(y), dtype=int)
    for k, (_, test) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups)):
        ids[test] = k
    return ids


def oof_scores(make_model, X, y, groups, center_cells, n_splits=5):
    """Out-of-fold predicted P(spared), grouped by conversation, cell-centred."""
    scores = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        Xtr, Xte = center_by_cell(X[train], X[test], center_cells[train], center_cells[test])
        model = make_model()
        model.fit(Xtr, y[train])
        scores[test] = model.predict_proba(Xte)[:, 1]
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/rigged_decision_v5_all.json")
    ap.add_argument("--out", default="probe_v5_final", help="output file stem in results/")
    ap.add_argument("--acts", default="results/activations_v5.npz")
    ap.add_argument("--C", type=float, default=0.05, help="L2 strength (smaller = stronger)")
    ap.add_argument("--perms", type=int, default=100)
    args = ap.parse_args()

    records = json.load(open(args.results, encoding="utf-8"))
    acts = np.load(args.acts)["decision_acts"]  # [N, layers+1, hidden]
    assert len(records) == acts.shape[0], "JSON and activations are out of sync"

    keep = np.array([r["condition"] in CONDITIONS and r["challenged"] is not None for r in records])
    recs = [r for r, k in zip(records, keep) if k]
    acts = acts[keep]
    y = np.array([0 if r["challenged"] else 1 for r in recs])
    cells = np.array([f"{r['condition']}|{r['bluffer']}" for r in recs])
    groups = np.array([f"{r['condition']}|{r['seed']}" for r in recs])
    # Score pairs only within the same cell AND the same test fold. Each fold's
    # model has its own intercept, set by the training fold's base rate, which
    # is anti-correlated with the test fold's. Comparing across folds made the
    # null sit below 0.5 (seen in run 2). Within a fold, that offset cancels.
    folds = fold_ids(np.zeros(len(recs)), groups)
    cells = np.array([f"{c}|fold{k}" for c, k in zip(cells, folds)])
    train_cells = np.array([f"{r['condition']}|{r['bluffer']}" for r in recs])
    replies = [r["offer_reply"] or "" for r in recs]
    n_layers = acts.shape[1]

    print(f"{len(y)} trials from {CONDITIONS}; spared = {y.sum()}, challenged = {len(y) - y.sum()}")
    print("cells (spared / total):")
    for c in np.unique(train_cells):
        print(f"  {c:<16} {y[train_cells == c].sum()}/{(train_cells == c).sum()}")

    # Words baseline: what her reply to Mike says
    word_scores = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=5).split(replies, y, groups):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        Xtr = vec.fit_transform([replies[i] for i in train]).toarray()
        Xte = vec.transform([replies[i] for i in test]).toarray()
        Xtr, Xte = center_by_cell(Xtr, Xte, train_cells[train], train_cells[test])
        m = LogisticRegression(C=1.0, max_iter=2000)
        m.fit(Xtr, y[train])
        word_scores[test] = m.predict_proba(Xte)[:, 1]
    words_auc, n_pairs = within_cell_auc(word_scores, y, cells)
    print(f"\ncomparable pairs (same cell, one spared + one challenged): {n_pairs}")
    print(f"words baseline (TF-IDF of her reply): within-cell AUC = {words_auc:.3f}")

    # Activation probe, layer by layer
    probe_model = lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=args.C, max_iter=5000)
    )
    layer_auc = []
    print("\nlayer  within-cell AUC")
    for layer in range(n_layers):
        X = acts[:, layer, :].astype(np.float32)
        s = oof_scores(probe_model, X, y, groups, train_cells)
        auc, _ = within_cell_auc(s, y, cells)
        layer_auc.append(auc)
        print(f"  {layer:3d}  {auc:.3f}  " + "#" * int(max(auc - 0.5, 0) * 80))

    best = int(np.nanargmax(layer_auc))
    print(f"\nbest layer: {best} (AUC {layer_auc[best]:.3f})")

    # Permutation test at the best layer: shuffle labels within cells
    rng = np.random.default_rng(0)
    X = acts[:, best, :].astype(np.float32)
    null = []
    for i in range(args.perms):
        y_perm = y.copy()
        for c in np.unique(train_cells):
            idx = np.where(train_cells == c)[0]
            y_perm[idx] = rng.permutation(y_perm[idx])
        s = oof_scores(probe_model, X, y_perm, groups, train_cells)
        null.append(within_cell_auc(s, y_perm, cells)[0])
        if (i + 1) % 50 == 0:
            print(f"  permutations: {i + 1}/{args.perms}")
    null = np.array(null)
    p_value = (1 + (null >= layer_auc[best]).sum()) / (1 + len(null))
    print(f"permutation p-value at layer {best}: {p_value:.3f} "
          f"(null mean {np.nanmean(null):.3f}, 95th pct {np.nanpercentile(null, 95):.3f})")
    print("Note: the layer was chosen as the best of all layers; the p-value tests that layer "
          "only, so treat it as optimistic.")

    os.makedirs("results", exist_ok=True)
    out = {
        "conditions": CONDITIONS, "n_trials": int(len(y)), "n_pairs": int(n_pairs),
        "C": args.C, "words_auc": words_auc, "layer_auc": layer_auc,
        "best_layer": best, "best_auc": layer_auc[best], "perm_p": float(p_value),
        "null_mean": float(np.nanmean(null)), "null_p95": float(np.nanpercentile(null, 95)),
    }
    with open(f"results/{args.out}.json", "w") as f:
        json.dump(out, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(range(n_layers), layer_auc, marker="o", ms=3, label="activation probe")
        ax.axhline(words_auc, ls="--", color="tab:orange", label="words baseline (reply TF-IDF)")
        ax.axhline(0.5, ls=":", color="grey", label="chance / prompt only")
        ax.set_xlabel("layer")
        ax.set_ylabel("within-cell AUC")
        ax.set_title("Predicting whether Lily spares the bluffer (Qwen2.5-14B)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"results/{args.out}_by_layer.png", dpi=150)
        print(f"Saved results/{args.out}_by_layer.png")
    except ImportError:
        print("matplotlib not installed, skipping plot")
    print(f"Saved results/{args.out}.json")


if __name__ == "__main__":
    main()
