"""
Drop-in replacement for the metrics / plotting helpers (same public names and signatures).

Conventions
-----------
* Scores may be torch tensors or numpy arrays.
* SCORES_ARE_LOGITS = True  -> scores are raw logits (the model is trained with a logit-based focal loss):
  a sigmoid is applied before any thresholded metric, so the 0.5 threshold really is p = 0.5.
  Rank metrics (AUC, AP, Hits@K, P@K) are unaffected.
* "+/-" returned by *_with_symmetrical_confidence is the HALF-WIDTH of a 95% percentile bootstrap interval over test pairs.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, auc, average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

SCORES_ARE_LOGITS = False       # train_link_prediction_v3 passes sigmoid probabilities. Set True if you pass raw logits.
THRESHOLD = 0.5      # modules read this at call time: set utils.THRESHOLD = <val-tuned threshold> before the thresholded metrics
BOOTSTRAP_SEED = 0


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _probs(x):
    x = _np(x)
    return _sigmoid(x) if SCORES_ARE_LOGITS else x


def _labels(pos, neg):
    return np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(int)


# --------------------------------------------------------------------------- point metrics (numpy)
def _auc(pos, neg):
    return roc_auc_score(_labels(pos, neg), np.r_[_np(pos), _np(neg)])


def _ap(pos, neg):                               # sklearn AP: tie-safe (the hand-written loop was optimistic on ties)
    return average_precision_score(_labels(pos, neg), np.r_[_np(pos), _np(neg)])


def _thr(fn, pos, neg, threshold=None):
    thr = THRESHOLD if threshold is None else threshold
    y = _labels(pos, neg)
    p = np.r_[_probs(pos), _probs(neg)]
    return fn(y, (p >= thr).astype(int))


def compute_auc(pos_score, neg_score):
    return _auc(pos_score, neg_score)


def compute_map(pos_score, neg_score):
    return _ap(pos_score, neg_score)


def compute_f1(pos_score, neg_score, threshold=None):
    return _thr(lambda y, h: f1_score(y, h, zero_division=0), pos_score, neg_score, threshold)


def compute_accuracy(pos_score, neg_score, threshold=None):
    return _thr(accuracy_score, pos_score, neg_score, threshold)


def compute_precision(pos_score, neg_score, threshold=None):
    return _thr(lambda y, h: precision_score(y, h, zero_division=0), pos_score, neg_score, threshold)   # 0, not 1, if nothing predicted positive


def compute_recall(pos_score, neg_score, threshold=None):
    return _thr(lambda y, h: recall_score(y, h, zero_division=0), pos_score, neg_score, threshold)


def compute_hits_k(pos_score, neg_score, k=10):
    """OGB-style Hits@K: share of positives scored strictly above the K-th highest NEGATIVE.
    (The old version returned the share of positives among the top-K of the pooled list, i.e. precision@K.)"""
    pos, neg = _np(pos_score), _np(neg_score)
    kth = np.sort(neg)[-min(k, len(neg))]
    return float((pos > kth).mean())


def compute_precision_at_k(pos_score, neg_score, k=100):
    s = np.r_[_np(pos_score), _np(neg_score)]
    y = _labels(pos_score, neg_score)
    order = np.argsort(-s, kind="mergesort")[:k]
    return float(y[order].mean())


def compute_focalloss(pos_score, neg_score, alpha=1, gamma=2):
    """Focal loss on LOGITS (numpy, stable BCE-with-logits)."""
    x = np.r_[_np(pos_score), _np(neg_score)]
    if not SCORES_ARE_LOGITS:                       # probabilities -> logits, so the loss is always computed on logits
        pc = np.clip(x, 1e-7, 1 - 1e-7); x = np.log(pc / (1 - pc))
    y = _labels(pos_score, neg_score)
    bce = np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))
    pt = np.exp(-bce)
    return float((alpha * (1 - pt) ** gamma * bce).mean())


# --------------------------------------------------------------------------- bootstrap
def bootstrap_confidence_interval(metric_func, pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95,
                                  seed=BOOTSTRAP_SEED):
    """Stratified bootstrap over test pairs; returns the half-width of the percentile interval. Seeded (old one was not)."""
    pos, neg = _np(pos_score), _np(neg_score)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_bootstraps):
        vals.append(metric_func(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]))
    a = (1 - confidence_level) / 2 * 100
    lo, hi = np.nanpercentile(vals, [a, 100 - a])
    return float((hi - lo) / 2)


def _with_ci(fn):
    def wrapper(pos_score, neg_score, n_bootstraps=1000, confidence_level=0.95):
        return fn(pos_score, neg_score), bootstrap_confidence_interval(fn, pos_score, neg_score, n_bootstraps, confidence_level)
    return wrapper


compute_auc_with_symmetrical_confidence = _with_ci(compute_auc)
compute_map_with_symmetrical_confidence = _with_ci(compute_map)
compute_f1_with_symmetrical_confidence = _with_ci(compute_f1)
compute_accuracy_with_symmetrical_confidence = _with_ci(compute_accuracy)
compute_precision_with_symmetrical_confidence = _with_ci(compute_precision)
compute_recall_with_symmetrical_confidence = _with_ci(compute_recall)


def compute_focalloss_with_symmetrical_confidence(pos_score, neg_score, alpha=1, gamma=2, n_bootstraps=1000, confidence_level=0.95):
    fn = lambda p, n: compute_focalloss(p, n, alpha, gamma)
    return fn(pos_score, neg_score), bootstrap_confidence_interval(fn, pos_score, neg_score, n_bootstraps, confidence_level)


# --------------------------------------------------------------------------- plots
def _finish(path, show=False):
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show() if show else None
    plt.close()


def plot_roc_curves(fold_results, output_path):
    """Per-fold ROC with the TRUE AUC in the legend (the old legend printed tpr.mean(), which is not an AUC)."""
    plt.figure(figsize=(8, 6))
    aucs = []
    for i, (y, s) in enumerate(fold_results):
        fpr, tpr, _ = roc_curve(y, s); aucs.append(auc(fpr, tpr))
        plt.plot(fpr, tpr, lw=1, label=f"Fold {i + 1} (AUC = {aucs[-1]:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=0.8, label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC (mean AUC = {np.mean(aucs):.4f} $\\pm$ {np.std(aucs, ddof=1):.4f} across folds)")
    plt.legend(loc="lower right", fontsize="small"); _finish(output_path)


def plot_pr_curves(fold_results, output_path_cross_pr):
    """Per-fold PR curves with average precision; the chance level is the positive rate (horizontal), not a diagonal."""
    plt.figure(figsize=(8, 6)); aps, prev = [], []
    for i, (y, s) in enumerate(fold_results):
        p, r, _ = precision_recall_curve(y, s); aps.append(average_precision_score(y, s)); prev.append(np.mean(y))
        plt.plot(r, p, lw=1, label=f"Fold {i + 1} (AP = {aps[-1]:.4f})")
    plt.axhline(np.mean(prev), ls="--", color="gray", lw=0.8, label=f"Chance (positive rate = {np.mean(prev):.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision-Recall (mean AP = {np.mean(aps):.4f} $\\pm$ {np.std(aps, ddof=1):.4f} across folds)")
    plt.legend(loc="lower left", fontsize="small"); _finish(output_path_cross_pr)
    print(f"Precision-Recall curves saved at: {output_path_cross_pr}")


def plot_roc_pr_curves(fold_results, output_path):
    """Side-by-side ROC and PR with the mean curve (interpolated) and +/-1 SD band."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    grid = np.linspace(0, 1, 201); tprs, precs, aucs, aps, prev = [], [], [], [], []
    for i, (y, s) in enumerate(fold_results):
        fpr, tpr, _ = roc_curve(y, s); aucs.append(auc(fpr, tpr))
        t = np.interp(grid, fpr, tpr); t[0] = 0.0; tprs.append(t)
        p, r, _ = precision_recall_curve(y, s); aps.append(average_precision_score(y, s)); prev.append(np.mean(y))
        precs.append(np.interp(grid, r[::-1], p[::-1]))
        ax[0].plot(fpr, tpr, lw=0.8, alpha=0.6, label=f"Fold {i + 1} (AUC = {aucs[-1]:.4f})")
        ax[1].plot(r, p, lw=0.8, alpha=0.6, label=f"Fold {i + 1} (AP = {aps[-1]:.4f})")
    for a, curves, lab in ((ax[0], tprs, f"Mean (AUC = {np.mean(aucs):.4f})"), (ax[1], precs, f"Mean (AP = {np.mean(aps):.4f})")):
        m, sd = np.mean(curves, 0), np.std(curves, 0)
        a.plot(grid, m, color="k", lw=1.5, label=lab); a.fill_between(grid, m - sd, m + sd, color="gray", alpha=0.2)
    ax[0].plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    ax[1].axhline(np.mean(prev), ls="--", color="gray", lw=0.8)
    ax[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    ax[0].legend(loc="lower right", fontsize="small"); ax[1].legend(loc="lower left", fontsize="small")
    fig.tight_layout(); _finish(output_path)


def plot_training_validation_metrics(train_accuracies, avg_train_accuracies, val_accuracies, avg_val_accuracies,
                                     train_losses, avg_train_losses, val_losses, avg_val_losses, output_path, args):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, curves, avg, ttl, yl in ((axes[0, 0], train_accuracies, avg_train_accuracies, "Training Accuracy", "Accuracy"),
                                     (axes[0, 1], val_accuracies, avg_val_accuracies, "Validation Accuracy", "Accuracy"),
                                     (axes[1, 0], train_losses, avg_train_losses, "Training Loss", "Loss"),
                                     (axes[1, 1], val_losses, avg_val_losses, "Validation Loss", "Loss")):
        for i, c in enumerate(curves): ax.plot(c, lw=1, label=f"Fold {i + 1}")
        ax.plot(avg, color="k", lw=1.5, label="Mean"); ax.set(xlabel="Epochs", ylabel=yl, title=ttl); ax.legend(fontsize="small")
    fig.tight_layout()
    _finish(os.path.join(output_path, f"train_val_metrics_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png"))


def plot_scores(epochs, train_f1_scores, val_f1_scores, train_focal_loss_scores, val_focal_loss_scores, train_auc_scores,
                val_auc_scores, train_map_scores, val_map_scores, train_recall_scores, val_recall_scores, train_acc_scores,
                val_acc_scores, train_precision_scores, val_precision_scores, output_path, args):
    os.makedirs(output_path, exist_ok=True)
    spec = [("f1", "F1 Score", train_f1_scores, val_f1_scores), ("loss", "FocalLoss", train_focal_loss_scores, val_focal_loss_scores),
            ("auc", "AUC", train_auc_scores, val_auc_scores), ("mAP", "mAP", train_map_scores, val_map_scores),
            ("recall", "Recall", train_recall_scores, val_recall_scores), ("acc", "Accuracy", train_acc_scores, val_acc_scores),
            ("precision", "Precision", train_precision_scores, val_precision_scores)]
    for tag, name, tr, va in spec:
        plt.figure(); plt.plot(epochs, tr, label=f"Training {name}"); plt.plot(epochs, va, label=f"Validation {name}")
        plt.xlabel("Epochs"); plt.ylabel(name); plt.title(f"Training and Validation {name} over Epochs"); plt.legend()
        _finish(os.path.join(output_path, f"{tag}_head{args.num_heads}_dim{args.out_feats}_lay{args.num_layers}_epo{args.epochs}.png"))


def plot_validation_accuracy(accuracies, avg_accuracies, output_path, args):
    plt.figure(figsize=(10, 6))
    for i, a in enumerate(accuracies): plt.plot(a, label=f"Fold {i + 1}")
    plt.plot(avg_accuracies, color="k", label="Mean"); plt.xlabel("Epochs"); plt.ylabel("Validation Accuracy"); plt.legend()
    _finish(os.path.join(output_path, f"accuracy_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png"))


def plot_loss(losses, avg_losses, output_path, args):
    plt.figure(figsize=(10, 6))
    for i, l in enumerate(losses): plt.plot(l, label=f"Fold {i + 1}")
    plt.plot(avg_losses, color="k", label="Mean"); plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend()
    _finish(os.path.join(output_path, f"loss_lr{args.lr}_lay{args.num_layers}_input{args.input_size}_dim{args.out_feats}_epoch{args.epochs}.png"))


plot_precision_recall_curves = plot_pr_curves          # old names kept as aliases
plot_roc_curves_ = plot_roc_curves


# --------------------------------------------------------------------------- literature evidence (vectorised)
def find_evidence(row, df2):
    """Single-row version kept for compatibility; prefer add_evidence(df, df2) for whole tables."""
    for a, b in ((row["source"], row["destination"]), (row["destination"], row["source"])):
        m = df2[(df2["miRNA"] == a) & (df2["disease"] == b)]
        if not m.empty:
            return m.iloc[0]["reference"]
    return "Unconfirmed"


def add_evidence(df, df2, src="source", dst="destination"):
    """Vectorised: one merge per orientation instead of scanning df2 for every row."""
    ref = df2.drop_duplicates(["miRNA", "disease"])[["miRNA", "disease", "reference"]]
    a = df[[src, dst]].merge(ref, left_on=[src, dst], right_on=["miRNA", "disease"], how="left")["reference"]
    b = df[[src, dst]].merge(ref, left_on=[dst, src], right_on=["miRNA", "disease"], how="left")["reference"]
    out = df.copy(); out["reference"] = a.values; out["reference"] = out["reference"].fillna(pd_series(b.values)).fillna("Unconfirmed")
    return out


def pd_series(v):
    import pandas as pd
    return pd.Series(v)
