"""Build every data figure and the derived statistics used in the paper.

Inputs are only the saved run files in models/audio_asvspoof_3a53cf51/ (history.csv,
test/predictions.csv, test/metrics.json). Nothing here is simulated: each number written to
paper/analysis.json is computed from those files, and gen_tex.py turns that JSON into the
LaTeX macros the paper uses.

Run from the repository root:   .venv/bin/python paper/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "models" / "audio_asvspoof_3a53cf51"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

SEED = 42
N_BOOT = 2000
SEEN_ALGORITHM = {"A16", "A19"}  # same algorithms as training attacks A04 / A06 (Wang et al. 2020)

# Okabe-Ito colour-blind-safe palette; markers/linestyles keep figures readable in grayscale.
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#555555"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
COL = 3.45  # IEEE single-column width in inches


def eer_of(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def ece_of(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, acc = p[m].mean(), y[m].mean()
            total += m.mean() * abs(acc - conf)
            rows.append((conf, acc, int(m.sum())))
    return float(total), rows


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


hist = pd.read_csv(RUN / "history.csv")
pred = pd.read_csv(RUN / "test" / "predictions.csv")
metrics = json.loads((RUN / "test" / "metrics.json").read_text())
y = pred["label"].to_numpy()
s = pred["score"].to_numpy()
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- derived statistics
A: dict = {}
A["n"] = int(len(y))
A["auc"] = float(roc_auc_score(y, s))
A["eer"], A["eer_threshold"] = eer_of(y, s)
assert abs(A["auc"] - metrics["auc"]) < 1e-9 and abs(A["eer"] - metrics["eer"]) < 1e-9, "metrics.json mismatch"

# Stratified bootstrap over clips (resample bona fide and spoof separately).
pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
boot_auc, boot_eer = [], []
for _ in range(N_BOOT):
    idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    boot_auc.append(roc_auc_score(y[idx], s[idx]))
    boot_eer.append(eer_of(y[idx], s[idx])[0])
A["auc_ci95"] = [float(np.percentile(boot_auc, 2.5)), float(np.percentile(boot_auc, 97.5))]
A["eer_ci95"] = [float(np.percentile(boot_eer, 2.5)), float(np.percentile(boot_eer, 97.5))]

cm05 = metrics["confusion_matrix"]
A["cm_0.5"] = cm05
tn, fp = cm05[0]
fn, tp = cm05[1]
A["fpr_0.5"], A["fnr_0.5"] = fp / (fp + tn), fn / (fn + tp)
thr_eer = A["eer_threshold"]
pe = (s >= thr_eer).astype(int)
A["cm_eer"] = [[int(((pe == 0) & (y == 0)).sum()), int(((pe == 1) & (y == 0)).sum())],
               [int(((pe == 0) & (y == 1)).sum()), int(((pe == 1) & (y == 1)).sum())]]
A["acc_eer"] = float((pe == y).mean())

# Calibration.
A["ece_15"], rel_rows = ece_of(y.astype(float), s)
A["brier"] = float(np.mean((s - y) ** 2))
eps = 1e-7
A["nll"] = float(-np.mean(y * np.log(np.clip(s, eps, 1)) + (1 - y) * np.log(np.clip(1 - s, eps, 1))))

# Per attack (each attack scored against all 2,500 bona fide test clips).
bona = pred[pred.category == "bonafide"]
rows = []
for cat in sorted(c for c in pred.category.unique() if c != "bonafide"):
    sp = pred[pred.category == cat]
    yy = np.r_[np.zeros(len(bona)), np.ones(len(sp))]
    ss = np.r_[bona.score.to_numpy(), sp.score.to_numpy()]
    e, _ = eer_of(yy, ss)
    rows.append(dict(attack=cat, n=int(len(sp)), detect=float((sp.score >= 0.5).mean()),
                     auc=float(roc_auc_score(yy, ss)), eer=e, median_score=float(sp.score.median()),
                     seen_algorithm=cat in SEEN_ALGORITHM))
att = pd.DataFrame(rows)
A["per_attack"] = rows
all_attacks = list(att.attack)


def subset_eer(cats):
    sp = pred[pred.category.isin(cats)]
    yy = np.r_[np.zeros(len(bona)), np.ones(len(sp))]
    ss = np.r_[bona.score.to_numpy(), sp.score.to_numpy()]
    return float(roc_auc_score(yy, ss)), eer_of(yy, ss)[0], int(len(sp))


A["excl_A17"] = subset_eer([c for c in all_attacks if c != "A17"])
A["seen_algorithms"] = subset_eer(sorted(SEEN_ALGORITHM))
A["unseen_excl_A17"] = subset_eer([c for c in all_attacks if c not in SEEN_ALGORITHM and c != "A17"])
A["A17"] = subset_eer(["A17"])
A["detected_at_0.5_excl_A17"] = float((pred[(pred.label == 1) & (pred.category != "A17")].score >= 0.5).mean())
A["detected_at_0.5_A17"] = float((pred[pred.category == "A17"].score >= 0.5).mean())

# A17 bootstrap CI (the one per-attack interval quoted in the text).
sp17 = pred[pred.category == "A17"]
b17 = []
for _ in range(1000):
    b = rng.choice(len(bona), len(bona))
    a = rng.choice(len(sp17), len(sp17))
    b17.append(eer_of(np.r_[np.zeros(len(b)), np.ones(len(a))],
                      np.r_[bona.score.to_numpy()[b], sp17.score.to_numpy()[a]])[0])
A["A17_eer_ci95"] = [float(np.percentile(b17, 2.5)), float(np.percentile(b17, 97.5))]

A["history"] = hist.round(6).to_dict(orient="list")
A["train_seconds_total"] = float(hist.seconds.sum())
A["seconds_per_epoch_mean"] = float(hist.seconds.mean())

(Path(__file__).resolve().parent / "analysis.json").write_text(json.dumps(A, indent=2))

# ---------------------------------------------------------------- Fig: training curves
fig, ax = plt.subplots(1, 2, figsize=(COL, 1.75))
ep = hist.epoch
ax[0].plot(ep, hist.train_loss, "o-", color=BLUE, ms=3, label="train")
ax[0].plot(ep, hist.val_loss, "s--", color=ORANGE, ms=3, label="validation")
ax[0].set_yscale("log")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Weighted BCE loss")
ax[0].axvline(1.5, color=GREY, lw=0.6, ls=":")
ax[0].text(1.55, ax[0].get_ylim()[0] * 1.6, "backbone\nunfrozen", fontsize=5.5, color=GREY, va="bottom")
ax[0].legend(frameon=False, loc="upper right")
ax[0].grid(True, which="major")
ax[1].plot(ep, hist.val_auc, "o-", color=GREEN, ms=3, label="val. AUC")
ax[1].plot(ep, hist.val_acc, "s--", color=PURPLE, ms=3, label="val. accuracy")
ax[1].axhline(A["auc"], color=RED, lw=0.9, ls="-.", label="test AUC")
ax[1].set_ylim(0.66, 1.01)
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Score")
ax[1].axvline(3, color=GREY, lw=0.6, ls=":")
ax[1].legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.12))
ax[1].grid(True)
for a in ax:
    a.set_xticks(ep)
fig.tight_layout(w_pad=1.0)
save(fig, "fig_training")

# ---------------------------------------------------------------- Fig: ROC + DET
fpr, tpr, _ = roc_curve(y, s)
fnr = 1 - tpr
y17 = np.r_[np.zeros(len(bona)), np.ones(len(sp17))]
f17, t17, _ = roc_curve(y17, np.r_[bona.score, sp17.score])
no17 = pred[pred.category != "A17"]
f2, t2, _ = roc_curve(no17.label, no17.score)
fig, ax = plt.subplots(1, 2, figsize=(COL, 1.85))
ax[0].plot(fpr, tpr, color=BLUE, label=f"All attacks (AUC {A['auc']:.3f})")
ax[0].plot(f17, t17, color=RED, ls="--", label=f"A17 only (AUC {A['A17'][0]:.3f})")
ax[0].plot(f2, t2, color=GREEN, ls="-.", label="Excluding A17")
ax[0].plot([0, 1], [0, 1], color=GREY, lw=0.6, ls=":")
ax[0].plot([A["eer"]], [1 - A["eer"]], "k*", ms=5)
ax[0].set_xlabel("False positive rate")
ax[0].set_ylabel("True positive rate")
ax[0].legend(frameon=False, loc="lower right", fontsize=5.5)
ax[0].grid(True)
ax[0].set_xlim(-0.01, 1)
ax[0].set_ylim(0, 1.01)
clip = lambda v: np.clip(v, 1e-4, 1 - 1e-4)
ax[1].plot(norm.ppf(clip(fpr)), norm.ppf(clip(fnr)), color=BLUE, label="All")
ax[1].plot(norm.ppf(clip(f17)), norm.ppf(clip(1 - t17)), color=RED, ls="--", label="A17")
ax[1].plot(norm.ppf(clip(f2)), norm.ppf(clip(1 - t2)), color=GREEN, ls="-.", label="Excl. A17")
ticks = [0.001, 0.01, 0.05, 0.2, 0.5, 0.9]
ax[1].set_xticks(norm.ppf(ticks))
ax[1].set_yticks(norm.ppf(ticks))
fmt = FuncFormatter(lambda v, _: f"{norm.cdf(v) * 100:g}")
ax[1].xaxis.set_major_formatter(fmt)
ax[1].yaxis.set_major_formatter(fmt)
ax[1].set_xlim(norm.ppf(0.0005), norm.ppf(0.9))
ax[1].set_ylim(norm.ppf(0.0005), norm.ppf(0.9))
ax[1].plot([norm.ppf(0.0005), norm.ppf(0.9)], [norm.ppf(0.0005), norm.ppf(0.9)], color=GREY, lw=0.6, ls=":")
ax[1].set_xlabel("False alarm rate (%)")
ax[1].set_ylabel("Miss rate (%)")
ax[1].legend(frameon=False, loc="upper right", fontsize=5.5)
ax[1].grid(True)
fig.tight_layout(w_pad=1.0)
save(fig, "fig_roc_det")

# ---------------------------------------------------------------- Fig: score histograms
fig, ax = plt.subplots(figsize=(COL, 1.8))
bins = np.linspace(0, 1, 41)
ax.hist(pred[pred.label == 0].score, bins=bins, color=BLUE, alpha=0.75, label="Bona fide", edgecolor="white", lw=0.2)
ax.hist(pred[pred.label == 1].score, bins=bins, color=RED, alpha=0.65, label="Spoof", edgecolor="white", lw=0.2, hatch="////")
ax.hist(pred[pred.category == "A17"].score, bins=bins, histtype="step", color="k", lw=1.0, label="Spoof, A17 only")
ax.set_yscale("log")
ax.axvline(0.5, color=GREY, ls="--", lw=0.8)
ax.axvline(thr_eer, color=GREEN, ls="-.", lw=0.9)
ax.text(0.51, 1500, "threshold 0.5", fontsize=5.8, color=GREY)
ax.annotate(f"EER threshold\n({thr_eer:.3f})", (thr_eer, 300), xytext=(0.12, 700), fontsize=5.8, color=GREEN,
            arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.5))
ax.set_xlabel("Predicted probability of spoof")
ax.set_ylabel("Clips (log scale)")
ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=5.8, bbox_to_anchor=(0.5, 1.2))
ax.grid(True, axis="y")
save(fig, "fig_scores")

# ---------------------------------------------------------------- Fig: per-attack EER + detection rate
fig, ax = plt.subplots(2, 1, figsize=(COL, 3.0), sharex=True)
x = np.arange(len(att))
cols = [GREEN if sa else (RED if a == "A17" else BLUE) for a, sa in zip(att.attack, att.seen_algorithm)]
ax[0].bar(x, att.eer * 100, color=cols, edgecolor="k", lw=0.3)
ax[0].set_yscale("log")
ax[0].set_ylim(0.03, 400)
ax[0].set_ylabel("EER vs. bona fide (%)")
ax[0].axhline(A["eer"] * 100, color=GREY, ls="--", lw=0.7)
ax[0].text(6, A["eer"] * 100 * 1.2, "pooled EER 6.2%", fontsize=5.8, color=GREY, ha="center")
ax[0].grid(True, axis="y", which="major")
ax[0].legend(handles=[Patch(fc=GREEN, ec="k", lw=0.3, label="Algorithm seen in training (A16, A19)"),
                      Patch(fc=BLUE, ec="k", lw=0.3, label="Unseen algorithm"),
                      Patch(fc=RED, ec="k", lw=0.3, label="A17")],
             frameon=False, loc="upper left", fontsize=5.5, ncol=1)
ax[1].bar(x, att.detect * 100, color=cols, edgecolor="k", lw=0.3)
ax[1].set_ylabel("Detected at 0.5 (%)")
ax[1].set_ylim(0, 108)
ax[1].set_xticks(x)
ax[1].set_xticklabels(att.attack, rotation=60, ha="right")
ax[1].grid(True, axis="y")
fig.tight_layout(h_pad=0.4)
save(fig, "fig_per_attack")

# ---------------------------------------------------------------- Fig: score ECDF per group
fig, ax = plt.subplots(figsize=(COL, 1.9))
easy = ["A07", "A08", "A09", "A11", "A14", "A15", "A16", "A19"]
groups = [("Bona fide", pred[pred.category == "bonafide"].score, BLUE, "-"),
          ("A17", pred[pred.category == "A17"].score, RED, "--"),
          ("A13", pred[pred.category == "A13"].score, PURPLE, "-."),
          ("A10, A12, A18", pred[pred.category.isin(["A10", "A12", "A18"])].score, ORANGE, ":"),
          ("A07-A09, A11, A14-A16, A19", pred[pred.category.isin(easy)].score, GREEN, (0, (5, 1, 1, 1)))]
for name, v, c, ls in groups:
    v = np.sort(v.to_numpy())
    ax.step(np.clip(v, 1e-6, 1), np.arange(1, len(v) + 1) / len(v), where="post", color=c, ls=ls, label=name)
ax.set_xscale("log")
ax.set_xlim(1e-6, 1.05)
ax.axvline(0.5, color=GREY, ls="--", lw=0.6)
ax.axvline(thr_eer, color="k", ls=":", lw=0.6)
ax.set_xlabel("Predicted probability of spoof (log scale)")
ax.set_ylabel("Cumulative share of clips")
ax.legend(frameon=False, loc="lower center", fontsize=5.6, ncol=3, bbox_to_anchor=(0.5, 1.0),
          columnspacing=0.8, handlelength=1.8)
ax.grid(True)
save(fig, "fig_score_ecdf")

# ---------------------------------------------------------------- Fig: confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(COL, 1.55))
for a, cm, ttl in zip(ax, [A["cm_0.5"], A["cm_eer"]], ["Threshold 0.5", f"EER threshold ({thr_eer:.3f})"]):
    cm = np.array(cm)
    a.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / cm[i].sum()
            a.text(j, i, f"{cm[i, j]}\n({frac * 100:.1f}%)", ha="center", va="center", fontsize=6.3,
                   color="white" if frac > 0.55 else "black")
    a.set_xticks([0, 1])
    a.set_yticks([0, 1])
    a.set_xticklabels(["bona fide", "spoof"])
    a.set_yticklabels(["bona fide", "spoof"], rotation=90, va="center")
    a.set_xlabel("Predicted")
    a.set_title(ttl, fontsize=6.8)
ax[0].set_ylabel("True")
fig.tight_layout(w_pad=0.8)
save(fig, "fig_confusion")

# ---------------------------------------------------------------- Fig: calibration + threshold sweep
fig, ax = plt.subplots(1, 2, figsize=(COL, 1.8))
conf = np.array([r[0] for r in rel_rows])
accs = np.array([r[1] for r in rel_rows])
ax[0].plot([0, 1], [0, 1], color=GREY, ls=":", lw=0.7, label="perfect")
ax[0].plot(conf, accs, "o-", color=BLUE, ms=2.5, label=f"model (ECE {A['ece_15']:.3f})")
ax[0].set_xlabel("Mean predicted P(spoof)")
ax[0].set_ylabel("Observed spoof fraction")
ax[0].legend(frameon=False, loc="lower right", fontsize=5.8)
ax[0].grid(True)
ths = np.linspace(0.0005, 0.9995, 600)
fpr_t = np.array([(s[y == 0] >= t).mean() for t in ths])
tpr_t = np.array([(s[y == 1] >= t).mean() for t in ths])
ax[1].plot(ths, fpr_t * 100, color=BLUE, label="False alarm (bona fide)")
ax[1].plot(ths, 100 - tpr_t * 100, color=RED, ls="--", label="Miss (spoof)")
ax[1].set_xscale("log")
ax[1].axvline(0.5, color=GREY, ls="--", lw=0.7)
ax[1].axvline(thr_eer, color=GREEN, ls="-.", lw=0.8)
ax[1].set_xlabel("Decision threshold")
ax[1].set_ylabel("Error rate (%)")
ax[1].legend(frameon=False, loc="center left", fontsize=5.6, bbox_to_anchor=(0.0, 0.62))
ax[1].grid(True, which="major")
fig.tight_layout(w_pad=1.0)
save(fig, "fig_calibration_threshold")

# ---------------------------------------------------------------- Fig: data split
fig, ax = plt.subplots(figsize=(COL, 1.35))
labels = ["Train\n(train partition)", "Validation\n(dev. partition)", "Test\n(eval. partition)"]
ax.barh(labels, [2500] * 3, color=BLUE, edgecolor="k", lw=0.3, label="Bona fide")
ax.barh(labels, [2500] * 3, left=[2500] * 3, color=RED, edgecolor="k", lw=0.3, hatch="////", label="Spoof")
for i, a in enumerate(["A01-A06", "A01-A06", "A07-A19"]):
    ax.text(3750, i, a, ha="center", va="center", color="white", fontsize=6.3, fontweight="bold")
ax.set_xlabel("Clips")
ax.invert_yaxis()
ax.legend(frameon=False, loc="lower right", fontsize=5.8, ncol=2, bbox_to_anchor=(1.0, 1.0))
ax.set_xlim(0, 5000)
save(fig, "fig_split")

print(json.dumps({k: v for k, v in A.items() if k not in ("per_attack", "history")}, indent=1))
