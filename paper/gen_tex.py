"""Turn paper/analysis.json into LaTeX macros and table rows, so no number is typed by hand."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
A = json.loads((here / "analysis.json").read_text())
pct = lambda v, d=1: f"{v * 100:.{d}f}"
out = []
def mac(name, val):
    out.append(f"\\newcommand{{\\{name}}}{{{val}}}")

mac("AUC", f"{A['auc']:.3f}")
mac("AUClo", f"{A['auc_ci95'][0]:.3f}")
mac("AUChi", f"{A['auc_ci95'][1]:.3f}")
mac("EER", pct(A["eer"]))
mac("EERlo", pct(A["eer_ci95"][0]))
mac("EERhi", pct(A["eer_ci95"][1]))
mac("EERthr", f"{A['eer_threshold']:.4f}")
mac("AccEER", pct(A["acc_eer"]))
mac("EERexcl", pct(A["excl_A17"][1]))
mac("AUCexcl", f"{A['excl_A17'][0]:.3f}")
mac("EERseen", pct(A["seen_algorithms"][1]))
mac("EERunseen", pct(A["unseen_excl_A17"][1]))
mac("AUCunseen", f"{A['unseen_excl_A17'][0]:.3f}")
mac("AUCAsev", f"{A['A17'][0]:.3f}")
mac("EERAsev", pct(A["A17"][1]))
mac("EERAsevlo", pct(A["A17_eer_ci95"][0]))
mac("EERAsevhi", pct(A["A17_eer_ci95"][1]))
mac("DetAsev", pct(A["detected_at_0.5_A17"]))
mac("DetOther", pct(A["detected_at_0.5_excl_A17"]))
mac("ECE", f"{A['ece_15']:.3f}")
mac("Brier", f"{A['brier']:.3f}")
mac("NLL", f"{A['nll']:.3f}")
mac("FPRhalf", pct(A["fpr_0.5"]))
mac("FNRhalf", pct(A["fnr_0.5"]))
mac("TrainMinutes", f"{A['train_seconds_total'] / 60:.1f}")
mac("SecPerEpoch", f"{A['seconds_per_epoch_mean']:.0f}")
(here / "numbers.tex").write_text("\n".join(out) + "\n")

rows = []
for r in A["per_attack"]:
    tag = r["attack"] + ("$^\\dagger$" if r["seen_algorithm"] else "")
    rows.append(f"{tag} & {r['n']} & {pct(r['detect'])} & {r['median_score']:.3g} & {r['auc']:.4f} & {pct(r['eer'], 2)} \\\\")
table = (
    "\\begin{tabular}{@{}lrrrcr@{}}\n\\toprule\n"
    "Attack & $n$ & Detected (\\%) & Median score & AUC & EER (\\%)\\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n"
    "\\multicolumn{6}{@{}p{0.97\\columnwidth}@{}}{\\scriptsize $^\\dagger$Same generation algorithm as a training attack (A04 / A06). "
    "``Detected'' uses threshold 0.5.}\n\\end{tabular}\n"
)
(here / "attack_table.tex").write_text(table)
print("\n".join(out))

H = A["history"]
lines = []
for i in range(len(H["epoch"])):
    lines.append(f"{H['epoch'][i]} & {H['train_loss'][i]:.4f} & {H['val_loss'][i]:.4f} & {H['val_auc'][i]:.6f} & "
                 f"{H['val_acc'][i] * 100:.2f} & {H['lr'][i]:.2e} & {H['seconds'][i]:.0f} \\\\")
(here / "history_table.tex").write_text(
    "\\begin{tabular}{@{}cccccrr@{}}\n\\toprule\n"
    "Epoch & Train loss & Val. loss & Val. AUC & Val. acc. (\\%) & LR & Time (s)\\\\\n\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
