"""
Figure generation for the Multivalent-Ion-Conductor-ML-project manuscript.
Two independent dataset angles, no redundant figures between them or
between main text and SI. No gridlines anywhere. 300dpi PDF+PNG.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 11, "axes.grid": False, "figure.dpi": 300,
    "font.family": "sans-serif",
})

PROJ = Path("/glade/work/ajalali/12 - Multivalent-Ion-Conductor-ML-project")
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MONO_COLOR = "#2166AC"
MULTI_COLOR = "#B2182B"
NEUTRAL = "#4D4D4D"


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {name}")


def no_grid(ax):
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# ---------- load everything ----------
cavd = pd.read_parquet(PROJ / "data/feature_table.parquet")
mig = pd.read_parquet(PROJ / "data/migration_barrier_table.parquet").dropna(subset=["mobile_cation"])
cavd_metrics = json.load(open(PROJ / "models/baseline/metrics.json"))
cal_metrics = json.load(open(PROJ / "models/baseline/calibration_metrics.json"))
mig_metrics = json.load(open(PROJ / "models/migration_baseline/metrics.json"))
strengthen = json.load(open(PROJ / "models/migration_baseline/strengthening_metrics.json"))
pred_random = pd.read_csv(PROJ / "models/migration_baseline/predictions_random_split.csv")
bench_mig = json.load(open(PROJ / "models/migration_baseline/benchmark_models.json"))
bench_cavd = json.load(open(PROJ / "models/baseline/benchmark_models.json"))

# ============================================================
# MAIN TEXT FIGURES
# ============================================================

# Fig 1: dataset overview -- per-cation sample counts, both datasets, one figure
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
cavd_counts = cavd["cation"].value_counts().reindex(["Li", "Na", "Mg", "Al"])
colors1 = [MONO_COLOR, MONO_COLOR, MULTI_COLOR, MULTI_COLOR]
axes[0].bar(cavd_counts.index, cavd_counts.values, color=colors1)
axes[0].set_ylabel("Number of structures")
axes[0].set_title("CAVD percolation dataset (n=7629)")
no_grid(axes[0])

mig_counts = mig["mobile_cation"].value_counts().reindex(["Li", "Na", "K", "Rb", "Ca", "Mg", "Zn", "Al", "Sr"])
colors2 = [MONO_COLOR]*4 + [MULTI_COLOR]*5
axes[1].bar(mig_counts.index, mig_counts.values, color=colors2)
axes[1].set_ylabel("Number of entries")
axes[1].set_title("Migration-barrier dataset (n=617)")
no_grid(axes[1])
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=MONO_COLOR, label="Monovalent"), Patch(color=MULTI_COLOR, label="Multivalent")],
           loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
fig.tight_layout()
save(fig, "fig1_dataset_overview")

# Fig 2: parity plot, migration barrier random split (XGBoost)
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.scatter(pred_random["target_Em"], pred_random["pred_Em"], s=18, alpha=0.6, color=NEUTRAL, edgecolor="none")
lims = [0, max(pred_random["target_Em"].max(), pred_random["pred_Em"].max()) * 1.05]
ax.plot(lims, lims, "--", color="black", linewidth=1)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("DFT-NEB migration barrier, true (eV)")
ax.set_ylabel("XGBoost predicted (eV)")
mae = mig_metrics["random_split"]["mae"]; r2 = mig_metrics["random_split"]["r2"]
ax.text(0.05, 0.92, f"MAE = {mae:.2f} eV\nR$^2$ = {r2:.2f}", transform=ax.transAxes, va="top")
no_grid(ax)
fig.tight_layout()
save(fig, "fig2_parity_migration_random_split")

# Fig 3: unified cross-valence extrapolation comparison, both datasets, both metrics
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
# left: CAVD balanced accuracy (binary), random / mono->multi / multi->mono
cavd_bin = cavd_metrics["percolates_binary"]
protos = ["random_split", "train_mono_test_multi", "train_multi_test_mono"]
labels = ["In-distribution\n(random split)", "Train mono\n→ test multi", "Train multi\n→ test mono"]
vals = [cavd_bin[p]["balanced_accuracy"] for p in protos]
bars = axes[0].bar(labels, vals, color=[NEUTRAL, MONO_COLOR, MULTI_COLOR])
axes[0].axhline(0.5, linestyle="--", color="black", linewidth=1, label="Chance level")
axes[0].set_ylabel("Balanced accuracy")
axes[0].set_title("CAVD: percolation classification")
axes[0].set_ylim(0, 1)
axes[0].legend(frameon=False, fontsize=9)
no_grid(axes[0])

# right: migration barrier MAE, same three protocols
mig_vals = [mig_metrics["random_split"]["mae"], mig_metrics["train_mono_test_multi"]["mae"],
            mig_metrics["train_multi_test_mono"]["mae"]]
naive = mig_metrics["naive_mean_predictor_mae_random_split"]
axes[1].bar(labels, mig_vals, color=[NEUTRAL, MONO_COLOR, MULTI_COLOR])
axes[1].axhline(naive, linestyle="--", color="black", linewidth=1, label="Naive mean predictor")
axes[1].set_ylabel("MAE (eV)")
axes[1].set_title("Migration barrier: regression")
axes[1].legend(frameon=False, fontsize=9)
no_grid(axes[1])
fig.tight_layout()
save(fig, "fig3_cross_valence_extrapolation")

# Fig 4: leave-one-cation-out, migration barrier MAE (bar chart), with n per cation labeled
fig, ax = plt.subplots(figsize=(6, 3.8))
loc = mig_metrics["leave_one_cation_out"]
cats = list(loc.keys())
mae_vals = [loc[c]["mae"] for c in cats]
n_vals = [loc[c].get("n_test", loc[c].get("n", None)) for c in cats]
colors = [MONO_COLOR if c in ("Li", "Na", "K", "Rb") else MULTI_COLOR for c in cats]
bars4 = ax.bar(cats, mae_vals, color=colors)
for b, n in zip(bars4, n_vals):
    label = f"n={n}" if n is not None else ""
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, label,
            ha="center", va="bottom", fontsize=8)
ax.axhline(naive, linestyle="--", color="black", linewidth=1, label="Naive mean predictor (random split)")
ax.set_ylabel("MAE (eV), left out")
ax.set_title("Leave-one-cation-out (migration barrier)")
ax.legend(frameon=False, fontsize=9)
no_grid(ax)
fig.tight_layout()
save(fig, "fig4_leave_one_cation_out")

# Fig 5: SHAP summary (top 15 features, bar)
fig, ax = plt.subplots(figsize=(6, 5))
shap_top = strengthen["shap_top20_features"][:15][::-1]
names = [d["feature"].replace("MagpieData ", "") for d in shap_top]
vals = [d["mean_abs_shap"] for d in shap_top]
ax.barh(names, vals, color=NEUTRAL)
ax.set_xlabel("Mean |SHAP value| (eV)")
ax.set_title("Top drivers of predicted migration barrier")
no_grid(ax)
fig.tight_layout()
save(fig, "fig5_shap_importance")

# Fig 6: consolidated multi-model benchmark, both dataset angles, same 3 protocols
# (replaces the narrower XGB-vs-GP-only comparison; that pairing's calibration
# story is preserved separately in the SI, figs5, since only those two models
# produce a native/tuned uncertainty estimate)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
proto_keys_mig = ["random_split", "mono_to_multi", "multi_to_mono"]
mig_model_order = ["XGBoost", "RandomForest", "SVR", "kNN", "Ridge", "MLP"]
x = np.arange(len(proto_keys_mig))
width = 0.13
cmap = plt.get_cmap("tab10")
for i, name in enumerate(mig_model_order):
    raw_vals = [bench_mig[name][p]["mae"] for p in proto_keys_mig]
    vals = [min(v, 2.0) for v in raw_vals]  # clip Ridge/MLP blowups for readability
    bars_i = axes[0].bar(x + (i - len(mig_model_order)/2 + 0.5) * width, vals, width, label=name, color=cmap(i))
    for b, rv in zip(bars_i, raw_vals):
        if rv > 2.0:  # label the true value for any bar that got clipped
            axes[0].text(b.get_x() + b.get_width() / 2, 2.0 + 0.03, f"{rv:.2f}",
                         ha="center", va="bottom", fontsize=6, rotation=90)
axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
axes[0].set_ylabel("MAE (eV), clipped at 2.0 (true value labeled if clipped)")
axes[0].set_title("Migration-barrier regression: 6 models")
axes[0].legend(frameon=False, fontsize=7, ncol=2)
no_grid(axes[0])

proto_keys_cavd = ["random_split", "train_mono_test_multi", "train_multi_test_mono"]
cavd_model_order = ["RandomForest", "SVM", "kNN", "LogisticRegression", "MLP"]
width2 = 0.15
for i, name in enumerate(cavd_model_order):
    vals = [bench_cavd[name][p]["balanced_accuracy"] for p in proto_keys_cavd]
    axes[1].bar(x + (i - len(cavd_model_order)/2 + 0.5) * width2, vals, width2, label=name, color=cmap(i))
axes[1].axhline(0.5, linestyle="--", color="black", linewidth=1, label="Chance")
axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
axes[1].set_ylabel("Balanced accuracy")
axes[1].set_title("Percolation classification: 5 models")
axes[1].set_ylim(0, 1)
axes[1].legend(frameon=False, fontsize=7, ncol=2)
no_grid(axes[1])
fig.tight_layout()
save(fig, "fig6_model_benchmark")

# ============================================================
# SI-ONLY FIGURES
# ============================================================

# Fig S1: CAVD calibration curves (reliability diagrams), 3 protocols
fig, ax = plt.subplots(figsize=(5, 5))
for proto, color, lab in [("random_split", NEUTRAL, "In-distribution"),
                          ("mono_to_multi", MONO_COLOR, "Mono→multi"),
                          ("multi_to_mono", MULTI_COLOR, "Multi→mono")]:
    c = cal_metrics[proto]
    ax.plot(c["calibration_curve_mean_predicted"], c["calibration_curve_fraction_positive"],
            marker="o", label=f"{lab} (Brier={c['brier_score']:.3f})", color=color)
ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed fraction percolating")
ax.set_title("CAVD classifier calibration")
ax.legend(frameon=False, fontsize=8)
no_grid(ax)
fig.tight_layout()
save(fig, "figs1_cavd_calibration")

# Fig S2: leave-one-structural-class-out, migration barrier (all classes n>=10)
fig, ax = plt.subplots(figsize=(7, 5))
loco = mig_metrics["leave_one_class_out"]
classes = sorted(loco.keys(), key=lambda c: loco[c]["r2"])
r2s = [max(loco[c]["r2"], -5) for c in classes]  # clip extreme outlier (OLIVINE R2=-32) for readability
ax.barh(classes, r2s, color=[MULTI_COLOR if v < 0 else MONO_COLOR for v in r2s])
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("R$^2$ (clipped at -5 for readability; see main text for exact OLIVINE value)")
ax.set_title("Leave-one-structural-class-out (migration barrier)")
no_grid(ax)
fig.tight_layout()
save(fig, "figs2_leave_one_class_out")

# Fig S3: quantile-regression regularization sweep (mean coverage vs. regularization)
fig, ax = plt.subplots(figsize=(5.5, 4))
sweep = strengthen["quantile_regularization_sweep"]["sweep"]
labels_sweep = [f"{s['n_estimators']}/{s['max_depth']}/{s['reg_lambda']:.0f}" for s in sweep]
means = [s["mean_coverage_across_protocols"] for s in sweep]
ax.plot(range(len(sweep)), means, marker="o", color=NEUTRAL)
ax.axhline(0.8, linestyle="--", color="black", linewidth=1, label="80% target")
ax.set_xticks(range(len(sweep)))
ax.set_xticklabels(labels_sweep, rotation=40, ha="right", fontsize=8)
ax.set_xlabel("n_estimators / max_depth / $\\lambda$")
ax.set_ylabel("Mean coverage (3 protocols)")
ax.set_title("Quantile-regression regularization sweep")
ax.legend(frameon=False)
no_grid(ax)
fig.tight_layout()
save(fig, "figs3_quantile_sweep")

# Fig S4: CAVD leave-one-cation-out (binary balanced accuracy), the classification analogue of Fig 4
fig, ax = plt.subplots(figsize=(5.5, 3.8))
loc_cavd = cavd_bin["leave_one_cation_out"] if "leave_one_cation_out" not in cavd_bin else cavd_bin["leave_one_cation_out"]
loc_cavd = cavd_metrics["percolates_binary"]["leave_one_cation_out"]
cats2 = list(loc_cavd.keys())
bal = [loc_cavd[c]["balanced_accuracy"] for c in cats2]
colors3 = [MONO_COLOR if c in ("Li", "Na") else MULTI_COLOR for c in cats2]
ax.bar(cats2, bal, color=colors3)
ax.axhline(0.5, linestyle="--", color="black", linewidth=1, label="Chance level")
ax.set_ylabel("Balanced accuracy, left out")
ax.set_title("CAVD leave-one-cation-out (percolation)")
ax.legend(frameon=False, fontsize=9)
no_grid(ax)
fig.tight_layout()
save(fig, "figs4_cavd_leave_one_cation_out")

# Fig S5: XGBoost (tuned quantile) vs GP (native) calibration, the pairing
# that make sense for uncertainty since only these two models produce a
# usable interval; kept separate from the broader fig6 point-estimate
# benchmark, which has no calibration axis for the other four models.
fig, ax = plt.subplots(figsize=(5.5, 4))
gp = strengthen["gaussian_process"]
gp_cov = [gp[p]["coverage_80"] for p in proto_keys_mig]
xgb_q_best = strengthen["quantile_regularization_sweep"]["chosen_results"]
xgb_cov = [xgb_q_best[p]["coverage"] for p in proto_keys_mig]
xw = np.arange(3); width3 = 0.35
ax.bar(xw - width3/2, xgb_cov, width3, label="XGBoost quantile (tuned)", color=MONO_COLOR)
ax.bar(xw + width3/2, gp_cov, width3, label="GP (native)", color=MULTI_COLOR)
ax.axhline(0.8, linestyle="--", color="black", linewidth=1, label="80% target")
ax.set_xticks(xw); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Empirical 80% coverage")
ax.set_title("Uncertainty calibration: XGBoost vs. GP")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=8)
no_grid(ax)
fig.tight_layout()
save(fig, "figs5_calibration_xgb_vs_gp")

# Fig S6: residual plot, migration barrier random split, large errors highlighted
fig, ax = plt.subplots(figsize=(5.5, 4.2))
resid = pred_random["pred_Em"] - pred_random["target_Em"]
large = resid.abs() > resid.abs().quantile(0.9)
ax.scatter(pred_random["target_Em"][~large], resid[~large], s=16, alpha=0.6, color=NEUTRAL, edgecolor="none")
ax.scatter(pred_random["target_Em"][large], resid[large], s=30, alpha=0.9, color=MULTI_COLOR,
           edgecolor="black", linewidth=0.3, label="Largest 10% of errors")
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("True migration barrier (eV)")
ax.set_ylabel("Residual, predicted $-$ true (eV)")
ax.set_title("Residuals, random split (XGBoost)")
ax.legend(frameon=False, fontsize=8)
no_grid(ax)
fig.tight_layout()
save(fig, "figs6_residuals")

# Fig S7: SHAP beeswarm, top 12 features, random-split training fold
try:
    import shap as _shap
    shap_vals = np.load(PROJ / "models/migration_baseline/shap_values_random_split.npy")
    X_shap = np.load(PROJ / "models/migration_baseline/shap_X_random_split.npy")
    shap_names = json.load(open(PROJ / "models/migration_baseline/shap_feature_names.json"))
    shap_names_clean = [n.replace("MagpieData ", "") for n in shap_names]
    order = np.argsort(-np.abs(shap_vals).mean(axis=0))[:12]
    fig = plt.figure(figsize=(7, 5.5))
    _shap.summary_plot(shap_vals[:, order], X_shap[:, order],
                        feature_names=[shap_names_clean[i] for i in order],
                        show=False, plot_size=None)
    fig = plt.gcf()
    ax = plt.gca()
    no_grid(ax)
    fig.tight_layout()
    save(fig, "figs7_shap_beeswarm")
except Exception as e:
    print(f"SHAP beeswarm skipped: {e}")

print("All figures generated.")
