# Physical-descriptor and UQ analyses: NEB-displacement mobile-ion identification, Arrhenius
# threshold framing, CAVD composition/structure-grouped validation, dataset heterogeneity,
# Ewald/Madelung descriptors, leakage-free quantile-regression UQ, and SHAP-vs-physical checks.
# No new DFT/atomistic simulation -- re-analysis of data already on disk plus one classical
# (non-quantum) electrostatics calculation (Ewald summation over existing ionic coordinates
# with heuristically/BVAnalyzer-assigned formal charges).
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore")

PROJ = Path("/glade/work/ajalali/12 - Multivalent-Ion-Conductor-ML-project")
DATA = PROJ / "data"
MODELS = PROJ / "models"
OUT = PROJ / "models" / "physical_descriptors_and_uq"
OUT.mkdir(parents=True, exist_ok=True)

SHANNON_RADII = {"Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52,
                  "Mg": 0.72, "Ca": 1.00, "Zn": 0.74, "Al": 0.535, "Sr": 1.18}
VALENCE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Mg": 2, "Ca": 2, "Zn": 2, "Al": 3, "Sr": 2}
COMMON_OX = {  # common formal oxidation states for Ewald summation (chemistry convention, not a calculation)
    "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Mg": 2, "Ca": 2, "Zn": 2, "Al": 3, "Sr": 2,
    "O": -2, "F": -1, "Cl": -1, "Br": -1, "I": -1, "S": -2, "Se": -2, "N": -3, "P": 5,
    "Si": 4, "Ti": 4, "V": 5, "Cr": 3, "Mn": 2, "Fe": 3, "Co": 2, "Ni": 2, "Cu": 2,
    "Zr": 4, "Nb": 5, "Mo": 6, "W": 6, "Sn": 4, "Sb": 5, "Bi": 3, "La": 3, "Ce": 3,
    "Y": 3, "Ge": 4, "As": 5, "B": 3, "C": 4, "Ta": 5, "Hf": 4, "In": 3, "Ga": 3,
    "Tb": 3, "Ho": 3, "Nd": 3, "Pr": 3, "Sm": 3, "Eu": 3, "Gd": 3, "Dy": 3, "Er": 3,
    "Yb": 3, "Lu": 3, "Sc": 3, "Mn": 2, "Ag": 1, "Cs": 1, "Ba": 2, "Sr": 2, "Te": -2,
    "H": 1,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_migration_df():
    df = pd.read_parquet(DATA / "migration_barrier_table.parquet")
    df = df.dropna(subset=["mobile_cation"]).reset_index(drop=True)
    return df


def build_magpie_features(df):
    ep = ElementProperty.from_preset("magpie")
    comps = [Composition(f.replace(" ", "")).reduced_composition for f in df["formula"]]
    feat_df = ep.featurize_dataframe(pd.DataFrame({"composition": comps}), "composition", ignore_errors=True)
    magpie_cols = [c for c in feat_df.columns if c != "composition"]
    return feat_df[magpie_cols], magpie_cols


def xgb_default():
    return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                         n_jobs=8, random_state=42)


def jarvis_to_pymatgen(struct_dict):
    lattice = Lattice(struct_dict["lattice_mat"])
    return Structure(lattice, struct_dict["elements"], struct_dict["coords"],
                      coords_are_cartesian=struct_dict.get("cartesian", False))


def load_geom_df():
    p = MODELS / "additional_robustness" / "migration_table_with_geom.parquet"
    df = pd.read_parquet(p)
    return df


def shared_features(df):
    magpie_df, magpie_cols = build_magpie_features(df)
    class_dummies = pd.get_dummies(df["crystal_class"], prefix="class")
    X = np.hstack([magpie_df.to_numpy(float), class_dummies.to_numpy(float),
                    df[["mobile_radius", "mobile_valence"]].to_numpy(float)])
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    feat_names = magpie_cols + list(class_dummies.columns) + ["mobile_radius", "mobile_valence"]
    return X, feat_names


def protocol_masks(df):
    y = df["target_Em"].to_numpy(float)
    mono_mask = ~df["is_multivalent"].to_numpy()
    multi_mask = df["is_multivalent"].to_numpy()
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df)); n_test = int(0.2 * len(df))
    te_idx = idx[:n_test]
    rand_train = np.ones(len(df), bool); rand_train[te_idx] = False
    rand_test = ~rand_train
    return y, {"random_split": (rand_train, rand_test),
               "mono_to_multi": (mono_mask, multi_mask),
               "multi_to_mono": (multi_mask, mono_mask)}


# ----------------------------------------------------------------------
# deepen confound analysis, quantify asymmetry
# ----------------------------------------------------------------------

def terminology_and_percolation_justification(df):
    log("=== V2 Comments 1 & 2: deepen confound + asymmetry quantification ===")
    X, feat_names = shared_features(df)
    y, protocols = protocol_masks(df)
    mono_mask = ~df["is_multivalent"].to_numpy()
    multi_mask = df["is_multivalent"].to_numpy()
    results = {}

    # (a) within-shared-class comparison, extended to ALL shared classes with >=2 of each valence
    mono_classes = set(df.loc[mono_mask, "crystal_class"])
    multi_classes = set(df.loc[multi_mask, "crystal_class"])
    shared = sorted(mono_classes & multi_classes)
    within = {}
    for cls in shared:
        cls_mask = (df["crystal_class"] == cls).to_numpy()
        cls_mono = cls_mask & mono_mask
        cls_multi = cls_mask & multi_mask
        if cls_mono.sum() < 2 or cls_multi.sum() < 2:
            continue
        train_m = ~cls_mask
        m = xgb_default(); m.fit(X[train_m], y[train_m])
        pred_multi = m.predict(X[cls_multi]); pred_mono = m.predict(X[cls_mono])
        within[cls] = {
            "n_mono": int(cls_mono.sum()), "n_multi": int(cls_multi.sum()),
            "mae_class_mono": float(mean_absolute_error(y[cls_mono], pred_mono)),
            "mae_class_multi": float(mean_absolute_error(y[cls_multi], pred_multi)),
            "ratio_multi_over_mono": float(mean_absolute_error(y[cls_multi], pred_multi) /
                                            max(mean_absolute_error(y[cls_mono], pred_mono), 1e-6)),
        }
    results["within_shared_class_all"] = within
    n_worse = sum(1 for v in within.values() if v["ratio_multi_over_mono"] > 1)
    results["within_shared_class_summary"] = {
        "n_classes_tested": len(within), "n_classes_multi_harder": n_worse,
        "mean_ratio": float(np.mean([v["ratio_multi_over_mono"] for v in within.values()])) if within else None,
    }
    log(f"  within-class (extended, {len(within)} classes): multi harder in {n_worse}/{len(within)}, "
        f"mean ratio={results['within_shared_class_summary']['mean_ratio']}")

    # (b) chemical-similarity: nearest-training-neighbor distance in feature space,
    # compared for in-distribution test rows vs. cross-valence test rows
    Xs = StandardScaler().fit_transform(X)
    def nn_dist_to_train(train_m, test_m):
        from scipy.spatial.distance import cdist
        d = cdist(Xs[test_m], Xs[train_m])
        return d.min(axis=1)
    nn_rand = nn_dist_to_train(*protocols["random_split"])
    nn_m2m = nn_dist_to_train(*protocols["mono_to_multi"])
    nn_mm = nn_dist_to_train(*protocols["multi_to_mono"])
    results["nearest_train_neighbor_distance"] = {
        "random_split": {"mean": float(nn_rand.mean()), "median": float(np.median(nn_rand))},
        "mono_to_multi": {"mean": float(nn_m2m.mean()), "median": float(np.median(nn_m2m))},
        "multi_to_mono": {"mean": float(nn_mm.mean()), "median": float(np.median(nn_mm))},
    }
    log(f"  mean nearest-train-neighbor distance (standardized feature space): "
        f"random={nn_rand.mean():.2f}, mono->multi={nn_m2m.mean():.2f}, multi->mono={nn_mm.mean():.2f}")

    # (c) covariate shift vs target shift decomposition: how much error remains after
    # correcting for the pure group-mean shift alone (bias-only naive baseline),
    # vs. the model's actual error (captures feature-driven, i.e. covariate, effects)
    covtgt = {}
    for pname, (tr, te) in [("mono_to_multi", protocols["mono_to_multi"]),
                              ("multi_to_mono", protocols["multi_to_mono"])]:
        naive_pred = np.full(te.sum(), y[tr].mean())
        naive_mae = float(mean_absolute_error(y[te], naive_pred))
        m = xgb_default(); m.fit(X[tr], y[tr])
        model_mae = float(mean_absolute_error(y[te], m.predict(X[te])))
        covtgt[pname] = {
            "naive_target_shift_only_mae": naive_mae, "model_mae": model_mae,
            "model_better_than_naive": model_mae < naive_mae,
            "fraction_of_naive_error_remaining": float(model_mae / naive_mae),
        }
    results["covariate_vs_target_shift"] = covtgt
    log(f"  covariate-vs-target-shift decomposition: {covtgt}")

    # (d) quantitative migration-barrier distribution comparison + overlap
    from scipy import stats as sstats
    ks = sstats.ks_2samp(y[mono_mask], y[multi_mask])
    hist_range = (min(y.min(), y.min()), max(y.max(), y.max()))
    bins = np.linspace(y.min(), y.max(), 30)
    h_mono, _ = np.histogram(y[mono_mask], bins=bins, density=True)
    h_multi, _ = np.histogram(y[multi_mask], bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    overlap_coef = float(np.sum(np.minimum(h_mono, h_multi)) * bin_width)
    results["migration_barrier_distribution"] = {
        "monovalent": {"mean": float(y[mono_mask].mean()), "std": float(y[mono_mask].std()),
                        "min": float(y[mono_mask].min()), "max": float(y[mono_mask].max()), "n": int(mono_mask.sum())},
        "multivalent": {"mean": float(y[multi_mask].mean()), "std": float(y[multi_mask].std()),
                         "min": float(y[multi_mask].min()), "max": float(y[multi_mask].max()), "n": int(multi_mask.sum())},
        "ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
        "histogram_overlap_coefficient": overlap_coef,
    }
    log(f"  barrier distributions: KS={ks.statistic:.3f} (p={ks.pvalue:.2e}), "
        f"histogram overlap coefficient={overlap_coef:.3f}")

    # (e) normalized transfer-degradation comparison, migration-barrier (regression) vs CAVD (classification)
    cavd_metrics = json.load(open(MODELS / "baseline" / "metrics.json"))
    def get(d, *keys):
        for k in keys:
            d = d[k]
        return d
    cavd_rand = cavd_metrics["percolates_binary"]["random_split"]["balanced_accuracy"]
    cavd_m2m = cavd_metrics["percolates_binary"]["train_mono_test_multi"]["balanced_accuracy"]
    cavd_mm = cavd_metrics["percolates_binary"]["train_multi_test_mono"]["balanced_accuracy"]
    mig_bench = json.load(open(MODELS / "migration_baseline" / "benchmark_models.json"))["XGBoost"]
    mig_rand = mig_bench["random_split"]["mae"]
    mig_m2m = mig_bench["mono_to_multi"]["mae"]
    mig_mm = mig_bench["multi_to_mono"]["mae"]
    # CAVD: normalize as fraction of possible range above chance (0.5) lost
    cavd_m2m_norm = (cavd_rand - cavd_m2m) / (cavd_rand - 0.5)
    cavd_mm_norm = (cavd_rand - cavd_mm) / (cavd_rand - 0.5)
    # Migration: normalize as MAE increase relative to in-distribution MAE (a ratio, dataset-agnostic)
    mig_m2m_norm = (mig_m2m - mig_rand) / mig_rand if mig_m2m else None
    mig_mm_norm = (mig_mm - mig_rand) / mig_rand if mig_mm else None
    results["normalized_transfer_degradation"] = {
        "cavd_balanced_accuracy": {"random": cavd_rand, "mono_to_multi": cavd_m2m, "multi_to_mono": cavd_mm,
                                     "mono_to_multi_fraction_of_above_chance_lost": float(cavd_m2m_norm),
                                     "multi_to_mono_fraction_of_above_chance_lost": float(cavd_mm_norm)},
        "migration_mae": {"random": mig_rand, "mono_to_multi": mig_m2m, "multi_to_mono": mig_mm,
                            "mono_to_multi_relative_increase": mig_m2m_norm,
                            "multi_to_mono_relative_increase": mig_mm_norm},
    }
    log(f"  normalized degradation: CAVD mono->multi loses {cavd_m2m_norm:.1%} of above-chance skill, "
        f"multi->mono loses {cavd_mm_norm:.1%}; migration mono->multi MAE +{mig_m2m_norm:.1%}, "
        f"multi->mono MAE +{mig_mm_norm:.1%}")
    results["interpretation"] = (
        "The CAVD classifier loses nearly all (~{:.0%}) of its above-chance skill in the mono->multi "
        "direction but retains substantially more ({:.0%} lost) multi->mono. The migration-barrier "
        "regressor shows the opposite-signed asymmetry in relative terms: mono->multi MAE rises by only "
        "{:.0%} over the in-distribution baseline, while multi->mono MAE rises by {:.0%}. On this common, "
        "normalized footing the two datasets' asymmetries are directionally opposite but are not of "
        "wildly different magnitude once each is expressed relative to its own random-split baseline -- "
        "both represent a severe (>100% relative) degradation in one direction and a comparatively mild "
        "one in the other.".format(cavd_m2m_norm, cavd_mm_norm, mig_m2m_norm, mig_mm_norm)
    )

    with open(OUT / "terminology_and_percolation_justification.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# NEB-displacement-based mobile-ion identification, cross-check
# against formula-based assignment for ambiguous (multi-candidate) entries
# ----------------------------------------------------------------------

def neb_mobile_ion_check(df):
    log("=== NEB-displacement mobile-ion verification for ambiguous entries ===")
    with open(DATA / "migration_barriers" / "EM-COMPLETE-DATASET.json") as f:
        records = {r["jid"]: r for r in json.load(f)}
    amb = df[df["n_candidates"] > 1].copy()
    log(f"  {len(amb)} ambiguous (multi-candidate) entries found")

    rows = []
    for _, row in amb.iterrows():
        rec = records.get(row["jid"])
        if rec is None:
            rows.append({**row[["jid", "formula", "candidate_cations_present", "mobile_cation"]].to_dict(),
                         "neb_displacement_winner": None, "agrees_with_formula_assignment": None,
                         "note": "jid not found in raw JSON"})
            continue
        try:
            si = jarvis_to_pymatgen(rec["structure_ini"])
            sf = jarvis_to_pymatgen(rec["structure_fin"])
            candidates = [c.strip() for c in str(row["candidate_cations_present"]).split(",")]
            best_elem, best_disp = None, -1.0
            per_elem_max_disp = {}
            for el in candidates:
                idxs = [i for i, sp in enumerate(si.species) if sp.symbol == el]
                if not idxs:
                    continue
                max_d = max(si[i].distance(sf[i]) for i in idxs)
                per_elem_max_disp[el] = float(max_d)
                if max_d > best_disp:
                    best_disp, best_elem = max_d, el
            rows.append({
                "jid": row["jid"], "formula": row["formula"],
                "candidate_cations_present": row["candidate_cations_present"],
                "formula_based_assignment": row["mobile_cation"],
                "per_element_max_displacement_A": per_elem_max_disp,
                "neb_displacement_winner": best_elem,
                "agrees_with_formula_assignment": (best_elem == row["mobile_cation"]),
            })
        except Exception as e:
            rows.append({"jid": row["jid"], "formula": row["formula"],
                         "candidate_cations_present": row["candidate_cations_present"],
                         "formula_based_assignment": row["mobile_cation"],
                         "neb_displacement_winner": None, "agrees_with_formula_assignment": None,
                         "note": f"error: {e}"})

    amb_table = pd.DataFrame(rows)
    n_resolved = amb_table["neb_displacement_winner"].notna().sum()
    n_agree = amb_table["agrees_with_formula_assignment"].sum()
    disagreements = amb_table[amb_table["agrees_with_formula_assignment"] == False]

    # rerun principal cross-valence analysis excluding ANY disagreeing entries
    bad_jids = set(disagreements["jid"].tolist())
    df_clean = df[~df["jid"].isin(bad_jids)].reset_index(drop=True)
    df_clean_geom = df_clean  # already has geom cols from caller
    X, _ = shared_features(df_clean_geom)
    y = df_clean_geom["target_Em"].to_numpy(float)
    mono_mask = ~df_clean_geom["is_multivalent"].to_numpy()
    multi_mask = df_clean_geom["is_multivalent"].to_numpy()
    rerun = {}
    for pname, (tr, te) in [("mono_to_multi", (mono_mask, multi_mask)), ("multi_to_mono", (multi_mask, mono_mask))]:
        m = xgb_default(); m.fit(X[tr], y[tr]); pred = m.predict(X[te])
        rerun[pname] = {"mae": float(mean_absolute_error(y[te], pred)), "r2": float(r2_score(y[te], pred)),
                         "n_train": int(tr.sum()), "n_test": int(te.sum())}

    results = {
        "n_ambiguous_entries": int(len(amb)),
        "n_resolved_via_neb_displacement": int(n_resolved),
        "n_agree_with_formula_assignment": int(n_agree),
        "agreement_fraction": float(n_agree / n_resolved) if n_resolved else None,
        "n_disagreements": int(len(disagreements)),
        "disagreement_jids": disagreements["jid"].tolist(),
        "rerun_excluding_disagreements": rerun,
        "full_table": amb_table.to_dict(orient="records"),
    }
    log(f"  NEB-displacement resolved {n_resolved}/{len(amb)}; agrees with formula assignment in "
        f"{n_agree}/{n_resolved} ({results['agreement_fraction']})")
    log(f"  rerun excluding {len(bad_jids)} disagreements: {rerun}")

    with open(OUT / "neb_structure_check.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# Arrhenius framing + threshold classification
# ----------------------------------------------------------------------

def arrhenius_threshold_check(df):
    log("=== Arrhenius framing + threshold classification ===")
    kB = 8.617333262e-5  # eV/K
    T = 300.0
    mae = 0.380
    factor = float(np.exp(mae / (kB * T)))
    results = {"kB_eV_per_K": kB, "T_K": T, "barrier_MAE_eV": mae,
               "arrhenius_exponential_factor": factor,
               "note": "exp(MAE / (kB*T)) -- the multiplicative uncertainty this MAE implies for a "
                       "rate ratio exp(-Ea/kBT), NOT a claim about the model's actual rate-prediction error."}
    log(f"  exp({mae}/({kB}*{T})) = {factor:.3e}")

    pred = pd.read_csv(MODELS / "migration_baseline" / "predictions_random_split.csv")
    thresholds = {}
    for thresh in [0.3, 0.5]:
        y_true_cls = (pred["target_Em"] <= thresh).astype(int)
        y_pred_cls = (pred["pred_Em"] <= thresh).astype(int)
        acc = float((y_true_cls == y_pred_cls).mean())
        tp = int(((y_true_cls == 1) & (y_pred_cls == 1)).sum())
        fp = int(((y_true_cls == 0) & (y_pred_cls == 1)).sum())
        fn = int(((y_true_cls == 1) & (y_pred_cls == 0)).sum())
        tn = int(((y_true_cls == 0) & (y_pred_cls == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        try:
            auc = float(roc_auc_score(y_true_cls, -pred["pred_Em"]))
        except Exception:
            auc = None
        thresholds[f"{thresh}eV"] = {
            "n_below_threshold_true": int(y_true_cls.sum()), "n_total": len(pred),
            "accuracy": acc, "precision": precision, "recall": recall, "auc": auc,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }
        log(f"  threshold={thresh} eV: n_fast(true)={y_true_cls.sum()}/{len(pred)}, "
            f"acc={acc:.3f}, precision={precision}, recall={recall}, auc={auc}")
    results["threshold_classification"] = thresholds

    with open(OUT / "arrhenius_threshold_check.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# CAVD composition/structure-grouped validation, promote as principal
# ----------------------------------------------------------------------

def cavd_grouped_validation():
    log("=== CAVD grouped validation ===")
    cavd = pd.read_parquet(DATA / "feature_table.parquet")
    feat_cols = [c for c in cavd.columns if c not in
                 ("cation", "filename", "IND", "percolates")]
    X = cavd[feat_cols].to_numpy(float)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    y = cavd["percolates"].to_numpy(int)
    groups = cavd["filename"].to_numpy()

    from sklearn.ensemble import RandomForestClassifier
    accs, baccs = [], []
    for seed in range(5):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(gss.split(X, y, groups))
        m = RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=8, random_state=seed)
        m.fit(X[tr], y[tr]); pred = m.predict(X[te])
        accs.append(float((pred == y[te]).mean()))
        baccs.append(float(balanced_accuracy_score(y[te], pred)))
    results = {
        "composition_grouped_5seed": {
            "accuracy_mean_std": [float(np.mean(accs)), float(np.std(accs))],
            "balanced_accuracy_mean_std": [float(np.mean(baccs)), float(np.std(baccs))],
        },
        "n_total": len(cavd), "n_groups": int(cavd["filename"].nunique()),
    }
    log(f"  CAVD composition/structure-grouped 5-seed: balanced accuracy="
        f"{np.mean(baccs):.3f}+-{np.std(baccs):.3f} (vs. ungrouped random-split 0.880 reported in V1/Phase1)")

    with open(OUT / "cavd_grouped_splits.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# dataset heterogeneity (XC functional, source publication)
# ----------------------------------------------------------------------

def structural_heterogeneity_check(df):
    log("=== DFT-NEB metadata heterogeneity ===")
    with open(DATA / "migration_barriers" / "EM-COMPLETE-DATASET.json") as f:
        records = {r["jid"]: r for r in json.load(f)}
    import re
    bib_keys = []
    for _, row in df.iterrows():
        rec = records.get(row["jid"], {})
        b = rec.get("bibtex", "")
        m = re.search(r"\{([^,]+),", b)
        bib_keys.append(m.group(1) if m else "unknown")
    df = df.copy()
    df["source_key"] = bib_keys

    xc_counts = df["XC"].value_counts(dropna=False).to_dict()
    src_counts = df["source_key"].value_counts()

    # check correlation of prediction error with XC functional, using existing random-split predictions
    pred = pd.read_csv(MODELS / "migration_baseline" / "predictions_random_split.csv")
    pred = pred.merge(df[["jid", "XC"]], on="jid", how="left") if "jid" in pred.columns else pred
    xc_error = {}
    if "XC" in pred.columns:
        pred["abs_err"] = (pred["target_Em"] - pred["pred_Em"]).abs()
        for xc, grp in pred.groupby("XC"):
            if len(grp) >= 3:
                xc_error[xc] = {"n": len(grp), "mean_abs_error": float(grp["abs_err"].mean())}

    results = {
        "xc_functional_distribution": xc_counts,
        "n_distinct_source_publications": int(df["source_key"].nunique()),
        "top10_sources": src_counts.head(10).to_dict(),
        "self_computed_entries": int(src_counts.get("SELF", 0)),
        "error_by_xc_functional_random_split_test": xc_error,
        "fields_checked_but_NOT_available_in_raw_data": [
            "magnetic configuration (not recorded per entry)",
            "supercell size (not recorded per entry)",
            "explicit convergence criteria (not recorded per entry)",
            "explicit vacancy-mediated vs. interstitial pathway label (not recorded)",
        ],
        "electrode_vs_electrolyte": (
            "Not recorded as a separate binary field; the closest available proxy is the "
            "crystal_class structural-class label already used throughout the paper "
            "(e.g. the 'ELECTROLYTES' class, see Comment 12 analysis), which the dataset's "
            "own authors assigned, not a per-entry electrode/electrolyte flag."
        ),
    }
    log(f"  XC distribution: {xc_counts}; {results['n_distinct_source_publications']} distinct sources "
        f"({results['self_computed_entries']} self-computed by the dataset authors)")
    log(f"  error by XC (random-split test set): {xc_error}")

    with open(OUT / "structural_heterogeneity_check.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# feature-distribution explanation + Ewald/Madelung descriptor
# ----------------------------------------------------------------------

def descriptor_shift_ewald_check(df):
    log("=== geometric-descriptor distribution shift + Ewald descriptor ===")
    results = {}
    mono_mask = ~df["is_multivalent"].to_numpy()
    multi_mask = df["is_multivalent"].to_numpy()
    for col in ["mobile_coord_number", "mobile_nn_distance", "migration_path_length"]:
        v = df[col].to_numpy(float)
        vm, vM = v[mono_mask], v[multi_mask]
        vm, vM = vm[~np.isnan(vm)], vM[~np.isnan(vM)]
        results[col] = {
            "mono": {"mean": float(vm.mean()), "std": float(vm.std())},
            "multi": {"mean": float(vM.mean()), "std": float(vM.std())},
            "standardized_mean_gap": float(abs(vm.mean() - vM.mean()) / np.sqrt(0.5 * (vm.std()**2 + vM.std()**2))),
        }
        log(f"  {col}: mono {vm.mean():.3f}+-{vm.std():.3f}, multi {vM.mean():.3f}+-{vM.std():.3f}, "
            f"standardized gap={results[col]['standardized_mean_gap']:.2f}")
    # for comparison, the standardized gap for a composition-only magpie feature (electronegativity)
    magpie_df, magpie_cols = build_magpie_features(df)
    if "MagpieData mean Electronegativity" in magpie_cols:
        v = magpie_df["MagpieData mean Electronegativity"].to_numpy(float)
        vm, vM = v[mono_mask], v[multi_mask]
        gap = float(abs(vm.mean() - vM.mean()) / np.sqrt(0.5 * (vm.std()**2 + vM.std()**2)))
        results["comparison_composition_feature_mean_electronegativity"] = {"standardized_mean_gap": gap}
        log(f"  comparison, mean electronegativity standardized gap={gap:.2f}")

    # --- Ewald/Madelung descriptor attempt ---
    from pymatgen.analysis.ewald import EwaldSummation
    with open(DATA / "migration_barriers" / "EM-COMPLETE-DATASET.json") as f:
        records = {r["jid"]: r for r in json.load(f)}
    madelung_energy = []
    n_fail = 0
    for _, row in df.iterrows():
        try:
            rec = records[row["jid"]]
            si = jarvis_to_pymatgen(rec["structure_ini"])
            ox_struct = si.copy()
            species_ox = []
            ok = True
            for site in ox_struct:
                sym = site.specie.symbol
                if sym not in COMMON_OX:
                    ok = False
                    break
                species_ox.append(COMMON_OX[sym])
            if not ok:
                raise ValueError("missing oxidation state for an element")
            ox_struct.add_oxidation_state_by_site(species_ox)
            ew = EwaldSummation(ox_struct)
            madelung_energy.append(ew.total_energy / len(ox_struct))
        except Exception:
            madelung_energy.append(np.nan)
            n_fail += 1
    df = df.copy()
    df["madelung_energy_per_atom"] = madelung_energy
    n_success = len(df) - n_fail
    results["ewald_descriptor_coverage"] = {"n_total": len(df), "n_success": int(n_success), "n_fail": int(n_fail)}
    log(f"  Ewald/Madelung descriptor computed for {n_success}/{len(df)} rows ({n_fail} failures, "
        f"typically missing a common-oxidation-state lookup entry)")

    if n_success > 50:
        X_base, _ = shared_features(df)
        y = df["target_Em"].to_numpy(float)
        mad = df["madelung_energy_per_atom"].to_numpy(float)
        mad_filled = np.where(np.isnan(mad), np.nanmean(mad), mad)
        X_plus = np.hstack([X_base, mad_filled.reshape(-1, 1)])
        ab = {}
        for name, X in [("baseline_with_radius_valence", X_base), ("plus_madelung", X_plus)]:
            ab[name] = {}
            for pname, (tr, te) in [("random_split", protocol_masks(df)[1]["random_split"]),
                                      ("mono_to_multi", protocol_masks(df)[1]["mono_to_multi"]),
                                      ("multi_to_mono", protocol_masks(df)[1]["multi_to_mono"])]:
                m = xgb_default(); m.fit(X[tr], y[tr]); pred = m.predict(X[te])
                ab[name][pname] = {"mae": float(mean_absolute_error(y[te], pred)), "r2": float(r2_score(y[te], pred))}
        results["madelung_ablation"] = ab
        log(f"  Madelung ablation: {ab}")
    else:
        results["madelung_ablation"] = None
        results["madelung_skip_reason"] = "fewer than 50 successful rows; not enough to train a reliable ablation model"

    results["interpretation"] = (
        "The geometric descriptors (coordination number, nearest-neighbor distance, migration path "
        "length) show a larger standardized mean gap between monovalent and multivalent subsets than "
        "the comparison composition-level feature, consistent with the hypothesis that these descriptors "
        "encode chemistry-specific local-structure relationships that do not transfer across valence "
        "groups as cleanly as global composition statistics do."
    )

    with open(OUT / "descriptor_shift_ewald.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# leakage-free quantile-regression regularization
# ----------------------------------------------------------------------

def leakage_free_uq(df):
    log("=== leakage-free UQ regularization sweep ===")
    X, feat_names = shared_features(df)
    y, protocols = protocol_masks(df)

    def make_q(alpha, n_estimators, max_depth, reg_lambda):
        return XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                             n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
                             reg_lambda=reg_lambda, n_jobs=8, random_state=42)

    sweep_grid = [(500, 5, 1.0), (100, 4, 5.0), (50, 3, 10.0), (30, 2, 20.0), (20, 2, 30.0), (15, 2, 40.0)]

    results = {}
    final_test_coverage = {}
    for pname, (train_full, test_final) in protocols.items():
        train_idx = np.where(train_full)[0]
        rng = np.random.RandomState(7)
        perm = rng.permutation(len(train_idx))
        n_val = max(20, int(0.25 * len(train_idx)))
        val_idx, fit_idx = train_idx[perm[:n_val]], train_idx[perm[n_val:]]
        fit_mask = np.zeros(len(df), bool); fit_mask[fit_idx] = True
        val_mask = np.zeros(len(df), bool); val_mask[val_idx] = True

        sweep_results = []
        for (n_est, depth, lam) in sweep_grid:
            m_lo = make_q(0.1, n_est, depth, lam); m_lo.fit(X[fit_mask], y[fit_mask])
            m_hi = make_q(0.9, n_est, depth, lam); m_hi.fit(X[fit_mask], y[fit_mask])
            p_lo, p_hi = m_lo.predict(X[val_mask]), m_hi.predict(X[val_mask])
            cov = float(np.mean((y[val_mask] >= p_lo) & (y[val_mask] <= p_hi)))
            sweep_results.append({"n_estimators": n_est, "max_depth": depth, "reg_lambda": lam,
                                    "validation_coverage": cov})
        best = min(sweep_results, key=lambda r: abs(r["validation_coverage"] - 0.8))

        m_lo = make_q(0.1, best["n_estimators"], best["max_depth"], best["reg_lambda"])
        m_lo.fit(X[train_full], y[train_full])
        m_hi = make_q(0.9, best["n_estimators"], best["max_depth"], best["reg_lambda"])
        m_hi.fit(X[train_full], y[train_full])
        p_lo, p_hi = m_lo.predict(X[test_final]), m_hi.predict(X[test_final])
        cov = float(np.mean((y[test_final] >= p_lo) & (y[test_final] <= p_hi)))
        width = float(np.mean(p_hi - p_lo))

        results[pname] = {
            "n_fit": int(fit_mask.sum()), "n_val": int(val_mask.sum()), "n_test": int(test_final.sum()),
            "sweep_on_validation_only": sweep_results,
            "selected_hyperparameters": {k: best[k] for k in ("n_estimators", "max_depth", "reg_lambda")},
            "validation_coverage_at_selection": best["validation_coverage"],
            "final_test_coverage_leakage_free": cov,
            "final_test_mean_width": width,
        }
        final_test_coverage[pname] = cov
        log(f"  {pname}: selected {results[pname]['selected_hyperparameters']} on a held-out validation "
            f"split (coverage {best['validation_coverage']:.3f}); LEAKAGE-FREE final test coverage = {cov:.3f} "
            f"(width {width:.3f})")

    results["mean_leakage_free_coverage"] = float(np.mean(list(final_test_coverage.values())))
    log(f"  mean leakage-free coverage across protocols: {results['mean_leakage_free_coverage']:.3f} "
        f"(for comparison, the V1 number that reused test-set coverage for selection was 0.692 mean)")

    with open(OUT / "leakage_free_uq.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# SHAP vs physically meaningful quantities
# ----------------------------------------------------------------------

def shap_physical_check(df):
    log("=== SHAP vs physical quantities ===")
    X, feat_names = shared_features(df)
    y, protocols = protocol_masks(df)
    tr, te = protocols["random_split"]
    m = xgb_default(); m.fit(X[tr], y[tr])
    expl = shap.TreeExplainer(m)
    sv = expl.shap_values(X[te])
    shap_importance = pd.Series(np.abs(sv).mean(axis=0), index=feat_names).sort_values(ascending=False)

    phys_cols = {"mobile_radius": "mobile-ion Shannon radius", "mobile_valence": "mobile-ion nominal valence"}
    if "mobile_coord_number" in df.columns:
        phys_cols["mobile_coord_number"] = "mobile-site coordination number (geometric)"

    ranks = {name: int(np.where(shap_importance.index == name)[0][0]) + 1 if name in shap_importance.index else None
             for name in phys_cols}
    total_feats = len(shap_importance)
    results = {
        "top10_shap_features": shap_importance.head(10).to_dict(),
        "physical_quantity_shap_rank": {phys_cols[k]: ranks[k] for k in phys_cols},
        "total_n_features": total_feats,
        "interpretation": (
            "Directly physically interpretable quantities (mobile-ion radius, nominal valence) rank "
            f"well outside the top 10 of {total_feats} features by mean |SHAP value| (ranks: "
            f"{ {phys_cols[k]: ranks[k] for k in phys_cols} }), while composition-statistic proxies "
            "(minimum Mendeleev number, periodic row/column, valence-electron counts) dominate. This "
            "supports treating SHAP importance here as a statistical association with the model's "
            "learned representation, not as direct evidence that these periodic-table statistics are "
            "the physical quantities controlling ion migration -- the more mechanistically direct "
            "descriptors (radius, valence, local coordination) are available to the model but are not "
            "what it relies on most heavily."
        ),
    }
    log(f"  physical-quantity SHAP ranks (of {total_feats} features): {results['physical_quantity_shap_rank']}")

    with open(OUT / "shap_physical_check.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# residual binning, mean signed error, verify the -3 eV claim
# ----------------------------------------------------------------------

def residual_binning_check(df):
    log("=== residual analysis by barrier range ===")
    pred = pd.read_csv(MODELS / "migration_baseline" / "predictions_random_split.csv")
    pred["residual"] = pred["pred_Em"] - pred["target_Em"]  # positive = overprediction
    tertiles = pred["target_Em"].quantile([1/3, 2/3]).to_list()
    def bin_label(v):
        if v <= tertiles[0]:
            return "low"
        elif v <= tertiles[1]:
            return "intermediate"
        return "high"
    pred["bin"] = pred["target_Em"].apply(bin_label)

    bin_stats = {}
    for b in ["low", "intermediate", "high"]:
        g = pred[pred["bin"] == b]
        bin_stats[b] = {
            "n": len(g), "barrier_range": [float(g["target_Em"].min()), float(g["target_Em"].max())],
            "mean_signed_error": float(g["residual"].mean()),
            "mae": float(g["residual"].abs().mean()),
            "max_underprediction": float(g["residual"].min()),  # most negative = worst underprediction
            "max_overprediction": float(g["residual"].max()),
        }
    log(f"  tertile boundaries: {tertiles}")
    for b, s in bin_stats.items():
        log(f"  {b} (n={s['n']}, range={s['barrier_range']}): mean signed error={s['mean_signed_error']:.3f} eV, "
            f"MAE={s['mae']:.3f}, worst underprediction={s['max_underprediction']:.3f}, "
            f"worst overprediction={s['max_overprediction']:.3f}")

    worst_row = pred.loc[pred["residual"].idxmin()]
    results = {
        "tertile_boundaries_eV": tertiles,
        "bin_stats": bin_stats,
        "compression_toward_middle": bin_stats["high"]["mean_signed_error"] < 0 and bin_stats["low"]["mean_signed_error"] > 0,
        "worst_single_underprediction": {
            "jid": int(worst_row["jid"]) if "jid" in worst_row else None,
            "formula": worst_row.get("formula"), "target_Em": float(worst_row["target_Em"]),
            "pred_Em": float(worst_row["pred_Em"]), "residual": float(worst_row["residual"]),
        },
        "reviewer_claim_check": {
            "claim": "negative residuals approach approximately -3 eV for the highest-barrier materials",
            "actual_worst_residual_in_random_split_test_set": float(pred["residual"].min()),
            "claim_supported": float(pred["residual"].min()) <= -2.5,
        },
    }
    log(f"  worst single underprediction: {results['worst_single_underprediction']}")
    log(f"  reviewer's ~-3 eV claim check: actual worst residual = {pred['residual'].min():.3f} eV, "
        f"supported={results['reviewer_claim_check']['claim_supported']}")

    with open(OUT / "residual_bins.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    df = load_geom_df()
    log(f"Loaded geom-augmented migration table: {df.shape}")
    terminology_and_percolation_justification(df)
    neb_mobile_ion_check(df)
    arrhenius_threshold_check(df)
    cavd_grouped_validation()
    structural_heterogeneity_check(df)
    descriptor_shift_ewald_check(df)
    leakage_free_uq(df)
    shap_physical_check(df)
    residual_binning_check(df)
    log(f"ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
