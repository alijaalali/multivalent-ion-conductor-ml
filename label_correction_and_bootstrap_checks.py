# Mobile-cation label correction, CAVD ionic-radius ablation, per-protocol bootstrap CIs,
# CAVD duplicate-structure check, and bootstrap CIs for UQ coverage. Reanalysis of data
# already on disk (NEB endpoint structures, CAVD geometric descriptors, migration-barrier
# labels) plus additional ML runs; no new DFT/atomistic simulation.
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, r2_score,
                              balanced_accuracy_score)
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

PROJ = Path("/glade/work/ajalali/12 - Multivalent-Ion-Conductor-ML-project")
DATA = PROJ / "data"
MODELS = PROJ / "models"
OUT = PROJ / "models" / "label_correction_and_bootstrap"
OUT.mkdir(parents=True, exist_ok=True)

VALENCE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Mg": 2, "Ca": 2, "Zn": 2, "Al": 3, "Sr": 2}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def xgb_default():
    return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                         n_jobs=8, random_state=42)


def build_magpie_features(df):
    ep = ElementProperty.from_preset("magpie")
    comps = [Composition(f.replace(" ", "")).reduced_composition for f in df["formula"]]
    feat_df = ep.featurize_dataframe(pd.DataFrame({"composition": comps}), "composition", ignore_errors=True)
    magpie_cols = [c for c in feat_df.columns if c != "composition"]
    return feat_df[magpie_cols], magpie_cols


def migration_features(df):
    magpie_df, magpie_cols = build_magpie_features(df)
    class_dummies = pd.get_dummies(df["crystal_class"], prefix="class")
    X = np.hstack([magpie_df.to_numpy(float), class_dummies.to_numpy(float)])
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    return X, magpie_cols + list(class_dummies.columns)


def bootstrap_diff_ci(y_true_a, pred_a, y_true_b, pred_b, metric, n_boot=2000, seed=42):
    """Bootstrap CI for metric(b) - metric(a), resampling each test set with replacement."""
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        ia = rng.randint(0, len(y_true_a), len(y_true_a))
        ib = rng.randint(0, len(y_true_b), len(y_true_b))
        diffs.append(metric(y_true_b[ib], pred_b[ib]) - metric(y_true_a[ia], pred_a[ia]))
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(diffs.mean())


def bootstrap_single_ci(y_true, pred, metric, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        vals.append(metric(y_true[idx], pred[idx]))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(vals.mean())


# ----------------------------------------------------------------------
# corrected mobile-cation table + NEB-preferred recalculation
# ----------------------------------------------------------------------

def corrected_mobile_cation_labels():
    log("=== fix SI Table S10 + recalculate with NEB-preferred labels ===")
    with open(MODELS / "physical_descriptors_and_uq" / "neb_structure_check.json") as f:
        neb = json.load(f)
    full_table = pd.DataFrame(neb["full_table"])
    # sanity: verify the 10 disagreements are exactly the Na,Ca entries, per direct inspection
    disagree = full_table[full_table["agrees_with_formula_assignment"] == False]
    log(f"  {len(disagree)} disagreements, candidate sets: "
        f"{disagree['candidate_cations_present'].unique().tolist()}")

    df = pd.read_parquet(DATA / "migration_barrier_table.parquet")
    df = df.dropna(subset=["mobile_cation"]).reset_index(drop=True)

    neb_winner_by_jid = dict(zip(full_table["jid"], full_table["neb_displacement_winner"]))
    df["mobile_cation_corrected"] = df.apply(
        lambda r: neb_winner_by_jid.get(r["jid"], r["mobile_cation"]), axis=1)
    n_changed = int((df["mobile_cation_corrected"] != df["mobile_cation"]).sum())
    log(f"  {n_changed} entries' mobile-cation assignment changed after preferring NEB verification")
    assert n_changed == len(disagree), "changed-label count should equal disagreement count"

    df["valence_corrected"] = df["mobile_cation_corrected"].map(VALENCE)
    df["is_multivalent_corrected"] = df["valence_corrected"] >= 2
    n_mono_c = int((~df["is_multivalent_corrected"]).sum())
    n_multi_c = int(df["is_multivalent_corrected"].sum())
    log(f"  corrected group sizes: monovalent {n_mono_c}, multivalent {n_multi_c} "
        f"(was 338/279); total {len(df)} entries unchanged")

    X, _ = migration_features(df)
    y = df["target_Em"].to_numpy(float)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df)); n_test = int(0.2 * len(df))
    te_idx = idx[:n_test]
    rand_train = np.ones(len(df), bool); rand_train[te_idx] = False
    rand_test = ~rand_train

    mono_c = ~df["is_multivalent_corrected"].to_numpy()
    multi_c = df["is_multivalent_corrected"].to_numpy()

    protocols = {"random_split": (rand_train, rand_test),
                 "mono_to_multi": (mono_c, multi_c),
                 "multi_to_mono": (multi_c, mono_c)}
    results = {}
    for name, (tr, te) in protocols.items():
        m = xgb_default(); m.fit(X[tr], y[tr]); pred = m.predict(X[te])
        results[name] = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                          "mae": float(mean_absolute_error(y[te], pred)),
                          "r2": float(r2_score(y[te], pred))}
        log(f"  [corrected labels] {name}: {results[name]}")

    # Build the complete, correct 27-row SI table from the actual data (fixes the
    # fabricated "Ca" / "Mg" entries that appeared in the prior draft's abbreviated table)
    si_rows = []
    for _, r in full_table.sort_values("jid").iterrows():
        si_rows.append({
            "jid": int(r["jid"]), "formula": r["formula"],
            "candidates": r["candidate_cations_present"],
            "formula_based": r["formula_based_assignment"],
            "neb_winner": r["neb_displacement_winner"],
            "agrees": bool(r["agrees_with_formula_assignment"]),
        })

    out = {
        "n_disagreements": len(disagree),
        "disagreement_formulas": sorted(disagree["formula"].unique().tolist()),
        "corrected_group_sizes": {"monovalent": n_mono_c, "multivalent": n_multi_c, "total": len(df)},
        "n_entries_relabeled": n_changed,
        "rerun_with_corrected_labels": results,
        "original_heuristic_results": {
            "random_split": {"mae": 0.380, "r2": 0.431},
            "mono_to_multi": {"mae": 0.518, "r2": 0.040},
            "multi_to_mono": {"mae": 1.049, "r2": -1.775},
        },
        "si_table_full_27_rows": si_rows,
    }
    with open(OUT / "corrected_labels_check.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


# ----------------------------------------------------------------------
# CAVD percolation, ionic-radius ablation
# ----------------------------------------------------------------------

def radius_ablation_check():
    log("=== CAVD percolation, ablate mobile-ion radius ===")
    df = pd.read_parquet(DATA / "feature_table.parquet")
    drop_cols = ["cation", "filename", "IND", "percolates"]
    feature_cols_full = [c for c in df.columns if c not in drop_cols]
    feature_cols_noradius = [c for c in feature_cols_full if c != "mobile_ion_radius_mean"]

    y = df["percolates"].to_numpy()
    mono_mask = df["cation"].isin(["Li", "Na"]).to_numpy()
    multi_mask = df["cation"].isin(["Mg", "Al"]).to_numpy()
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df)); n_test = int(0.2 * len(df))
    te_idx = idx[:n_test]
    rand_train = np.ones(len(df), bool); rand_train[te_idx] = False
    rand_test = ~rand_train
    protocols = {"random_split": (rand_train, rand_test),
                 "mono_to_multi": (mono_mask, multi_mask),
                 "multi_to_mono": (multi_mask, mono_mask)}

    def run(feature_cols):
        X = df[feature_cols].to_numpy(dtype=float)
        X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
        out = {}
        for name, (tr, te) in protocols.items():
            m = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=2,
                                        class_weight="balanced_subsample", n_jobs=8, random_state=42)
            m.fit(X[tr], y[tr]); pred = m.predict(X[te])
            out[name] = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                         "balanced_accuracy": float(balanced_accuracy_score(y[te], pred))}
        return out

    with_radius = run(feature_cols_full)
    without_radius = run(feature_cols_noradius)
    log(f"  with radius: {with_radius}")
    log(f"  without radius: {without_radius}")

    results = {"n_features_with_radius": len(feature_cols_full),
               "n_features_without_radius": len(feature_cols_noradius),
               "with_radius": with_radius, "without_radius": without_radius}
    with open(OUT / "radius_ablation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# per-protocol mean/median baselines + bootstrap CIs
# ----------------------------------------------------------------------

def baseline_confidence_intervals():
    log("=== per-protocol baselines + bootstrap CIs ===")
    df = pd.read_parquet(DATA / "migration_barrier_table.parquet")
    df = df.dropna(subset=["mobile_cation"]).reset_index(drop=True)
    X, _ = migration_features(df)
    y = df["target_Em"].to_numpy(float)

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df)); n_test = int(0.2 * len(df))
    te_idx = idx[:n_test]
    rand_train = np.ones(len(df), bool); rand_train[te_idx] = False
    rand_test = ~rand_train
    mono_mask = ~df["is_multivalent"].to_numpy()
    multi_mask = df["is_multivalent"].to_numpy()
    protocols = {"random_split": (rand_train, rand_test),
                 "mono_to_multi": (mono_mask, multi_mask),
                 "multi_to_mono": (multi_mask, mono_mask)}

    baselines = {}
    model_results = {}
    for name, (tr, te) in protocols.items():
        train_mean = y[tr].mean()
        train_median = np.median(y[tr])
        naive_mean_mae = float(mean_absolute_error(y[te], np.full(te.sum(), train_mean)))
        naive_median_mae = float(mean_absolute_error(y[te], np.full(te.sum(), train_median)))
        baselines[name] = {"train_mean": float(train_mean), "train_median": float(train_median),
                            "naive_mean_predictor_mae": naive_mean_mae,
                            "naive_median_predictor_mae": naive_median_mae}

        m = xgb_default(); m.fit(X[tr], y[tr]); pred = m.predict(X[te])
        mae_lo, mae_hi, mae_mean = bootstrap_single_ci(y[te], pred, mean_absolute_error)
        r2_lo, r2_hi, r2_mean = bootstrap_single_ci(y[te], pred, r2_score)
        model_results[name] = {
            "mae_point": float(mean_absolute_error(y[te], pred)),
            "mae_bootstrap_95ci": [mae_lo, mae_hi], "mae_bootstrap_mean": mae_mean,
            "r2_point": float(r2_score(y[te], pred)),
            "r2_bootstrap_95ci": [r2_lo, r2_hi], "r2_bootstrap_mean": r2_mean,
        }
        log(f"  {name}: baseline(mean)={naive_mean_mae:.3f} baseline(median)={naive_median_mae:.3f} "
            f"model MAE 95% CI=[{mae_lo:.3f},{mae_hi:.3f}]")

    # Bootstrap CI for the *difference* random_split vs mono_to_multi, and vs multi_to_mono
    tr_r, te_r = protocols["random_split"]
    mr = xgb_default(); mr.fit(X[tr_r], y[tr_r]); pred_r = mr.predict(X[te_r])
    tr_m2m, te_m2m = protocols["mono_to_multi"]
    m1 = xgb_default(); m1.fit(X[tr_m2m], y[tr_m2m]); pred_m2m = m1.predict(X[te_m2m])
    tr_mm2, te_mm2 = protocols["multi_to_mono"]
    m2 = xgb_default(); m2.fit(X[tr_mm2], y[tr_mm2]); pred_mm2 = m2.predict(X[te_mm2])

    diff_m2m = bootstrap_diff_ci(y[te_r], pred_r, y[te_m2m], pred_m2m, mean_absolute_error)
    diff_mm2 = bootstrap_diff_ci(y[te_r], pred_r, y[te_mm2], pred_mm2, mean_absolute_error)
    log(f"  MAE(mono->multi) - MAE(random): 95% CI [{diff_m2m[0]:.3f},{diff_m2m[1]:.3f}], mean {diff_m2m[2]:.3f}")
    log(f"  MAE(multi->mono) - MAE(random): 95% CI [{diff_mm2[0]:.3f},{diff_mm2[1]:.3f}], mean {diff_mm2[2]:.3f}")

    # target distribution: mean/median/IQR per valence group
    mono_y = y[mono_mask]; multi_y = y[multi_mask]
    dist = {
        "monovalent": {"mean": float(mono_y.mean()), "median": float(np.median(mono_y)),
                        "std": float(mono_y.std()), "iqr": [float(np.percentile(mono_y, 25)),
                                                              float(np.percentile(mono_y, 75))]},
        "multivalent": {"mean": float(multi_y.mean()), "median": float(np.median(multi_y)),
                         "std": float(multi_y.std()), "iqr": [float(np.percentile(multi_y, 25)),
                                                                float(np.percentile(multi_y, 75))]},
    }

    results = {"baselines_per_protocol": baselines, "model_bootstrap_per_protocol": model_results,
               "mae_diff_mono_to_multi_minus_random_95ci": diff_m2m,
               "mae_diff_multi_to_mono_minus_random_95ci": diff_mm2,
               "target_distribution_by_valence": dist}
    with open(OUT / "baseline_confidence_intervals.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# CAVD recurring structure identifiers -- what are they?
# ----------------------------------------------------------------------

def cavd_duplicate_structure_check():
    log("=== CAVD recurring structure-identifier investigation ===")
    import openpyxl
    xl = pd.read_excel(DATA / "Descriptors.xlsx", sheet_name=None)
    frames = []
    for cation, sheet in xl.items():
        sheet = sheet.copy()
        sheet["cation"] = cation
        frames.append(sheet)
    raw = pd.concat(frames, ignore_index=True)
    dup_mask = raw.duplicated(subset=["Filename"], keep=False)
    n_dup_rows = int(dup_mask.sum())
    n_dup_ids = int(raw.loc[dup_mask, "Filename"].nunique())

    # for duplicated Filename, check: same cation every time? same IND every time?
    grp = raw[dup_mask].groupby("Filename")
    same_cation = int((grp["cation"].nunique() == 1).sum())
    diff_cation = int((grp["cation"].nunique() > 1).sum())
    same_ind = int((grp["IND"].nunique() == 1).sum()) if "IND" in raw.columns else None
    diff_ind = int((grp["IND"].nunique() > 1).sum()) if "IND" in raw.columns else None

    # fully identical rows (every column) vs rows that share only the Filename
    dup_cols = [c for c in raw.columns if c != "Filename"]
    def row_key(row):
        return tuple(str(row[c]) for c in dup_cols)
    raw["_rowkey"] = raw.apply(row_key, axis=1)
    fully_identical_groups = int((raw[dup_mask].groupby("Filename")["_rowkey"].nunique() == 1).sum())

    results = {
        "n_total_rows": len(raw), "n_duplicated_rows": n_dup_rows, "n_distinct_duplicated_ids": n_dup_ids,
        "ids_with_single_cation_across_duplicates": same_cation,
        "ids_with_multiple_cations_across_duplicates": diff_cation,
        "ids_with_single_IND_value": same_ind, "ids_with_multiple_IND_values": diff_ind,
        "ids_where_all_duplicate_rows_fully_identical": fully_identical_groups,
        "ids_where_duplicate_rows_differ_in_some_column": n_dup_ids - fully_identical_groups,
    }
    log(f"  {results}")
    with open(OUT / "cavd_duplicate_check.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# bootstrap CI for UQ coverage + quantile-crossing check
# ----------------------------------------------------------------------

def uq_bootstrap_check():
    log("=== bootstrap CI for coverage + quantile-crossing check ===")
    from xgboost import XGBRegressor as XGBR

    # Reproduces the earlier leakage-free regularization sweep's feature set and protocol
    # masks exactly (on the geometry-augmented table) so the point estimates match Table S9
    # precisely; this adds a bootstrap CI and a quantile-crossing count on top of it.
    geom_df = pd.read_parquet(MODELS / "additional_robustness" / "migration_table_with_geom.parquet")
    magpie_df, magpie_cols = build_magpie_features(geom_df)
    class_dummies = pd.get_dummies(geom_df["crystal_class"], prefix="class")
    X = np.hstack([magpie_df.to_numpy(float), class_dummies.to_numpy(float),
                    geom_df[["mobile_radius", "mobile_valence"]].to_numpy(float)])
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    y = geom_df["target_Em"].to_numpy(float)

    mono_mask = ~geom_df["is_multivalent"].to_numpy()
    multi_mask = geom_df["is_multivalent"].to_numpy()
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(geom_df)); n_test = int(0.2 * len(geom_df))
    te_idx = idx[:n_test]
    rand_train = np.ones(len(geom_df), bool); rand_train[te_idx] = False
    rand_test = ~rand_train

    # Leakage-corrected settings selected during the earlier regularization sweep / Table S9 (reused here, not re-selected)
    settings = {
        "random_split": (rand_train, rand_test, 15, 2, 40),
        "mono_to_multi": (mono_mask, multi_mask, 30, 2, 20),
        "multi_to_mono": (multi_mask, mono_mask, 15, 2, 40),
    }

    def qmodel(alpha, n_est, depth, lam):
        return XGBR(objective="reg:quantileerror", quantile_alpha=alpha,
                     n_estimators=n_est, max_depth=depth, learning_rate=0.1,
                     reg_lambda=lam, n_jobs=8, random_state=42)

    results = {}
    for name, (tr, te, n_est, depth, lam) in settings.items():
        m10 = qmodel(0.1, n_est, depth, lam); m10.fit(X[tr], y[tr])
        m90 = qmodel(0.9, n_est, depth, lam); m90.fit(X[tr], y[tr])
        p10, p90 = m10.predict(X[te]), m90.predict(X[te])

        n_crossed = int((p10 > p90).sum())
        p10c = np.minimum(p10, p90)
        p90c = np.maximum(p10, p90)
        covered = (y[te] >= p10c) & (y[te] <= p90c)
        coverage_point = float(covered.mean())

        rng2 = np.random.RandomState(0)
        boot = []
        n = len(covered)
        for _ in range(3000):
            bi = rng2.randint(0, n, n)
            boot.append(covered[bi].mean())
        boot = np.array(boot)
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

        results[name] = {
            "n_test": n, "coverage_point": coverage_point, "coverage_95ci": ci,
            "n_quantile_crossings": n_crossed, "frac_crossed": n_crossed / n,
        }
        log(f"  {name}: coverage={coverage_point:.3f} 95% CI={ci}, crossings={n_crossed}/{n}")

    with open(OUT / "uq_bootstrap.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def main():
    t0 = time.time()
    r1 = corrected_mobile_cation_labels()
    r2 = radius_ablation_check()
    r3 = baseline_confidence_intervals()
    r4 = cavd_duplicate_structure_check()
    r7 = uq_bootstrap_check()
    log(f"ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
