"""
Additional cross-valence robustness checks for the migration-barrier and
CAVD percolation datasets: NEB-corrected-label reruns of the model
benchmark, descriptor ablation, SHAP and Gaussian-process comparisons; a
GGA-only functional-matched subset test; composition-grouped training
resampling and within-valence-group splits; a CAVD matched-host subset
check; an electronegativity-difference bonding descriptor; and screening
retention rates under cross-valence extrapolation.
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import (mean_absolute_error, r2_score,
                              balanced_accuracy_score, precision_score,
                              recall_score)
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import ks_2samp
from xgboost import XGBRegressor, XGBClassifier
import shap

warnings.filterwarnings("ignore")

PROJ = Path("/glade/work/ajalali/12 - Multivalent-Ion-Conductor-ML-project")  # adjust to local checkout root
DATA = PROJ / "data"
MODELS = PROJ / "models"
OUT = PROJ / "models" / "revision5"
OUT.mkdir(parents=True, exist_ok=True)

VALENCE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Mg": 2, "Ca": 2, "Zn": 2, "Al": 3, "Sr": 2}
MONO_IONS = {"Li", "Na", "K", "Rb"}
MULTI_IONS = {"Mg", "Ca", "Zn", "Al", "Sr"}


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


def load_corrected_df():
    """Migration-barrier table with NEB-corrected mobile-cation labels (328 mono / 289 multi)."""
    with open(MODELS / "revision2" / "v2_comment3_neb_check.json") as f:
        neb = json.load(f)
    full_table = pd.DataFrame(neb["full_table"])
    df = pd.read_parquet(DATA / "migration_barrier_table.parquet")
    df = df.dropna(subset=["mobile_cation"]).reset_index(drop=True)
    neb_winner_by_jid = dict(zip(full_table["jid"], full_table["neb_displacement_winner"]))
    df["mobile_cation_corrected"] = df.apply(
        lambda r: neb_winner_by_jid.get(r["jid"], r["mobile_cation"]), axis=1)
    df["valence_corrected"] = df["mobile_cation_corrected"].map(VALENCE)
    df["is_multivalent_corrected"] = df["valence_corrected"] >= 2
    reduced = [Composition(f.replace(" ", "")).reduced_formula for f in df["formula"]]
    df["reduced_formula"] = reduced
    return df


def migration_features(df, extra_cols=None):
    magpie_df, magpie_cols = build_magpie_features(df)
    class_dummies = pd.get_dummies(df["crystal_class"], prefix="class")
    blocks = [magpie_df.to_numpy(float), class_dummies.to_numpy(float)]
    names = list(magpie_cols) + list(class_dummies.columns)
    if extra_cols:
        blocks.append(df[extra_cols].to_numpy(float))
        names += list(extra_cols)
    X = np.hstack(blocks)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    return X, names


def protocols_corrected(df):
    mono = ~df["is_multivalent_corrected"].to_numpy()
    multi = df["is_multivalent_corrected"].to_numpy()
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df)); n_test = int(0.2 * len(df))
    te_idx = idx[:n_test]
    rand_train = np.ones(len(df), bool); rand_train[te_idx] = False
    rand_test = ~rand_train
    return {"random_split": (rand_train, rand_test), "mono_to_multi": (mono, multi),
            "multi_to_mono": (multi, mono)}


# ----------------------------------------------------------------------
# Regenerate the model benchmark, descriptor ablation, SHAP/leave-one-cation-out,
# GP comparison, and confound-isolation checks with the NEB-corrected mobile-
# cation labels (328 monovalent / 289 multivalent), for internal consistency
# with the paper's primary headline result.
# ----------------------------------------------------------------------

def rebuild_benchmark_corrected_labels():
    log("=== Migration-barrier + CAVD benchmark, NEB-corrected labels ===")
    df = load_corrected_df()
    X, _ = migration_features(df)
    y = df["target_Em"].to_numpy(float)
    protos = protocols_corrected(df)

    def run_reg(model_fn, scale=False):
        out = {}
        for name, (tr, te) in protos.items():
            Xtr, Xte = X[tr], X[te]
            if scale:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            m = model_fn(); m.fit(Xtr, y[tr]); pred = m.predict(Xte)
            out[name] = float(mean_absolute_error(y[te], pred))
        return out

    from sklearn.ensemble import RandomForestRegressor
    migration_benchmark = {
        "XGBoost": run_reg(xgb_default),
        "RandomForest": run_reg(lambda: RandomForestRegressor(n_estimators=400, min_samples_leaf=2, n_jobs=8, random_state=42)),
        "SVR": run_reg(lambda: SVR(C=10, epsilon=0.05), scale=True),
        "kNN": run_reg(lambda: KNeighborsRegressor(n_neighbors=8, weights="distance"), scale=True),
        "GaussianProcess": run_reg(lambda: GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 10.0)),
            normalize_y=True, n_restarts_optimizer=3, random_state=42), scale=True),
        "Ridge": run_reg(lambda: Ridge(alpha=5.0), scale=True),
        "MLP": run_reg(lambda: MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True,
                                             max_iter=2000, random_state=42), scale=True),
    }
    for k, v in migration_benchmark.items():
        log(f"  migration {k}: {v}")

    cavd = pd.read_parquet(DATA / "feature_table.parquet")
    drop_cols = ["cation", "filename", "IND", "percolates"]
    feat_cols = [c for c in cavd.columns if c not in drop_cols]
    Xc = cavd[feat_cols].to_numpy(float)
    Xc = np.where(np.isnan(Xc), np.nanmean(Xc, axis=0), Xc)
    yc = cavd["percolates"].to_numpy()
    mono_c = cavd["cation"].isin(["Li", "Na"]).to_numpy()
    multi_c = cavd["cation"].isin(["Mg", "Al"]).to_numpy()
    rng = np.random.RandomState(42)
    idxc = rng.permutation(len(cavd)); n_test_c = int(0.2 * len(cavd))
    te_c = idxc[:n_test_c]; tr_c = np.ones(len(cavd), bool); tr_c[te_c] = False; te_c_mask = ~tr_c
    cavd_protos = {"random_split": (tr_c, te_c_mask), "mono_to_multi": (mono_c, multi_c),
                   "multi_to_mono": (multi_c, mono_c)}

    def run_clf(model_fn, scale=False):
        out = {}
        for name, (tr, te) in cavd_protos.items():
            Xtr, Xte = Xc[tr], Xc[te]
            if scale:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            m = model_fn(); m.fit(Xtr, yc[tr]); pred = m.predict(Xte)
            out[name] = float(balanced_accuracy_score(yc[te], pred))
        return out

    cavd_benchmark = {
        "RandomForest": run_clf(lambda: RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=2,
                                                                 class_weight="balanced_subsample", n_jobs=8, random_state=42)),
        "SVM": run_clf(lambda: SVC(C=5, class_weight="balanced", random_state=42), scale=True),
        "kNN": run_clf(lambda: KNeighborsClassifier(n_neighbors=15, weights="distance"), scale=True),
        "MLP": run_clf(lambda: MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                              max_iter=1000, random_state=42), scale=True),
        "LogisticRegression": run_clf(lambda: LogisticRegression(class_weight="balanced", max_iter=2000), scale=True),
    }
    for k, v in cavd_benchmark.items():
        log(f"  cavd {k}: {v}")

    results = {"migration_benchmark_mae_corrected_labels": migration_benchmark,
               "cavd_benchmark_balanced_accuracy": cavd_benchmark,
               "note": "CAVD benchmark unaffected by the migration-barrier mobile-cation correction; "
                       "included here so Table 3 is regenerated as one consistent table."}
    with open(OUT / "rebuild_benchmark_corrected_labels.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def rebuild_descriptor_ablation_corrected_labels():
    log("=== Descriptor ablation, NEB-corrected labels ===")
    geom_df = pd.read_parquet(MODELS / "revision1" / "migration_table_with_geom.parquet")
    with open(MODELS / "revision2" / "v2_comment3_neb_check.json") as f:
        neb = json.load(f)
    full_table = pd.DataFrame(neb["full_table"])
    neb_winner_by_jid = dict(zip(full_table["jid"], full_table["neb_displacement_winner"]))
    geom_df["mobile_cation_corrected"] = geom_df["jid"].map(neb_winner_by_jid).fillna(geom_df["mobile_cation"])
    geom_df["valence_corrected"] = geom_df["mobile_cation_corrected"].map(VALENCE)
    geom_df["is_multivalent_corrected"] = geom_df["valence_corrected"] >= 2
    protos = protocols_corrected(geom_df)
    y = geom_df["target_Em"].to_numpy(float)

    Xb, _ = migration_features(geom_df)
    Xr, _ = migration_features(geom_df, extra_cols=["mobile_radius", "mobile_valence"])
    geom_cols = [c for c in ["mobile_coord_number", "mobile_nn_distance", "migration_path_length"] if c in geom_df.columns]
    Xg, _ = migration_features(geom_df, extra_cols=["mobile_radius", "mobile_valence"] + geom_cols)

    def eval_set(X):
        out = {}
        for name, (tr, te) in protos.items():
            m = xgb_default(); m.fit(X[tr], y[tr]); pred = m.predict(X[te])
            out[name] = {"mae": float(mean_absolute_error(y[te], pred)), "r2": float(r2_score(y[te], pred))}
        return out

    results = {"baseline": eval_set(Xb), "plus_radius_charge": eval_set(Xr), "plus_geometric": eval_set(Xg)}
    log(f"  {results}")
    with open(OUT / "rebuild_descriptor_ablation_corrected_labels.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def rebuild_shap_and_cationout_corrected_labels():
    log("=== SHAP + leave-one-cation-out, NEB-corrected labels ===")
    df = load_corrected_df()
    X, names = migration_features(df)
    y = df["target_Em"].to_numpy(float)
    m = xgb_default(); m.fit(X, y)
    explainer = shap.TreeExplainer(m)
    sv = explainer.shap_values(X)
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)
    top15 = [(names[i], float(mean_abs[i])) for i in order[:15]]

    cation_col = df["mobile_cation_corrected"]
    counts = cation_col.value_counts()
    cationout = {}
    for cat in counts[counts >= 15].index:
        te = (cation_col == cat).to_numpy()
        tr = ~te
        mm = xgb_default(); mm.fit(X[tr], y[tr]); pred = mm.predict(X[te])
        cationout[cat] = {"n": int(te.sum()), "mae": float(mean_absolute_error(y[te], pred)),
                           "r2": float(r2_score(y[te], pred))}
    log(f"  top15 SHAP: {top15[:5]} ...")
    log(f"  leave-one-cation-out: {cationout}")
    results = {"top15_shap_corrected_labels": top15, "leave_one_cation_out_corrected_labels": cationout}
    with open(OUT / "shap_cationout_corrected_labels.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def rebuild_gp_comparison_corrected_labels():
    log("=== Gaussian process vs XGBoost, NEB-corrected labels ===")
    df = load_corrected_df()
    X, _ = migration_features(df)
    y = df["target_Em"].to_numpy(float)
    protos = protocols_corrected(df)
    results = {}
    for name, (tr, te) in protos.items():
        sc = StandardScaler().fit(X[tr])
        Xtr_s, Xte_s = sc.transform(X[tr]), sc.transform(X[te])
        mx = xgb_default(); mx.fit(X[tr], y[tr]); predx = mx.predict(X[te])
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 10.0))
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=42)
        gp.fit(Xtr_s, y[tr]); predg, stdg = gp.predict(Xte_s, return_std=True)
        results[name] = {"xgb_mae": float(mean_absolute_error(y[te], predx)),
                          "gp_mae": float(mean_absolute_error(y[te], predg)),
                          "gp_mean_std": float(stdg.mean())}
        log(f"  {name}: {results[name]}")
    with open(OUT / "rebuild_gp_comparison_corrected_labels.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def rebuild_confound_checks_corrected_labels():
    log("=== Confound-isolation checks, NEB-corrected labels ===")
    df = load_corrected_df()
    y = df["target_Em"].to_numpy(float)
    mono_mask = ~df["is_multivalent_corrected"].to_numpy()
    multi_mask = df["is_multivalent_corrected"].to_numpy()

    # radius/charge
    Xb, _ = migration_features(df)
    geom_df = pd.read_parquet(MODELS / "revision1" / "migration_table_with_geom.parquet")
    geom_aligned = geom_df.set_index("jid").reindex(df["jid"]).reset_index()
    if {"mobile_radius", "mobile_valence"}.issubset(geom_aligned.columns) and not geom_aligned["mobile_radius"].isna().all():
        Xr, _ = migration_features(df.assign(mobile_radius=geom_aligned["mobile_radius"].to_numpy(),
                                              mobile_valence=geom_aligned["mobile_valence"].to_numpy()),
                                    extra_cols=["mobile_radius", "mobile_valence"])
    else:
        Xr = Xb

    def eval_mae(X, tr, te):
        m = xgb_default(); m.fit(X[tr], y[tr]); return float(mean_absolute_error(y[te], m.predict(X[te])))

    radius_check = {
        "mono_to_multi_baseline": eval_mae(Xb, mono_mask, multi_mask),
        "mono_to_multi_with_radius_charge": eval_mae(Xr, mono_mask, multi_mask),
        "multi_to_mono_baseline": eval_mae(Xb, multi_mask, mono_mask),
        "multi_to_mono_with_radius_charge": eval_mae(Xr, multi_mask, mono_mask),
    }

    # structural-class check (>=3 rows each valence, corrected labels)
    class_counts = df.groupby("crystal_class").apply(
        lambda g: (g["is_multivalent_corrected"] == False).sum() >= 3 and (g["is_multivalent_corrected"] == True).sum() >= 3)
    shared_classes = class_counts[class_counts].index.tolist()
    class_results = {}
    for cls in shared_classes:
        held = df["crystal_class"] == cls
        tr = ~held
        m = xgb_default(); m.fit(Xb[tr], y[tr])
        held_mono = held & mono_mask
        held_multi = held & multi_mask
        mae_mono = float(mean_absolute_error(y[held_mono], m.predict(Xb[held_mono]))) if held_mono.sum() else None
        mae_multi = float(mean_absolute_error(y[held_multi], m.predict(Xb[held_multi]))) if held_multi.sum() else None
        class_results[cls] = {"n_mono": int(held_mono.sum()), "n_multi": int(held_multi.sum()),
                              "mae_mono": mae_mono, "mae_multi": mae_multi}

    # training-size matched (subsample mono to multi's size)
    n_multi = int(multi_mask.sum())
    subsample_maes = []
    rng = np.random.RandomState(0)
    mono_idx = np.where(mono_mask)[0]
    for i in range(10):
        sub = rng.choice(mono_idx, size=n_multi, replace=False)
        tr = np.zeros(len(df), bool); tr[sub] = True
        subsample_maes.append(eval_mae(Xb, tr, multi_mask))
    size_matched = {"mono_to_multi_subsampled_mae_mean_std": [float(np.mean(subsample_maes)), float(np.std(subsample_maes))],
                     "mono_to_multi_full_mae": radius_check["mono_to_multi_baseline"]}

    # nearest-neighbor distance in standardized feature space
    sc = StandardScaler().fit(Xb)
    Xs = sc.transform(Xb)
    rng2 = np.random.RandomState(42)
    idx2 = rng2.permutation(len(df)); n_test2 = int(0.2 * len(df))
    te2 = idx2[:n_test2]; tr2 = np.ones(len(df), bool); tr2[te2] = False; te2m = ~tr2

    def nn_dist(train_mask, test_mask):
        from scipy.spatial.distance import cdist
        d = cdist(Xs[test_mask], Xs[train_mask])
        return float(d.min(axis=1).mean())

    nn_distances = {"random_split": nn_dist(tr2, te2m), "mono_to_multi": nn_dist(mono_mask, multi_mask),
                     "multi_to_mono": nn_dist(multi_mask, mono_mask)}

    # covariate vs target shift + KS test
    protos = {"mono_to_multi": (mono_mask, multi_mask), "multi_to_mono": (multi_mask, mono_mask)}
    covariate_target = {}
    for name, (tr, te) in protos.items():
        naive_mae = float(mean_absolute_error(y[te], np.full(te.sum(), y[tr].mean())))
        model_mae = eval_mae(Xb, tr, te)
        covariate_target[name] = {"naive_group_mean_mae": naive_mae, "model_mae": model_mae,
                                   "model_beats_naive": model_mae < naive_mae}
    ks_stat, ks_p = ks_2samp(y[mono_mask], y[multi_mask])

    results = {"radius_composition_check": radius_check, "structural_class_check": class_results,
               "training_size_matched": size_matched, "nearest_neighbor_distances": nn_distances,
               "covariate_vs_target_shift": covariate_target, "ks_test": {"statistic": float(ks_stat), "p": float(ks_p)}}
    log(f"  radius check: {radius_check}")
    log(f"  covariate/target shift: {covariate_target}")
    with open(OUT / "rebuild_confound_checks_corrected_labels.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# GGA-only subset test (273 monovalent / 270 multivalent, NEB-corrected
# labels), to check whether the cross-valence asymmetry survives once the
# exchange-correlation functional is held fixed across both groups.
# ----------------------------------------------------------------------

def gga_only_subset_check():
    log("=== GGA-only, functional-matched cross-valence test ===")
    df = load_corrected_df()
    gga = df[df["XC"] == "GGA"].reset_index(drop=True)
    n_mono = int((~gga["is_multivalent_corrected"]).sum())
    n_multi = int(gga["is_multivalent_corrected"].sum())
    X, _ = migration_features(gga)
    y = gga["target_Em"].to_numpy(float)
    mono_mask = ~gga["is_multivalent_corrected"].to_numpy()
    multi_mask = gga["is_multivalent_corrected"].to_numpy()

    def eval_mae_r2(tr, te):
        m = xgb_default(); m.fit(X[tr], y[tr]); pred = m.predict(X[te])
        return {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                "mae": float(mean_absolute_error(y[te], pred)), "r2": float(r2_score(y[te], pred))}

    results = {"n_mono_gga": n_mono, "n_multi_gga": n_multi,
               "mono_to_multi": eval_mae_r2(mono_mask, multi_mask),
               "multi_to_mono": eval_mae_r2(multi_mask, mono_mask),
               "full_dataset_comparison": {"mono_to_multi_mae": 0.536, "multi_to_mono_mae": 1.035}}
    log(f"  {results}")
    with open(OUT / "gga_only_subset_check.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# Composition-grouped resampling of the training split itself (not just a
# test-set bootstrap), for all three protocols.
# ----------------------------------------------------------------------

def composition_grouped_training_resampling():
    log("=== Composition-grouped resampling of training splits ===")
    df = load_corrected_df()
    X, _ = migration_features(df)
    y = df["target_Em"].to_numpy(float)
    groups = df["reduced_formula"].to_numpy()
    mono_mask = ~df["is_multivalent_corrected"].to_numpy()
    multi_mask = df["is_multivalent_corrected"].to_numpy()

    # random split: resample distinct compositions (grouped) with replacement, refit, re-evaluate
    rng = np.random.RandomState(0)
    uniq_formulas = df["reduced_formula"].unique()
    random_split_maes = []
    for i in range(200):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=i)
        tr, te = next(gss.split(X, y, groups))
        m = xgb_default(); m.fit(X[tr], y[tr]); pred = m.predict(X[te])
        random_split_maes.append(mean_absolute_error(y[te], pred))
    random_split_maes = np.array(random_split_maes)

    # mono_to_multi / multi_to_mono: resample compositions WITHIN each valence group (train side
    # resampled with replacement by composition; test side held fixed as the full opposite group),
    # to put a CI on training-set variability specifically.
    def grouped_bootstrap(train_mask, test_mask, n_boot=200):
        train_formulas = df.loc[train_mask, "reduced_formula"].unique()
        maes = []
        for i in range(n_boot):
            rs = np.random.RandomState(i)
            sampled_formulas = rs.choice(train_formulas, size=len(train_formulas), replace=True)
            sel = df["reduced_formula"].isin(set(sampled_formulas)) & train_mask
            if sel.sum() < 10:
                continue
            m = xgb_default(); m.fit(X[sel], y[sel])
            pred = m.predict(X[test_mask])
            maes.append(mean_absolute_error(y[test_mask], pred))
        return np.array(maes)

    m2m = grouped_bootstrap(mono_mask, multi_mask)
    mm2 = grouped_bootstrap(multi_mask, mono_mask)

    def summarize(a):
        return {"mean": float(a.mean()), "std": float(a.std()),
                "95ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))], "n_boot": int(len(a))}

    results = {"random_split_composition_grouped": summarize(random_split_maes),
               "mono_to_multi_composition_grouped_train_resample": summarize(m2m),
               "multi_to_mono_composition_grouped_train_resample": summarize(mm2)}
    log(f"  {results}")
    with open(OUT / "composition_grouped_training_resampling.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# Random splits within each valence group separately (not pooled), 10
# seeds each.
# ----------------------------------------------------------------------

def within_valence_group_splits():
    log("=== Within-valence-group random splits ===")
    df = load_corrected_df()
    X, _ = migration_features(df)
    y = df["target_Em"].to_numpy(float)
    groups = df["reduced_formula"].to_numpy()

    def within_group(mask):
        sub_idx = np.where(mask)[0]
        Xg, yg, gg = X[sub_idx], y[sub_idx], groups[sub_idx]
        r2s, maes = [], []
        for seed in range(10):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            tr, te = next(gss.split(Xg, yg, gg))
            m = xgb_default(); m.fit(Xg[tr], yg[tr]); pred = m.predict(Xg[te])
            r2s.append(r2_score(yg[te], pred)); maes.append(mean_absolute_error(yg[te], pred))
        return {"n": int(mask.sum()), "r2_range": [float(min(r2s)), float(max(r2s))],
                "r2_mean_std": [float(np.mean(r2s)), float(np.std(r2s))],
                "mae_mean_std": [float(np.mean(maes)), float(np.std(maes))]}

    results = {"monovalent_only": within_group(~df["is_multivalent_corrected"].to_numpy()),
               "multivalent_only": within_group(df["is_multivalent_corrected"].to_numpy())}
    log(f"  {results}")
    with open(OUT / "within_valence_group_splits.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# Of the 656 CAVD structures analyzed for >=2 candidate ions, how many
# contain both a monovalent and a multivalent candidate, and does the
# cross-valence asymmetry persist on exactly those matched hosts?
# ----------------------------------------------------------------------

def matched_host_structures_check():
    log("=== CAVD matched-host (both valence groups) subset ===")
    xl = pd.read_excel(DATA / "Descriptors.xlsx", sheet_name=None)
    frames = []
    for cation, sheet in xl.items():
        sheet = sheet.copy(); sheet["cation"] = cation
        frames.append(sheet)
    raw = pd.concat(frames, ignore_index=True)
    dup_mask = raw.duplicated(subset=["Filename"], keep=False)
    recurring = raw[dup_mask]
    grp = recurring.groupby("Filename")["cation"].apply(set)
    both_valence = grp[grp.apply(lambda s: bool(s & MONO_IONS) and bool(s & MULTI_IONS))]
    n_both = len(both_valence)
    log(f"  {n_both} of {len(grp)} recurring structures have both a mono and a multi candidate ion")

    matched_filenames = set(both_valence.index)
    cavd = pd.read_parquet(DATA / "feature_table.parquet")
    matched_mask = cavd["filename"].isin(matched_filenames)
    matched_mono = matched_mask & cavd["cation"].isin(["Li", "Na"])
    matched_multi = matched_mask & cavd["cation"].isin(["Mg", "Al"])

    drop_cols = ["cation", "filename", "IND", "percolates"]
    feat_cols = [c for c in cavd.columns if c not in drop_cols]
    Xc = cavd[feat_cols].to_numpy(float)
    Xc = np.where(np.isnan(Xc), np.nanmean(Xc, axis=0), Xc)
    yc = cavd["percolates"].to_numpy()

    full_mono = cavd["cation"].isin(["Li", "Na"]).to_numpy()
    full_multi = cavd["cation"].isin(["Mg", "Al"]).to_numpy()

    def run(train_mask, test_mask):
        m = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=2,
                                    class_weight="balanced_subsample", n_jobs=8, random_state=42)
        m.fit(Xc[train_mask], yc[train_mask])
        pred = m.predict(Xc[test_mask])
        return {"n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
                "balanced_accuracy": float(balanced_accuracy_score(yc[test_mask], pred))}

    results = {
        "n_recurring_structures": len(grp),
        "n_with_both_valence_groups": n_both,
        "matched_subset_n_mono_rows": int(matched_mono.sum()),
        "matched_subset_n_multi_rows": int(matched_multi.sum()),
        "full_pool_train_matched_host_test_mono_to_multi": run(full_mono & ~matched_mask.to_numpy(), matched_multi.to_numpy()),
        "full_pool_train_matched_host_test_multi_to_mono": run(full_multi & ~matched_mask.to_numpy(), matched_mono.to_numpy()),
        "full_pool_reference": {"mono_to_multi_balanced_accuracy": 0.526, "multi_to_mono_balanced_accuracy": 0.622},
    }
    log(f"  {results}")
    with open(OUT / "matched_host_structures_check.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# A bonding/covalency descriptor (electronegativity difference between
# mobile ion and host), not just composition statistics -- does it help
# explain the transfer failure?
# ----------------------------------------------------------------------

def bonding_descriptor_check():
    log("=== Electronegativity-difference (bonding/covalency) descriptor ===")
    df = load_corrected_df()

    def en_diff(formula, mobile_cation):
        comp = Composition(formula.replace(" ", ""))
        mobile_en = Element(mobile_cation).X
        other_en = [Element(el.symbol).X * amt for el, amt in comp.items() if el.symbol != mobile_cation]
        other_amt = [amt for el, amt in comp.items() if el.symbol != mobile_cation]
        if not other_amt:
            return np.nan
        mean_other_en = sum(other_en) / sum(other_amt)
        return abs(mobile_en - mean_other_en)

    df["en_diff"] = [en_diff(f, c) for f, c in zip(df["formula"], df["mobile_cation_corrected"])]
    mono_en = df.loc[~df["is_multivalent_corrected"], "en_diff"].dropna()
    multi_en = df.loc[df["is_multivalent_corrected"], "en_diff"].dropna()
    pooled_std = np.sqrt((mono_en.std() ** 2 + multi_en.std() ** 2) / 2)
    smd_en_diff = float((multi_en.mean() - mono_en.mean()) / pooled_std)

    Xb, names_b = migration_features(df)
    Xen = np.hstack([Xb, df[["en_diff"]].fillna(df["en_diff"].mean()).to_numpy(float)])
    y = df["target_Em"].to_numpy(float)
    protos = protocols_corrected(df)

    def eval_set(X):
        out = {}
        for name, (tr, te) in protos.items():
            m = xgb_default(); m.fit(X[tr], y[tr]); pred = m.predict(X[te])
            out[name] = {"mae": float(mean_absolute_error(y[te], pred)), "r2": float(r2_score(y[te], pred))}
        return out

    baseline_results = eval_set(Xb)
    en_results = eval_set(Xen)

    m = xgb_default(); m.fit(Xen, y)
    explainer = shap.TreeExplainer(m)
    sv = explainer.shap_values(Xen)
    mean_abs = np.abs(sv).mean(axis=0)
    all_names = names_b + ["en_diff_cation_host"]
    rank = int(np.argsort(-mean_abs).tolist().index(len(all_names) - 1)) + 1

    results = {"electronegativity_diff_standardized_mean_difference_mono_vs_multi": smd_en_diff,
               "mean_electronegativity_comparison_smd_reference": 0.21,
               "baseline_results": baseline_results, "with_en_diff_feature_results": en_results,
               "en_diff_shap_rank_of_170_plus_1": rank, "n_total_features": len(all_names)}
    log(f"  SMD(en_diff)={smd_en_diff:.3f}; SHAP rank of en_diff: {rank}/{len(all_names)}")
    log(f"  baseline vs +en_diff: {baseline_results} | {en_results}")
    with open(OUT / "bonding_descriptor_check.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# Fraction of genuinely low-barrier candidates retained at a cutoff,
# specifically under cross-valence extrapolation (not just the random
# split already reported).
# ----------------------------------------------------------------------

def screening_retention_under_extrapolation():
    log("=== Low-barrier screening retention under cross-valence extrapolation ===")
    df = load_corrected_df()
    X, _ = migration_features(df)
    y = df["target_Em"].to_numpy(float)
    protos = protocols_corrected(df)
    results = {}
    for cutoff in (0.3, 0.5):
        results[f"cutoff_{cutoff}"] = {}
        for name, (tr, te) in protos.items():
            m = xgb_default(); m.fit(X[tr], y[tr]); pred = m.predict(X[te])
            y_true_low = (y[te] < cutoff).astype(int)
            y_pred_low = (pred < cutoff).astype(int)
            n_true_low = int(y_true_low.sum())
            n_retained = int(((y_true_low == 1) & (y_pred_low == 1)).sum())
            recall = float(recall_score(y_true_low, y_pred_low)) if n_true_low > 0 else None
            precision = float(precision_score(y_true_low, y_pred_low, zero_division=0))
            results[f"cutoff_{cutoff}"][name] = {
                "n_true_low_barrier_candidates": n_true_low,
                "n_retained_by_model": n_retained,
                "fraction_retained_recall": recall, "precision": precision,
            }
    log(f"  {results}")
    with open(OUT / "screening_retention_under_extrapolation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def main():
    t0 = time.time()
    rebuild_benchmark_corrected_labels()
    rebuild_descriptor_ablation_corrected_labels()
    rebuild_shap_and_cationout_corrected_labels()
    rebuild_gp_comparison_corrected_labels()
    rebuild_confound_checks_corrected_labels()
    gga_only_subset_check()
    composition_grouped_training_resampling()
    within_valence_group_splits()
    matched_host_structures_check()
    bonding_descriptor_check()
    screening_retention_under_extrapolation()
    log(f"ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
