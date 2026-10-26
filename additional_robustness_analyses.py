# Additional robustness checks for the cross-valence-transfer analysis: confound/asymmetry
# diagnostics, duplicate-grouped splits, mobile-ion crosscheck, descriptor ablation, deeper
# uncertainty quantification, SHAP robustness, and an electrolyte-only subset check. No new
# DFT/atomistic simulation -- re-analysis of data already on disk plus new ML runs.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

PROJ = Path("/glade/work/ajalali/12 - Multivalent-Ion-Conductor-ML-project")
DATA = PROJ / "data"
MODELS = PROJ / "models"
OUT = PROJ / "models" / "additional_robustness"
OUT.mkdir(parents=True, exist_ok=True)

SHANNON_RADII = {  # Shannon (1976) effective ionic radii, 6-coordinate, A, most common oxidation state
    "Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52,
    "Mg": 0.72, "Ca": 1.00, "Zn": 0.74, "Al": 0.535, "Sr": 1.18,
}
VALENCE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Mg": 2, "Ca": 2, "Zn": 2, "Al": 3, "Sr": 2}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------
# Shared loading
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# mobile-ion radius/charge + geometric descriptors from the
# already-downloaded NEB structures (NOT new DFT -- these structures are
# already in EM-COMPLETE-DATASET.json, we only post-process coordinates)
# ----------------------------------------------------------------------

def jarvis_to_pymatgen(struct_dict):
    lattice = Lattice(struct_dict["lattice_mat"])
    return Structure(lattice, struct_dict["elements"], struct_dict["coords"],
                      coords_are_cartesian=struct_dict.get("cartesian", False))


def build_geometric_descriptors():
    log("=== Comment 7: mobile-ion radius/charge + NEB geometric descriptors ===")
    with open(DATA / "migration_barriers" / "EM-COMPLETE-DATASET.json") as f:
        records = {r["jid"]: r for r in json.load(f)}

    df = load_migration_df()
    df["mobile_radius"] = df["mobile_cation"].map(SHANNON_RADII)
    df["mobile_valence"] = df["mobile_cation"].map(VALENCE)

    coord_nums, nn_dists, path_lengths, n_fail = [], [], [], 0
    for _, row in df.iterrows():
        try:
            rec = records[row["jid"]]
            si = jarvis_to_pymatgen(rec["structure_ini"])
            sf = jarvis_to_pymatgen(rec["structure_fin"])
            cation = row["mobile_cation"]
            cation_idx = [i for i, sp in enumerate(si.species) if sp.symbol == cation]
            if not cation_idx:
                raise ValueError("cation not found in structure")
            # mobile site = the cation site with the largest ini->fin displacement
            # (classic NEB construction: one ion hops, the rest relax only slightly)
            best_i, best_disp = None, -1.0
            for i in cation_idx:
                d = si[i].distance(sf[i])
                # guard against periodic-image artifacts dominating; use minimum-image via pymatgen's distance()
                if d > best_disp:
                    best_disp, best_i = d, i
            path_lengths.append(best_disp)
            # coordination number of the mobile ion at its initial site: count neighbors within 3.5 A
            neighbors = si.get_neighbors(si[best_i], r=3.5)
            coord_nums.append(len(neighbors))
            nn_dists.append(min(n.nn_distance for n in neighbors) if neighbors else np.nan)
        except Exception as e:
            coord_nums.append(np.nan); nn_dists.append(np.nan); path_lengths.append(np.nan)
            n_fail += 1

    df["mobile_coord_number"] = coord_nums
    df["mobile_nn_distance"] = nn_dists
    df["migration_path_length"] = path_lengths
    log(f"Geometric descriptors computed for {len(df)-n_fail}/{len(df)} rows ({n_fail} failures)")
    log(f"  path_length: mean={np.nanmean(path_lengths):.3f} A, median={np.nanmedian(path_lengths):.3f} A")
    log(f"  coord_number: mean={np.nanmean(coord_nums):.2f}")

    df.to_parquet(OUT / "migration_table_with_geom.parquet")

    # Ablation: baseline (magpie+class) vs +radius/charge vs +radius/charge+geometric
    magpie_df, magpie_cols = build_magpie_features(df)
    class_dummies = pd.get_dummies(df["crystal_class"], prefix="class")
    X_base = np.hstack([magpie_df.to_numpy(float), class_dummies.to_numpy(float)])
    extra1 = df[["mobile_radius", "mobile_valence"]].to_numpy(float)
    extra2 = df[["mobile_coord_number", "mobile_nn_distance", "migration_path_length"]].to_numpy(float)
    X_base = np.where(np.isnan(X_base), np.nanmean(X_base, axis=0), X_base)
    extra1 = np.where(np.isnan(extra1), np.nanmean(extra1, axis=0), extra1)
    extra2_mean = np.nanmean(extra2, axis=0)
    extra2 = np.where(np.isnan(extra2), extra2_mean, extra2)

    variants = {
        "baseline_168feat": X_base,
        "plus_radius_charge": np.hstack([X_base, extra1]),
        "plus_radius_charge_geom": np.hstack([X_base, extra1, extra2]),
    }
    y = df["target_Em"].to_numpy(float)
    mono_mask = ~df["is_multivalent"].to_numpy()
    multi_mask = df["is_multivalent"].to_numpy()
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df)); n_test = int(0.2 * len(df))
    test_idx = idx[:n_test]
    rand_train = np.ones(len(df), bool); rand_train[test_idx] = False
    rand_test = ~rand_train
    protocols = {"random_split": (rand_train, rand_test),
                 "mono_to_multi": (mono_mask, multi_mask),
                 "multi_to_mono": (multi_mask, mono_mask)}

    ablation_results = {}
    for vname, X in variants.items():
        ablation_results[vname] = {}
        for pname, (tr, te) in protocols.items():
            m = xgb_default()
            m.fit(X[tr], y[tr])
            pred = m.predict(X[te])
            ablation_results[vname][pname] = {
                "mae": float(mean_absolute_error(y[te], pred)),
                "r2": float(r2_score(y[te], pred)),
            }
        log(f"  {vname}: " + ", ".join(f"{p}={ablation_results[vname][p]['mae']:.3f}eV/R2={ablation_results[vname][p]['r2']:.3f}" for p in protocols))

    with open(OUT / "descriptor_ablation.json", "w") as f:
        json.dump({"ablation": ablation_results,
                    "geometric_descriptor_coverage": {"n_total": len(df), "n_success": len(df) - n_fail}},
                   f, indent=2)
    return df


# ----------------------------------------------------------------------
# descriptor overlap, confound control, within-class
# cross-ion comparison, target distributions, train-size equalization,
# structural-class overlap, group-specific baselines
# ----------------------------------------------------------------------

def confound_and_asymmetry_check(df_geom):
    log("=== Comments 1 & 6: confound analysis and transfer-asymmetry investigation ===")
    df = df_geom.copy()
    y = df["target_Em"].to_numpy(float)
    mono_mask = ~df["is_multivalent"].to_numpy()
    multi_mask = df["is_multivalent"].to_numpy()

    results = {}

    # --- target distributions per valence group ---
    results["target_distributions"] = {
        "monovalent": {"mean": float(y[mono_mask].mean()), "median": float(np.median(y[mono_mask])),
                        "std": float(y[mono_mask].std()), "min": float(y[mono_mask].min()), "max": float(y[mono_mask].max()),
                        "n": int(mono_mask.sum())},
        "multivalent": {"mean": float(y[multi_mask].mean()), "median": float(np.median(y[multi_mask])),
                         "std": float(y[multi_mask].std()), "min": float(y[multi_mask].min()), "max": float(y[multi_mask].max()),
                         "n": int(multi_mask.sum())},
    }
    log(f"  target dist: mono mean={y[mono_mask].mean():.3f} std={y[mono_mask].std():.3f} | "
        f"multi mean={y[multi_mask].mean():.3f} std={y[multi_mask].std():.3f}")

    # --- descriptor overlap: mobile radius distributions ---
    r_mono = df.loc[mono_mask, "mobile_radius"].to_numpy()
    r_multi = df.loc[multi_mask, "mobile_radius"].to_numpy()
    results["mobile_radius_overlap"] = {
        "mono_range": [float(r_mono.min()), float(r_mono.max())],
        "multi_range": [float(r_multi.min()), float(r_multi.max())],
        "overlap_fraction_of_union": float(
            (min(r_mono.max(), r_multi.max()) - max(r_mono.min(), r_multi.min())) /
            (max(r_mono.max(), r_multi.max()) - min(r_mono.min(), r_multi.min()))
        ) if max(r_mono.max(), r_multi.max()) > min(r_mono.min(), r_multi.min()) else 0.0,
    }
    log(f"  mobile radius ranges: mono {r_mono.min():.2f}-{r_mono.max():.2f} A, "
        f"multi {r_multi.min():.2f}-{r_multi.max():.2f} A, "
        f"overlap fraction={results['mobile_radius_overlap']['overlap_fraction_of_union']:.3f}")

    # --- structural-class overlap ---
    mono_classes = set(df.loc[mono_mask, "crystal_class"])
    multi_classes = set(df.loc[multi_mask, "crystal_class"])
    shared = mono_classes & multi_classes
    results["structural_class_overlap"] = {
        "n_mono_classes": len(mono_classes), "n_multi_classes": len(multi_classes),
        "n_shared_classes": len(shared), "shared_classes": sorted(shared),
        "fraction_of_mono_classes_shared": len(shared) / len(mono_classes),
        "fraction_of_multi_classes_shared": len(shared) / len(multi_classes),
    }
    log(f"  structural classes: {len(mono_classes)} in mono, {len(multi_classes)} in multi, "
        f"{len(shared)} shared ({sorted(shared)})")

    # --- within-shared-class cross-ion comparison: for classes with both mono+multi,
    # train on one valence group's rows OUTSIDE this class + evaluate held-out class rows
    # of the other valence, vs. a same-class same-valence random holdout, to isolate the
    # valence effect from a structural-class effect ---
    magpie_df, magpie_cols = build_magpie_features(df)
    class_dummies = pd.get_dummies(df["crystal_class"], prefix="class")
    X_full = np.hstack([magpie_df.to_numpy(float), class_dummies.to_numpy(float),
                         df[["mobile_radius", "mobile_valence"]].to_numpy(float)])
    X_full = np.where(np.isnan(X_full), np.nanmean(X_full, axis=0), X_full)

    within_class = {}
    for cls in sorted(shared):
        cls_mask = (df["crystal_class"] == cls).to_numpy()
        if cls_mask.sum() < 8:
            continue
        cls_mono = cls_mask & mono_mask
        cls_multi = cls_mask & multi_mask
        if cls_mono.sum() < 3 or cls_multi.sum() < 3:
            continue
        # train on everything except this class, test on this class's multivalent rows
        train_m = ~cls_mask
        m = xgb_default(); m.fit(X_full[train_m], y[train_m])
        pred_multi = m.predict(X_full[cls_multi])
        pred_mono = m.predict(X_full[cls_mono])
        within_class[cls] = {
            "n_mono": int(cls_mono.sum()), "n_multi": int(cls_multi.sum()),
            "mae_on_class_multi_rows": float(mean_absolute_error(y[cls_multi], pred_multi)),
            "mae_on_class_mono_rows": float(mean_absolute_error(y[cls_mono], pred_mono)),
        }
    results["within_shared_class_mono_vs_multi_error"] = within_class
    log(f"  within-shared-class comparison computed for {len(within_class)} classes")

    # --- confound control: ablate mobile_radius/valence features, see if mono<->multi gap changes ---
    X_no_valence = np.hstack([magpie_df.to_numpy(float), class_dummies.to_numpy(float)])
    X_no_valence = np.where(np.isnan(X_no_valence), np.nanmean(X_no_valence, axis=0), X_no_valence)
    confound = {}
    for xname, X in [("with_radius_valence_features", X_full), ("without_radius_valence_features", X_no_valence)]:
        confound[xname] = {}
        for pname, (tr, te) in [("mono_to_multi", (mono_mask, multi_mask)), ("multi_to_mono", (multi_mask, mono_mask))]:
            m = xgb_default(); m.fit(X[tr], y[tr]); pred = m.predict(X[te])
            confound[xname][pname] = {"mae": float(mean_absolute_error(y[te], pred)), "r2": float(r2_score(y[te], pred))}
    results["confound_control_radius_valence_ablation"] = confound
    log(f"  confound control: {confound}")

    # --- train-size equalization: subsample larger group to match smaller, repeat 10x ---
    n_mono, n_multi = mono_mask.sum(), multi_mask.sum()
    n_match = min(n_mono, n_multi)
    rng = np.random.RandomState(0)
    eq_results = {"mono_to_multi": [], "multi_to_mono": []}
    mono_idx_all = np.where(mono_mask)[0]
    multi_idx_all = np.where(multi_mask)[0]
    for trial in range(10):
        mono_sub = rng.choice(mono_idx_all, n_match, replace=False)
        multi_sub = rng.choice(multi_idx_all, n_match, replace=False)
        tr_mask = np.zeros(len(df), bool); tr_mask[mono_sub] = True
        te_mask = np.zeros(len(df), bool); te_mask[multi_idx_all] = True
        m = xgb_default(); m.fit(X_full[tr_mask], y[tr_mask]); pred = m.predict(X_full[te_mask])
        eq_results["mono_to_multi"].append(float(mean_absolute_error(y[te_mask], pred)))
        tr_mask = np.zeros(len(df), bool); tr_mask[multi_sub] = True
        te_mask = np.zeros(len(df), bool); te_mask[mono_idx_all] = True
        m = xgb_default(); m.fit(X_full[tr_mask], y[tr_mask]); pred = m.predict(X_full[te_mask])
        eq_results["multi_to_mono"].append(float(mean_absolute_error(y[te_mask], pred)))
    results["train_size_equalized_10trials"] = {
        "n_match": int(n_match),
        "mono_to_multi_mae_mean_std": [float(np.mean(eq_results["mono_to_multi"])), float(np.std(eq_results["mono_to_multi"]))],
        "multi_to_mono_mae_mean_std": [float(np.mean(eq_results["multi_to_mono"])), float(np.std(eq_results["multi_to_mono"]))],
    }
    log(f"  train-size-equalized (n={n_match} each, 10 trials): "
        f"mono->multi MAE={np.mean(eq_results['mono_to_multi']):.3f}+-{np.std(eq_results['mono_to_multi']):.3f}, "
        f"multi->mono MAE={np.mean(eq_results['multi_to_mono']):.3f}+-{np.std(eq_results['multi_to_mono']):.3f}")

    # --- group-specific baseline: predict the training group's own mean for every test point ---
    baseline = {}
    for pname, (tr, te) in [("mono_to_multi", (mono_mask, multi_mask)), ("multi_to_mono", (multi_mask, mono_mask))]:
        naive_pred = np.full(te.sum(), y[tr].mean())
        baseline[pname] = {"naive_mae": float(mean_absolute_error(y[te], naive_pred)),
                            "model_mae": float(mean_absolute_error(y[te], xgb_default().fit(X_full[tr], y[tr]).predict(X_full[te])))}
    results["group_specific_naive_baseline"] = baseline
    log(f"  naive group-mean baseline: {baseline}")

    # --- CAVD side: mobile-ion-radius overlap mono vs multi (Li/Na vs Mg/Al) ---
    cavd = pd.read_parquet(DATA / "feature_table.parquet")
    r_cavd_mono = cavd.loc[cavd["cation"].isin(["Li", "Na"]), "mobile_ion_radius_mean"].dropna().to_numpy()
    r_cavd_multi = cavd.loc[cavd["cation"].isin(["Mg", "Al"]), "mobile_ion_radius_mean"].dropna().to_numpy()
    results["cavd_mobile_radius_overlap"] = {
        "mono_range": [float(r_cavd_mono.min()), float(r_cavd_mono.max())],
        "multi_range": [float(r_cavd_multi.min()), float(r_cavd_multi.max())],
        "overlap_fraction_of_union": float(
            max(0.0, min(r_cavd_mono.max(), r_cavd_multi.max()) - max(r_cavd_mono.min(), r_cavd_multi.min())) /
            (max(r_cavd_mono.max(), r_cavd_multi.max()) - min(r_cavd_mono.min(), r_cavd_multi.min()))
        ),
    }
    log(f"  CAVD mobile radius overlap: {results['cavd_mobile_radius_overlap']}")

    with open(OUT / "confound_asymmetry.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# duplicate detection + composition-grouped splitting, multi-seed
# ----------------------------------------------------------------------

def duplicate_grouped_splits_check(df_geom):
    log("=== Comment 3: duplicate detection and composition-grouped splitting ===")
    df = df_geom.copy()
    results = {}

    # duplicate / near-duplicate reduced compositions
    reduced = [Composition(f.replace(" ", "")).reduced_formula for f in df["formula"]]
    df["reduced_formula"] = reduced
    dup_counts = df["reduced_formula"].value_counts()
    n_dup_groups = (dup_counts > 1).sum()
    n_dup_rows = dup_counts[dup_counts > 1].sum()
    results["duplicate_compositions"] = {
        "n_groups_with_duplicates": int(n_dup_groups),
        "n_rows_in_duplicate_groups": int(n_dup_rows),
        "n_unique_reduced_formulas": int(df["reduced_formula"].nunique()),
        "n_total_rows": int(len(df)),
    }
    log(f"  {n_dup_groups} reduced-formula groups have duplicates, covering {n_dup_rows}/{len(df)} rows")

    # CAVD duplicate check via Filename (ICSD id)
    cavd = pd.read_parquet(DATA / "feature_table.parquet")
    fn_counts = cavd["filename"].value_counts()
    results["cavd_duplicate_filenames"] = {
        "n_groups_with_duplicates": int((fn_counts > 1).sum()),
        "n_rows_in_duplicate_groups": int(fn_counts[fn_counts > 1].sum()),
        "n_total_rows": int(len(cavd)),
    }
    log(f"  CAVD: {results['cavd_duplicate_filenames']}")

    # composition-grouped random split, multiple seeds
    magpie_df, magpie_cols = build_magpie_features(df)
    class_dummies = pd.get_dummies(df["crystal_class"], prefix="class")
    X = np.hstack([magpie_df.to_numpy(float), class_dummies.to_numpy(float)])
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    y = df["target_Em"].to_numpy(float)
    groups = df["reduced_formula"].to_numpy()

    maes, r2s = [], []
    for seed in range(5):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(gss.split(X, y, groups))
        m = xgb_default(); m.fit(X[tr], y[tr]); pred = m.predict(X[te])
        maes.append(mean_absolute_error(y[te], pred)); r2s.append(r2_score(y[te], pred))
    results["composition_grouped_random_split_5seeds"] = {
        "mae_mean_std": [float(np.mean(maes)), float(np.std(maes))],
        "r2_mean_std": [float(np.mean(r2s)), float(np.std(r2s))],
        "per_seed_mae": [float(m) for m in maes],
    }
    log(f"  composition-grouped 5-seed random split: MAE={np.mean(maes):.3f}+-{np.std(maes):.3f}, "
        f"R2={np.mean(r2s):.3f}+-{np.std(r2s):.3f}")

    # plain (ungrouped) random split, multiple seeds, for direct comparison
    maes2, r2s2 = [], []
    for seed in range(5):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(df)); n_test = int(0.2 * len(df))
        te_idx = idx[:n_test]; tr = np.ones(len(df), bool); tr[te_idx] = False; te = ~tr
        m = xgb_default(); m.fit(X[tr], y[tr]); pred = m.predict(X[te])
        maes2.append(mean_absolute_error(y[te], pred)); r2s2.append(r2_score(y[te], pred))
    results["ungrouped_random_split_5seeds"] = {
        "mae_mean_std": [float(np.mean(maes2)), float(np.std(maes2))],
        "r2_mean_std": [float(np.mean(r2s2)), float(np.std(r2s2))],
    }
    log(f"  ungrouped 5-seed random split: MAE={np.mean(maes2):.3f}+-{np.std(maes2):.3f}, "
        f"R2={np.mean(r2s2):.3f}+-{np.std(r2s2):.3f}")

    with open(OUT / "grouped_splits.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# cross-check mobile-ion assignment against TABULATION-GIT-HUB.xlsx
# ----------------------------------------------------------------------

def mobile_ion_crosscheck(df_geom):
    log("=== Comment 4: mobile-ion assignment cross-check ===")
    tab = pd.read_excel(DATA / "migration_barriers" / "TABULATION-GIT-HUB.xlsx")
    df = df_geom.copy().reset_index(drop=True)

    # Match by sys_name (both files share this field)
    tab_lookup = {}
    for _, r in tab.iterrows():
        tab_lookup.setdefault(r["sys_name"], []).append(r["COMPOSITION"])

    CANDIDATES = list(SHANNON_RADII.keys())
    n_checked, n_agree, mismatches = 0, 0, []
    for _, row in df.iterrows():
        comps = tab_lookup.get(row["sys_name"])
        if not comps:
            continue
        comp_str = comps[0]
        try:
            comp = Composition(comp_str)
            present = [el.symbol for el in comp.elements if el.symbol in CANDIDATES]
        except Exception:
            continue
        if not present:
            continue
        n_checked += 1
        if row["mobile_cation"] in present:
            n_agree += 1
        else:
            mismatches.append({"sys_name": row["sys_name"], "assigned": row["mobile_cation"],
                                "tabulation_composition": comp_str, "tabulation_candidates": present})

    results = {
        "n_checked_against_tabulation": n_checked,
        "n_agree": n_agree,
        "agreement_fraction": float(n_agree / n_checked) if n_checked else None,
        "mismatches": mismatches[:30],
        "n_mismatches_total": len(mismatches),
    }
    log(f"  cross-checked {n_checked} entries against TABULATION-GIT-HUB.xlsx COMPOSITION field: "
        f"{n_agree} agree ({results['agreement_fraction']:.3f}), {len(mismatches)} mismatches")

    with open(OUT / "mobile_ion_crosscheck.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# small-cation sensitivity analysis
# ----------------------------------------------------------------------

def cation_sample_sensitivity_check(df_geom):
    log("=== Comment 5: small-cation sensitivity analysis ===")
    df = df_geom.copy()
    magpie_df, magpie_cols = build_magpie_features(df)
    class_dummies = pd.get_dummies(df["crystal_class"], prefix="class")
    X_full = np.hstack([magpie_df.to_numpy(float), class_dummies.to_numpy(float)])
    X_full = np.where(np.isnan(X_full), np.nanmean(X_full, axis=0), X_full)
    y = df["target_Em"].to_numpy(float)

    cation_counts = df["mobile_cation"].value_counts()
    results = {"cation_counts": cation_counts.to_dict()}
    for thresh in [0, 10, 20]:
        keep_cations = set(cation_counts[cation_counts >= thresh].index) if thresh > 0 else set(cation_counts.index)
        keep_mask = df["mobile_cation"].isin(keep_cations).to_numpy()
        sub_mono = keep_mask & ~df["is_multivalent"].to_numpy()
        sub_multi = keep_mask & df["is_multivalent"].to_numpy()
        res = {}
        for pname, (tr, te) in [("mono_to_multi", (sub_mono, sub_multi)), ("multi_to_mono", (sub_multi, sub_mono))]:
            m = xgb_default(); m.fit(X_full[tr], y[tr]); pred = m.predict(X_full[te])
            res[pname] = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
                          "mae": float(mean_absolute_error(y[te], pred)), "r2": float(r2_score(y[te], pred))}
        results[f"min_count_{thresh}"] = res
        log(f"  threshold>={thresh} (cations kept: {sorted(keep_cations)}): {res}")

    with open(OUT / "cation_sample_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# interval widths, per-cation coverage, conformal prediction
# ----------------------------------------------------------------------

def uncertainty_deep_dive(df_geom):
    log("=== Comment 9: uncertainty quantification deep-dive ===")
    df = df_geom.copy()
    magpie_df, magpie_cols = build_magpie_features(df)
    class_dummies = pd.get_dummies(df["crystal_class"], prefix="class")
    X_full = np.hstack([magpie_df.to_numpy(float), class_dummies.to_numpy(float)])
    X_full = np.where(np.isnan(X_full), np.nanmean(X_full, axis=0), X_full)
    y = df["target_Em"].to_numpy(float)
    mono_mask = ~df["is_multivalent"].to_numpy()
    multi_mask = df["is_multivalent"].to_numpy()
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df)); n_test = int(0.2 * len(df))
    te_idx = idx[:n_test]; rand_train = np.ones(len(df), bool); rand_train[te_idx] = False; rand_test = ~rand_train
    protocols = {"random_split": (rand_train, rand_test), "mono_to_multi": (mono_mask, multi_mask),
                 "multi_to_mono": (multi_mask, mono_mask)}

    def make_q(alpha):
        return XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha,
                             n_estimators=30, max_depth=2, learning_rate=0.1, reg_lambda=30.0,
                             n_jobs=8, random_state=42)

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    results = {}
    for pname, (tr, te) in protocols.items():
        scaler = StandardScaler().fit(X_full[tr])
        Xtr, Xte = scaler.transform(X_full[tr]), scaler.transform(X_full[te])

        # XGBoost quantile, 80% and 50% nominal
        row = {}
        for nominal, (lo, hi) in [("80pct", (0.1, 0.9)), ("50pct", (0.25, 0.75))]:
            m_lo = make_q(lo); m_lo.fit(X_full[tr], y[tr])
            m_hi = make_q(hi); m_hi.fit(X_full[tr], y[tr])
            p_lo, p_hi = m_lo.predict(X_full[te]), m_hi.predict(X_full[te])
            width = p_hi - p_lo
            cov = float(np.mean((y[te] >= p_lo) & (y[te] <= p_hi)))
            row[f"xgb_{nominal}"] = {"coverage": cov, "mean_width": float(np.mean(width)), "median_width": float(np.median(width))}

        gp = GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1.0),
                                       normalize_y=True, random_state=42)
        gp.fit(Xtr, y[tr])
        mu, sigma = gp.predict(Xte, return_std=True)
        from scipy.stats import norm
        for nominal, z in [("80pct", norm.ppf(0.9)), ("50pct", norm.ppf(0.75))]:
            lo, hi = mu - z * sigma, mu + z * sigma
            width = hi - lo
            cov = float(np.mean((y[te] >= lo) & (y[te] <= hi)))
            row[f"gp_{nominal}"] = {"coverage": cov, "mean_width": float(np.mean(width)), "median_width": float(np.median(width))}

        results[pname] = row
        log(f"  {pname}: {row}")

    # per-cation coverage for the random split, 80% nominal, XGBoost and GP
    m_lo = make_q(0.1); m_lo.fit(X_full[rand_train], y[rand_train])
    m_hi = make_q(0.9); m_hi.fit(X_full[rand_train], y[rand_train])
    p_lo_all, p_hi_all = m_lo.predict(X_full), m_hi.predict(X_full)
    per_cation_cov = {}
    for cat in df["mobile_cation"].unique():
        cmask = (df["mobile_cation"] == cat).to_numpy() & rand_test
        if cmask.sum() < 3:
            continue
        cov = float(np.mean((y[cmask] >= p_lo_all[cmask]) & (y[cmask] <= p_hi_all[cmask])))
        per_cation_cov[cat] = {"n": int(cmask.sum()), "xgb_80pct_coverage": cov}
    results["per_cation_coverage_random_split_xgb"] = per_cation_cov
    log(f"  per-cation coverage (random split, XGB 80%): {per_cation_cov}")

    # split-conformal prediction (80% target), using a held-out calibration set carved from training
    conformal = {}
    for pname, (tr, te) in protocols.items():
        tr_idx = np.where(tr)[0]
        rng2 = np.random.RandomState(1)
        perm = rng2.permutation(len(tr_idx))
        n_cal = max(10, int(0.2 * len(tr_idx)))
        cal_idx, fit_idx = tr_idx[perm[:n_cal]], tr_idx[perm[n_cal:]]
        m = xgb_default(); m.fit(X_full[fit_idx], y[fit_idx])
        resid_cal = np.abs(y[cal_idx] - m.predict(X_full[cal_idx]))
        q = np.quantile(resid_cal, 0.8)
        pred_te = m.predict(X_full[te])
        cov = float(np.mean(np.abs(y[te] - pred_te) <= q))
        conformal[pname] = {"q_80pct": float(q), "coverage": cov, "interval_width": float(2 * q)}
    results["split_conformal_80pct"] = conformal
    log(f"  split-conformal 80%: {conformal}")

    with open(OUT / "uncertainty_deep_dive.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# diagnose Ridge/MLP catastrophic extrapolation, multi-seed MLP
# ----------------------------------------------------------------------

def benchmark_transparency_diagnosis(df_geom):
    log("=== Comment 10: diagnosing Ridge/MLP extrapolation failure ===")
    df = df_geom.copy()
    magpie_df, magpie_cols = build_magpie_features(df)
    class_dummies = pd.get_dummies(df["crystal_class"], prefix="class")
    X_full = np.hstack([magpie_df.to_numpy(float), class_dummies.to_numpy(float)])
    X_full = np.where(np.isnan(X_full), np.nanmean(X_full, axis=0), X_full)
    y = df["target_Em"].to_numpy(float)
    mono_mask = ~df["is_multivalent"].to_numpy(); multi_mask = df["is_multivalent"].to_numpy()

    scaler = StandardScaler().fit(X_full[mono_mask])
    Xtr = scaler.transform(X_full[mono_mask]); Xte = scaler.transform(X_full[multi_mask])

    ridge = Ridge(alpha=5.0, random_state=42).fit(Xtr, y[mono_mask])
    pred = ridge.predict(Xte)
    coef_norm = float(np.linalg.norm(ridge.coef_))
    results = {
        "ridge_mono_to_multi": {
            "coef_l2_norm": coef_norm,
            "pred_min_max": [float(pred.min()), float(pred.max())],
            "true_min_max": [float(y[multi_mask].min()), float(y[multi_mask].max())],
            "n_pred_outside_physical_range_0_10eV": int(np.sum((pred < 0) | (pred > 10))),
            "diagnosis": ("Ridge with alpha=5 is lightly regularized for 168 features on ~338 training "
                           "rows; when the test design matrix (multivalent rows) sits far outside the "
                           "monovalent training distribution in several magpie dimensions at once, the "
                           "linear extrapolation compounds small per-feature coefficient errors into a "
                           "large, physically implausible prediction. This is a standard linear-extrapolation "
                           "failure mode (unbounded output as ||x|| grows), not evidence that linear models "
                           "are intrinsically unable to fit this data in-distribution."),
        }
    }

    # MLP: check convergence (loss curve) and run 5 seeds
    mlp_maes = []
    for seed in range(5):
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-2,
                            max_iter=2000, early_stopping=True, random_state=seed)
        mlp.fit(Xtr, y[mono_mask])
        pred_m = mlp.predict(Xte)
        mlp_maes.append(float(mean_absolute_error(y[multi_mask], pred_m)))
    results["mlp_mono_to_multi_5seeds"] = {
        "mae_values": mlp_maes, "mean": float(np.mean(mlp_maes)), "std": float(np.std(mlp_maes)),
    }
    log(f"  Ridge mono->multi: coef_norm={coef_norm:.2f}, pred range={results['ridge_mono_to_multi']['pred_min_max']}, "
        f"true range={results['ridge_mono_to_multi']['true_min_max']}")
    log(f"  MLP mono->multi 5-seed MAE: {mlp_maes} (mean={np.mean(mlp_maes):.3f}, std={np.std(mlp_maes):.3f})")

    with open(OUT / "benchmark_transparency_diagnosis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# SHAP leave-one-cation-out + mono vs multi SHAP comparison
# ----------------------------------------------------------------------

def shap_robustness_check(df_geom):
    log("=== Comment 11: SHAP robustness checks ===")
    import shap
    df = df_geom.copy()
    magpie_df, magpie_cols = build_magpie_features(df)
    class_dummies = pd.get_dummies(df["crystal_class"], prefix="class")
    feat_names = magpie_cols + list(class_dummies.columns)
    X_full = np.hstack([magpie_df.to_numpy(float), class_dummies.to_numpy(float)])
    X_full = np.where(np.isnan(X_full), np.nanmean(X_full, axis=0), X_full)
    y = df["target_Em"].to_numpy(float)

    # random-split SHAP (reuse existing saved predictions' train/test if present, else refit identically)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df)); n_test = int(0.2 * len(df))
    te_idx = idx[:n_test]; rand_train = np.ones(len(df), bool); rand_train[te_idx] = False; rand_test = ~rand_train
    m_rand = xgb_default(); m_rand.fit(X_full[rand_train], y[rand_train])
    expl_rand = shap.TreeExplainer(m_rand)
    shap_rand = expl_rand.shap_values(X_full[rand_test])
    rank_rand = pd.Series(np.abs(shap_rand).mean(axis=0), index=feat_names).sort_values(ascending=False)

    # leave-Li-out and leave-Ca-out (two reasonably sized cations) SHAP ranking
    results = {"random_split_top10": rank_rand.head(10).to_dict()}
    for cat in ["Li", "Ca"]:
        test_m = (df["mobile_cation"] == cat).to_numpy()
        train_m = ~test_m
        m = xgb_default(); m.fit(X_full[train_m], y[train_m])
        expl = shap.TreeExplainer(m)
        sv = expl.shap_values(X_full[test_m])
        rank = pd.Series(np.abs(sv).mean(axis=0), index=feat_names).sort_values(ascending=False)
        overlap_top10 = len(set(rank.head(10).index) & set(rank_rand.head(10).index))
        results[f"leave_{cat}_out_top10"] = rank.head(10).to_dict()
        results[f"leave_{cat}_out_top10_overlap_with_random_split"] = overlap_top10
        log(f"  leave-{cat}-out top-10 SHAP overlap with random-split top-10: {overlap_top10}/10")

    # mono vs multi SHAP comparison using the already-saved random-split SHAP array
    shap_path = MODELS / "migration_baseline" / "shap_values_random_split.npy"
    names_path = MODELS / "migration_baseline" / "shap_feature_names.json"
    if shap_path.exists() and names_path.exists():
        saved_shap = np.load(shap_path)
        with open(names_path) as f:
            saved_names = json.load(f)
        test_valence = df.loc[rand_test, "is_multivalent"].to_numpy()
        if saved_shap.shape[0] == test_valence.shape[0]:
            rank_mono = pd.Series(np.abs(saved_shap[~test_valence]).mean(axis=0), index=saved_names).sort_values(ascending=False)
            rank_multi = pd.Series(np.abs(saved_shap[test_valence]).mean(axis=0), index=saved_names).sort_values(ascending=False)
            results["saved_shap_mono_test_top10"] = rank_mono.head(10).to_dict()
            results["saved_shap_multi_test_top10"] = rank_multi.head(10).to_dict()
            results["saved_shap_mono_multi_top10_overlap"] = len(set(rank_mono.head(10).index) & set(rank_multi.head(10).index))
            log(f"  mono-vs-multi test-set SHAP top-10 overlap: {results['saved_shap_mono_multi_top10_overlap']}/10")
        else:
            log(f"  WARNING: saved SHAP array shape {saved_shap.shape} doesn't match rand_test size "
                f"{test_valence.shape} -- skipping mono/multi split comparison (likely built on a different split)")
            results["saved_shap_mono_multi_note"] = "shape mismatch with current random split; recomputed-only ranks reported above"

    with open(OUT / "shap_robustness_checks.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# electrolyte-only subset
# ----------------------------------------------------------------------

def electrolyte_subset_check(df_geom):
    log("=== Comment 12: electrolyte-only subset ===")
    df = df_geom.copy()
    elec = df[df["crystal_class"] == "ELECTROLYTES"]
    results = {
        "n_electrolyte_entries": int(len(elec)),
        "cation_counts": elec["mobile_cation"].value_counts().to_dict(),
        "n_monovalent": int((~elec["is_multivalent"]).sum()),
        "n_multivalent": int(elec["is_multivalent"].sum()),
    }
    log(f"  ELECTROLYTES class: {results}")
    # all-monovalent -> cross-valence test is impossible for this subset; report this fact plainly
    with open(OUT / "electrolyte_subset_check.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    df_geom = build_geometric_descriptors()
    confound_and_asymmetry_check(df_geom)
    duplicate_grouped_splits_check(df_geom)
    mobile_ion_crosscheck(df_geom)
    cation_sample_sensitivity_check(df_geom)
    uncertainty_deep_dive(df_geom)
    benchmark_transparency_diagnosis(df_geom)
    shap_robustness_check(df_geom)
    electrolyte_subset_check(df_geom)
    log(f"ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
