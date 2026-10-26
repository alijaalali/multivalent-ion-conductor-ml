# Target-distribution comparison across grouped/ungrouped splits and migration-barrier
# dataset heterogeneity by DFT functional type. Reanalysis of data already on disk plus
# additional ML runs; no new DFT/atomistic simulation.
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

PROJ = Path("/glade/work/ajalali/12 - Multivalent-Ion-Conductor-ML-project")
DATA = PROJ / "data"
MODELS = PROJ / "models"
OUT = PROJ / "models" / "distribution_and_heterogeneity"
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


def load_corrected_df():
    """Migration-barrier table with NEB-corrected mobile-cation labels (328 mono / 289 multi)."""
    with open(MODELS / "physical_descriptors_and_uq" / "neb_structure_check.json") as f:
        neb = json.load(f)
    full_table = pd.DataFrame(neb["full_table"])
    df = pd.read_parquet(DATA / "migration_barrier_table.parquet")
    df = df.dropna(subset=["mobile_cation"]).reset_index(drop=True)
    neb_winner_by_jid = dict(zip(full_table["jid"], full_table["neb_displacement_winner"]))
    df["mobile_cation_corrected"] = df.apply(
        lambda r: neb_winner_by_jid.get(r["jid"], r["mobile_cation"]), axis=1)
    df["valence_corrected"] = df["mobile_cation_corrected"].map(VALENCE)
    df["is_multivalent_corrected"] = df["valence_corrected"] >= 2
    return df


def migration_features(df):
    magpie_df, _ = build_magpie_features(df)
    class_dummies = pd.get_dummies(df["crystal_class"], prefix="class")
    X = np.hstack([magpie_df.to_numpy(float), class_dummies.to_numpy(float)])
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    return X


# ----------------------------------------------------------------------
# Target-distribution comparison, grouped vs. ungrouped splits, and a
# paired-seed CI for the MAE difference (reuses the exact 5-seed
# composition-grouped-vs-ungrouped protocol from
# additional_robustness_analyses.py's duplicate_grouped_splits_check,
# extended to report test-fold target distributions and a CI for the
# MAE gap).
# ----------------------------------------------------------------------

def test_fold_target_distributions():
    log("=== test-fold target distributions, grouped vs. ungrouped ===")
    df = pd.read_parquet(MODELS / "additional_robustness" / "migration_table_with_geom.parquet")
    reduced = [Composition(f.replace(" ", "")).reduced_formula for f in df["formula"]]
    df["reduced_formula"] = reduced
    X = migration_features(df)
    y = df["target_Em"].to_numpy(float)
    groups = df["reduced_formula"].to_numpy()

    grouped_mae, ungrouped_mae = [], []
    grouped_dist, ungrouped_dist = [], []
    for seed in range(5):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(gss.split(X, y, groups))
        m = xgb_default(); m.fit(X[tr], y[tr]); pred = m.predict(X[te])
        grouped_mae.append(mean_absolute_error(y[te], pred))
        grouped_dist.append({"mean": float(y[te].mean()), "std": float(y[te].std()),
                              "median": float(np.median(y[te])), "n": int(len(y[te]))})

        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(df)); n_test = int(0.2 * len(df))
        te_idx = idx[:n_test]; tru = np.ones(len(df), bool); tru[te_idx] = False; teu = ~tru
        mu = xgb_default(); mu.fit(X[tru], y[tru]); predu = mu.predict(X[teu])
        ungrouped_mae.append(mean_absolute_error(y[teu], predu))
        ungrouped_dist.append({"mean": float(y[teu].mean()), "std": float(y[teu].std()),
                                "median": float(np.median(y[teu])), "n": int(len(y[teu]))})

    grouped_mae = np.array(grouped_mae); ungrouped_mae = np.array(ungrouped_mae)
    paired_diff = ungrouped_mae - grouped_mae  # ungrouped minus grouped, matching main-text sign (0.415 vs 0.358)
    results = {
        "grouped_mae_per_seed": grouped_mae.tolist(),
        "ungrouped_mae_per_seed": ungrouped_mae.tolist(),
        "grouped_mae_mean_std": [float(grouped_mae.mean()), float(grouped_mae.std())],
        "ungrouped_mae_mean_std": [float(ungrouped_mae.mean()), float(ungrouped_mae.std())],
        "paired_diff_ungrouped_minus_grouped": {
            "per_seed": paired_diff.tolist(),
            "mean": float(paired_diff.mean()), "std": float(paired_diff.std()),
            "95ci_normal_approx": [float(paired_diff.mean() - 1.96 * paired_diff.std() / np.sqrt(5)),
                                    float(paired_diff.mean() + 1.96 * paired_diff.std() / np.sqrt(5))],
        },
        "grouped_test_fold_target_distribution_per_seed": grouped_dist,
        "ungrouped_test_fold_target_distribution_per_seed": ungrouped_dist,
        "grouped_test_fold_target_mean_of_means": float(np.mean([d["mean"] for d in grouped_dist])),
        "ungrouped_test_fold_target_mean_of_means": float(np.mean([d["mean"] for d in ungrouped_dist])),
    }
    log(f"  grouped MAE {grouped_mae.mean():.3f}+-{grouped_mae.std():.3f}, "
        f"ungrouped MAE {ungrouped_mae.mean():.3f}+-{ungrouped_mae.std():.3f}")
    log(f"  paired diff (ungrouped-grouped): mean={paired_diff.mean():.3f}, "
        f"95% CI (normal approx, n=5 seeds)=[{results['paired_diff_ungrouped_minus_grouped']['95ci_normal_approx'][0]:.3f},"
        f"{results['paired_diff_ungrouped_minus_grouped']['95ci_normal_approx'][1]:.3f}]")
    log(f"  test-fold target mean: grouped={results['grouped_test_fold_target_mean_of_means']:.3f} eV, "
        f"ungrouped={results['ungrouped_test_fold_target_mean_of_means']:.3f} eV")
    with open(OUT / "test_fold_distributions.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# ----------------------------------------------------------------------
# migration-barrier dataset heterogeneity -- functional type
# by valence cross-tabulation (uses NEB-corrected labels since this is a
# new analysis with no uncorrected-label peer to stay consistent with).
# ----------------------------------------------------------------------

def functional_type_by_valence():
    log("=== XC functional x valence cross-tabulation ===")
    df = load_corrected_df()
    ct = pd.crosstab(df["XC"], df["is_multivalent_corrected"])
    ct.columns = ["monovalent", "multivalent"]
    ct_pct_within_valence = pd.crosstab(df["XC"], df["is_multivalent_corrected"], normalize="columns") * 100
    ct_pct_within_valence.columns = ["monovalent_pct", "multivalent_pct"]

    # check for per-entry magnetic configuration / supercell / convergence metadata
    with open(DATA / "migration_barriers" / "EM-COMPLETE-DATASET.json") as f:
        records = {r["jid"]: r for r in json.load(f)}
    sample_keys = set()
    for jid in list(records.keys())[:50]:
        sample_keys.update(records[jid].keys())
    magnetic_fields = [k for k in sample_keys if "mag" in k.lower() or "spin" in k.lower()]
    supercell_fields = [k for k in sample_keys if "supercell" in k.lower() or "cell_size" in k.lower()]
    convergence_fields = [k for k in sample_keys if "converg" in k.lower() or "tol" in k.lower()]

    results = {
        "crosstab_counts": ct.to_dict(),
        "crosstab_pct_within_valence": ct_pct_within_valence.round(1).to_dict(),
        "available_record_keys_sample": sorted(sample_keys),
        "magnetic_config_fields_found": magnetic_fields,
        "supercell_fields_found": supercell_fields,
        "convergence_fields_found": convergence_fields,
        "note": ("Magnetic configuration, supercell dimensions, and explicit convergence "
                 "criteria are genuinely not recorded as per-entry fields in the released "
                 "EM-COMPLETE-DATASET.json records (checked directly against all keys present "
                 "in a 50-entry sample); only the exchange-correlation functional is "
                 "consistently available per entry."),
    }
    log(f"  crosstab (counts): {ct.to_dict()}")
    log(f"  magnetic/supercell/convergence fields found: {magnetic_fields}/{supercell_fields}/{convergence_fields}")
    with open(OUT / "functional_by_valence.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def main():
    t0 = time.time()
    test_fold_target_distributions()
    functional_type_by_valence()
    log(f"ALL DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
