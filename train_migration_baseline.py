# XGBoost baseline for DFT-NEB migration barriers (Devi et al. 2025), using magpie composition
# statistics plus one-hot structural class, evaluated in-distribution, leave-one-structural-
# class-out, and under cross-valence extrapolation in both directions.
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

DATA_DIR = Path("/glade/work/ajalali/Multivalent-Ion-Conductor-ML-project/data")
OUT_DIR = Path("/glade/work/ajalali/Multivalent-Ion-Conductor-ML-project/models/migration_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_CLASS_SIZE = 10


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_features(df):
    ep = ElementProperty.from_preset("magpie")
    comps = [Composition(f.replace(" ", "")).reduced_composition for f in df["formula"]]
    feat_df = ep.featurize_dataframe(pd.DataFrame({"composition": comps}), "composition", ignore_errors=True)
    magpie_cols = [c for c in feat_df.columns if c != "composition"]
    X_magpie = feat_df[magpie_cols].to_numpy(dtype=float)

    class_dummies = pd.get_dummies(df["crystal_class"], prefix="class").to_numpy(dtype=float)
    X = np.hstack([X_magpie, class_dummies])
    col_names = magpie_cols + [f"class_{c}" for c in pd.get_dummies(df["crystal_class"], prefix="class").columns]
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), col_mean, X)
    return X, col_names


def fit_eval(X, y, train_mask, test_mask, n_estimators=300, max_depth=4):
    model = XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
        n_jobs=8, random_state=42,
    )
    model.fit(X[train_mask], y[train_mask])
    pred = model.predict(X[test_mask])
    return {
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "mae": float(mean_absolute_error(y[test_mask], pred)),
        "r2": float(r2_score(y[test_mask], pred)) if test_mask.sum() > 1 else None,
    }, pred


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA_DIR / "migration_barrier_table.parquet")
    df = df.dropna(subset=["mobile_cation"]).reset_index(drop=True)
    log(f"{len(df)} entries with a resolved mobile cation")

    X, feat_names = build_features(df)
    y = df["target_Em"].to_numpy(dtype=float)
    log(f"Feature matrix: {X.shape}")

    results = {}

    # --- 1. random split ---
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df))
    n_test = int(0.2 * len(df))
    test_idx = idx[:n_test]
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[test_idx] = False
    test_mask = ~train_mask
    m, pred = fit_eval(X, y, train_mask, test_mask)
    results["random_split"] = m
    log(f"Random split: {m}")
    pred_df = df.loc[test_mask, ["jid", "formula", "crystal_class", "mobile_cation"]].copy()
    pred_df["target_Em"] = y[test_mask]
    pred_df["pred_Em"] = pred
    pred_df.to_csv(OUT_DIR / "predictions_random_split.csv", index=False)

    # --- 2. leave-one-structural-class-out ---
    log("Leave-one-structural-class-out...")
    class_results = {}
    for cls, count in df["crystal_class"].value_counts().items():
        if count < MIN_CLASS_SIZE:
            continue
        test_m = (df["crystal_class"] == cls).to_numpy()
        train_m = ~test_m
        m, _ = fit_eval(X, y, train_m, test_m)
        class_results[cls] = m
        log(f"  leave-{cls}-out: n={m['n_test']} MAE={m['mae']:.3f} R2={m['r2']}")
    results["leave_one_class_out"] = class_results

    # --- 3. cross-valence extrapolation ---
    log("Cross-valence extrapolation...")
    mono_mask = ~df["is_multivalent"].to_numpy()
    multi_mask = df["is_multivalent"].to_numpy()
    m_mono2multi, pred_m2m = fit_eval(X, y, mono_mask, multi_mask)
    results["train_mono_test_multi"] = m_mono2multi
    log(f"  train-monovalent -> test-multivalent: {m_mono2multi}")
    m_multi2mono, pred_mm2 = fit_eval(X, y, multi_mask, mono_mask)
    results["train_multi_test_mono"] = m_multi2mono
    log(f"  train-multivalent -> test-monovalent: {m_multi2mono}")

    pred_df2 = df.loc[multi_mask, ["jid", "formula", "crystal_class", "mobile_cation"]].copy()
    pred_df2["target_Em"] = y[multi_mask]
    pred_df2["pred_Em"] = pred_m2m
    pred_df2.to_csv(OUT_DIR / "predictions_mono_to_multi.csv", index=False)

    # --- 4. leave-one-cation-out (only cations with >= 15 entries) ---
    log("Leave-one-cation-out...")
    cation_results = {}
    for cat, count in df["mobile_cation"].value_counts().items():
        if count < 15:
            continue
        test_m = (df["mobile_cation"] == cat).to_numpy()
        train_m = ~test_m
        m, _ = fit_eval(X, y, train_m, test_m)
        cation_results[cat] = m
        log(f"  leave-{cat}-out: n={m['n_test']} MAE={m['mae']:.3f} R2={m['r2']}")
    results["leave_one_cation_out"] = cation_results

    # --- baseline: naive mean-predictor MAE, for context ---
    naive_mae = float(mean_absolute_error(y[test_mask], np.full(test_mask.sum(), y[train_mask].mean())))
    results["naive_mean_predictor_mae_random_split"] = naive_mae
    log(f"Naive mean-predictor MAE (random split test set): {naive_mae:.3f}")

    results["n_total"] = len(df)
    results["n_monovalent"] = int(mono_mask.sum())
    results["n_multivalent"] = int(multi_mask.sum())
    results["runtime_seconds"] = time.time() - t0
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
