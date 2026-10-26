# Random Forest baseline predicting ion-transport network percolation (binary and 4-class)
# from composition/symmetry-only features, evaluated in-distribution and under cross-valence
# (monovalent-vs-multivalent) extrapolation in both directions.
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split

DATA_DIR = Path("/glade/work/ajalali/Multivalent-Ion-Conductor-ML-project/data")
OUT_DIR = Path("/glade/work/ajalali/Multivalent-Ion-Conductor-ML-project/models/baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MONOVALENT = ["Li", "Na"]
MULTIVALENT = ["Mg", "Al"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data():
    df = pd.read_parquet(DATA_DIR / "feature_table.parquet")
    drop_cols = ["cation", "filename", "IND", "percolates"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].to_numpy(dtype=float)
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), col_mean, X)
    return df, X, feature_cols


def make_model():
    return RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=8, random_state=42,
    )


def fit_eval(X, y, train_mask, test_mask, label):
    model = make_model()
    model.fit(X[train_mask], y[train_mask])
    pred = model.predict(X[test_mask])
    metrics = {
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "accuracy": float(accuracy_score(y[test_mask], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_mask], pred)),
        "macro_f1": float(f1_score(y[test_mask], pred, average="macro")),
        "confusion_matrix": confusion_matrix(y[test_mask], pred).tolist(),
        "classes": sorted(np.unique(y).tolist()),
    }
    log(f"  [{label}] n_train={metrics['n_train']} n_test={metrics['n_test']} "
        f"acc={metrics['accuracy']:.3f} bal_acc={metrics['balanced_accuracy']:.3f} "
        f"macro_f1={metrics['macro_f1']:.3f}")
    return model, metrics


def run_target(df, X, feature_cols, target_col):
    log(f"=== Target: {target_col} ===")
    y = df[target_col].to_numpy()
    results = {}

    # --- 1. stratified random split, pooled across all four cations ---
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=df[["cation", target_col]].astype(str).agg("_".join, axis=1)
    )
    train_mask = np.zeros(len(df), dtype=bool); train_mask[train_idx] = True
    test_mask = np.zeros(len(df), dtype=bool); test_mask[test_idx] = True
    _, results["random_split"] = fit_eval(X, y, train_mask, test_mask, "random_split")

    # --- 2. monovalent -> multivalent extrapolation ---
    mono_mask = df["cation"].isin(MONOVALENT).to_numpy()
    multi_mask = df["cation"].isin(MULTIVALENT).to_numpy()
    _, results["train_mono_test_multi"] = fit_eval(X, y, mono_mask, multi_mask, "train_mono_test_multi")

    # --- 3. multivalent -> monovalent extrapolation ---
    _, results["train_multi_test_mono"] = fit_eval(X, y, multi_mask, mono_mask, "train_multi_test_mono")

    # --- 4. per-cation breakdown when training on the other three ---
    per_cation = {}
    for cat in ["Li", "Na", "Mg", "Al"]:
        test_m = (df["cation"] == cat).to_numpy()
        train_m = ~test_m
        _, m = fit_eval(X, y, train_m, test_m, f"leave_{cat}_out")
        per_cation[cat] = m
    results["leave_one_cation_out"] = per_cation

    return results


def main():
    t0 = time.time()
    df, X, feature_cols = load_data()
    log(f"Loaded {df.shape}, {len(feature_cols)} features")
    log(f"Feature columns: {feature_cols}")

    all_results = {
        "percolates_binary": run_target(df, X, feature_cols, "percolates"),
        "IND_multiclass": run_target(df, X, feature_cols, "IND"),
        "feature_columns": feature_cols,
        "runtime_seconds": time.time() - t0,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
