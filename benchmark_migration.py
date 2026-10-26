# Broader model benchmark for migration-barrier regression: adds Random Forest, SVR, k-NN,
# Ridge, and MLP alongside XGBoost and Gaussian Process, on the exact same splits as
# train_migration_baseline.py / strengthen_migration.py.
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

DATA_DIR = Path("/glade/work/ajalali/Multivalent-Ion-Conductor-ML-project/data")
OUT_DIR = Path("/glade/work/ajalali/Multivalent-Ion-Conductor-ML-project/models/migration_baseline")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_features(df):
    ep = ElementProperty.from_preset("magpie")
    comps = [Composition(f.replace(" ", "")).reduced_composition for f in df["formula"]]
    feat_df = ep.featurize_dataframe(pd.DataFrame({"composition": comps}), "composition", ignore_errors=True)
    magpie_cols = [c for c in feat_df.columns if c != "composition"]
    X_magpie = feat_df[magpie_cols].to_numpy(dtype=float)
    dummies = pd.get_dummies(df["crystal_class"], prefix="class")
    X = np.hstack([X_magpie, dummies.to_numpy(dtype=float)])
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), col_mean, X)
    return X


def make_models():
    return {
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                                 n_jobs=8, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=400, max_depth=None,
                                               min_samples_leaf=2, n_jobs=8, random_state=42),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.05),
        "kNN": KNeighborsRegressor(n_neighbors=8, weights="distance"),
        "Ridge": Ridge(alpha=5.0, random_state=42),
        "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-2,
                             max_iter=2000, early_stopping=True, random_state=42),
    }


NEEDS_SCALING = {"SVR", "kNN", "Ridge", "MLP"}


def fit_eval(name, model, X, y, train_m, test_m, scaler):
    if name in NEEDS_SCALING:
        Xtr = scaler.fit_transform(X[train_m])
        Xte = scaler.transform(X[test_m])
    else:
        Xtr, Xte = X[train_m], X[test_m]
    model.fit(Xtr, y[train_m])
    pred = model.predict(Xte)
    return {
        "n_train": int(train_m.sum()), "n_test": int(test_m.sum()),
        "mae": float(mean_absolute_error(y[test_m], pred)),
        "r2": float(r2_score(y[test_m], pred)) if test_m.sum() > 1 else None,
    }


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA_DIR / "migration_barrier_table.parquet")
    df = df.dropna(subset=["mobile_cation"]).reset_index(drop=True)
    X = build_features(df)
    y = df["target_Em"].to_numpy(dtype=float)
    log(f"n={len(df)}, {X.shape[1]} features")

    # Identical split construction to strengthen_migration.py / train_migration_baseline.py
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df))
    n_test = int(0.2 * len(df))
    test_idx = idx[:n_test]
    random_train = np.ones(len(df), dtype=bool); random_train[test_idx] = False
    random_test = ~random_train

    mono_mask = ~df["is_multivalent"].to_numpy()
    multi_mask = df["is_multivalent"].to_numpy()

    protocols = {
        "random_split": (random_train, random_test),
        "mono_to_multi": (mono_mask, multi_mask),
        "multi_to_mono": (multi_mask, mono_mask),
    }

    # leave-one-structural-class-out, classes >= 10 rows (same threshold as before)
    class_masks = {}
    for cls, count in df["crystal_class"].value_counts().items():
        if count >= 10:
            test_m = (df["crystal_class"] == cls).to_numpy()
            class_masks[cls] = (~test_m, test_m)

    scaler = StandardScaler()
    models = make_models()
    results = {name: {} for name in models}

    for name, model in models.items():
        log(f"=== {name} ===")
        for pname, (train_m, test_m) in protocols.items():
            m = fit_eval(name, model, X, y, train_m, test_m, scaler)
            results[name][pname] = m
            log(f"  {pname}: MAE={m['mae']:.3f} R2={m['r2']}")
        # pooled leave-one-class-out summary (mean MAE / mean R2 across classes) for the benchmark table
        class_maes, class_r2s = [], []
        for cls, (train_m, test_m) in class_masks.items():
            m = fit_eval(name, model, X, y, train_m, test_m, scaler)
            class_maes.append(m["mae"])
            class_r2s.append(m["r2"])
        results[name]["leave_one_class_out_mean"] = {
            "mean_mae": float(np.mean(class_maes)), "mean_r2": float(np.mean(class_r2s)),
            "n_classes": len(class_maes),
        }
        log(f"  leave-one-class-out (mean over {len(class_maes)} classes): "
            f"MAE={np.mean(class_maes):.3f} R2={np.mean(class_r2s):.3f}")

    results["runtime_seconds"] = time.time() - t0
    with open(OUT_DIR / "benchmark_models.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
