# Tuned quantile-regression uncertainty sweep, a Gaussian Process point-estimate/uncertainty
# comparison against XGBoost, and SHAP feature importance for the migration-barrier regression.
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

DATA_DIR = Path("/glade/work/ajalali/Multivalent-Ion-Conductor-ML-project/data")
OUT_DIR = Path("/glade/work/ajalali/Multivalent-Ion-Conductor-ML-project/models/migration_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    col_names = magpie_cols + [f"class_{c}" for c in dummies.columns]
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), col_mean, X)
    return X, col_names


PROTOCOLS = {}  # filled in main() once df is loaded


def make_quantile_model(alpha, n_estimators, max_depth, reg_lambda):
    return XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=alpha,
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=reg_lambda,
        n_jobs=8, random_state=42,
    )


def eval_quantile_setting(X, y, protocols, n_estimators, max_depth, reg_lambda):
    """Average coverage across all protocol test masks for one hyperparameter setting."""
    coverages, widths = [], []
    for name, (train_m, test_m) in protocols.items():
        m10 = make_quantile_model(0.1, n_estimators, max_depth, reg_lambda)
        m10.fit(X[train_m], y[train_m])
        m90 = make_quantile_model(0.9, n_estimators, max_depth, reg_lambda)
        m90.fit(X[train_m], y[train_m])
        p10, p90 = m10.predict(X[test_m]), m90.predict(X[test_m])
        covered = (y[test_m] >= p10) & (y[test_m] <= p90)
        coverages.append(float(np.mean(covered)))
        widths.append(float(np.mean(p90 - p10)))
    return float(np.mean(coverages)), coverages, widths


def make_gp():
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-4, 10.0))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=42)


def fit_eval_gp(X, y, train_m, test_m, scaler):
    Xtr = scaler.fit_transform(X[train_m])
    Xte = scaler.transform(X[test_m])
    gp = make_gp()
    gp.fit(Xtr, y[train_m])
    pred, std = gp.predict(Xte, return_std=True)
    lo, hi = pred - 1.2816 * std, pred + 1.2816 * std  # 80% interval under Gaussian assumption
    covered = (y[test_m] >= lo) & (y[test_m] <= hi)
    return {
        "n_train": int(train_m.sum()), "n_test": int(test_m.sum()),
        "mae": float(mean_absolute_error(y[test_m], pred)),
        "r2": float(r2_score(y[test_m], pred)) if test_m.sum() > 1 else None,
        "coverage_80": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(hi - lo)),
    }


def fit_eval_xgb(X, y, train_m, test_m):
    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
        n_jobs=8, random_state=42,
    )
    model.fit(X[train_m], y[train_m])
    pred = model.predict(X[test_m])
    return model, {
        "n_train": int(train_m.sum()), "n_test": int(test_m.sum()),
        "mae": float(mean_absolute_error(y[test_m], pred)),
        "r2": float(r2_score(y[test_m], pred)) if test_m.sum() > 1 else None,
    }


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA_DIR / "migration_barrier_table.parquet")
    df = df.dropna(subset=["mobile_cation"]).reset_index(drop=True)
    X, feat_names = build_features(df)
    y = df["target_Em"].to_numpy(dtype=float)
    log(f"n={len(df)}, {X.shape[1]} features")

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

    # --- 1. Quantile regression hyperparameter sweep ---
    log("=== Quantile-regression regularization sweep ===")
    sweep_grid = [
        (500, 5, 1.0), (100, 4, 5.0), (50, 3, 10.0),
        (30, 2, 20.0), (20, 2, 30.0), (15, 2, 40.0),
    ]
    sweep_results = []
    for n_est, depth, lam in sweep_grid:
        mean_cov, covs, widths = eval_quantile_setting(X, y, protocols, n_est, depth, lam)
        sweep_results.append({
            "n_estimators": n_est, "max_depth": depth, "reg_lambda": lam,
            "mean_coverage_across_protocols": mean_cov,
            "per_protocol_coverage": dict(zip(protocols.keys(), covs)),
            "per_protocol_width": dict(zip(protocols.keys(), widths)),
        })
        log(f"  n_est={n_est} depth={depth} lambda={lam}: mean_coverage={mean_cov:.3f} "
            f"per_protocol={dict(zip(protocols.keys(), [round(c,3) for c in covs]))}")

    best = min(sweep_results, key=lambda r: abs(r["mean_coverage_across_protocols"] - 0.80))
    log(f"Best setting (closest to 80% target): {best['n_estimators']}/{best['max_depth']}/{best['reg_lambda']} "
        f"-> mean coverage {best['mean_coverage_across_protocols']:.3f}")

    quantile_final = {
        "sweep": sweep_results,
        "chosen_setting": {"n_estimators": best["n_estimators"], "max_depth": best["max_depth"],
                            "reg_lambda": best["reg_lambda"]},
        "chosen_results": {
            name: {"coverage": best["per_protocol_coverage"][name], "mean_width": best["per_protocol_width"][name]}
            for name in protocols
        },
    }

    # --- 2. Gaussian Process comparison ---
    log("=== Gaussian Process Regression (comparison model) ===")
    scaler = StandardScaler()
    gp_results = {}
    xgb_results = {}
    for name, (train_m, test_m) in protocols.items():
        gp_results[name] = fit_eval_gp(X, y, train_m, test_m, scaler)
        log(f"  GP  [{name}]: MAE={gp_results[name]['mae']:.3f} R2={gp_results[name]['r2']} "
            f"coverage80={gp_results[name]['coverage_80']:.3f} width={gp_results[name]['mean_interval_width']:.3f}")
        _, xgb_results[name] = fit_eval_xgb(X, y, train_m, test_m)
        log(f"  XGB [{name}]: MAE={xgb_results[name]['mae']:.3f} R2={xgb_results[name]['r2']}")

    # --- 3. SHAP on the full-data-trained XGBoost point-estimate model ---
    log("=== SHAP feature importance (XGBoost, random-split training fold) ===")
    import shap
    xgb_full, _ = fit_eval_xgb(X, y, random_train, random_test)
    explainer = shap.TreeExplainer(xgb_full)
    shap_values = explainer.shap_values(X[random_train])
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1][:20]
    top_features = [{"feature": feat_names[i], "mean_abs_shap": float(mean_abs_shap[i])} for i in order]
    for tf in top_features[:15]:
        log(f"  {tf['feature']}: {tf['mean_abs_shap']:.4f}")

    np.save(OUT_DIR / "shap_values_random_split.npy", shap_values)
    np.save(OUT_DIR / "shap_X_random_split.npy", X[random_train])
    with open(OUT_DIR / "shap_feature_names.json", "w") as f:
        json.dump(feat_names, f)

    all_results = {
        "quantile_regularization_sweep": quantile_final,
        "gaussian_process": gp_results,
        "xgboost_point_estimate_same_protocols": xgb_results,
        "shap_top20_features": top_features,
        "runtime_seconds": time.time() - t0,
    }
    with open(OUT_DIR / "strengthening_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
