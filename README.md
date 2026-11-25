# Cross-valence ML for multivalent-ion solid-state conductors (CAVD percolation + migration-barrier regression)

Scripts used to build features, train models, run robustness/sensitivity analyses, and generate figures/tables for the manuscript. Two independent dataset angles: (1) CAVD composition/symmetry-based percolation-dimensionality classification, (2) DFT-NEB migration-barrier regression (Devi et al. 2025).

## Pipeline order

1. `build_feature_table.py` — composition/symmetry-only feature table from the CAVD descriptor dataset
2. `train_baseline.py` — Random Forest percolation classifier (binary and 4-class), in-distribution and cross-valence protocols
3. `benchmark_cavd.py` — broader percolation-classification model benchmark (SVM, k-NN, Logistic Regression, MLP)
4. `calibration_cavd.py` — calibration curve and Brier score for the percolation classifier
5. `build_migration_dataset.py` — builds the migration-barrier table from the Devi et al. dataset, resolving the mobile cation
6. `train_migration_baseline.py` — XGBoost migration-barrier regression baseline, in-distribution, leave-one-class-out, and cross-valence protocols
7. `benchmark_migration.py` — broader migration-barrier regression model benchmark (Random Forest, SVR, k-NN, Ridge, MLP)
8. `strengthen_migration.py` — quantile-regression uncertainty sweep, Gaussian Process comparison, and SHAP feature importance
9. `additional_robustness_analyses.py` — confound/asymmetry diagnostics, duplicate-grouped splits, mobile-ion crosscheck, descriptor ablation, deeper UQ, SHAP robustness, electrolyte-only subset
10. `physical_descriptors_and_uq_analyses.py` — NEB-displacement mobile-ion identification, Arrhenius threshold framing, CAVD grouped validation, dataset heterogeneity, Ewald/Madelung descriptors, leakage-free UQ, SHAP-vs-physical checks
11. `label_correction_and_bootstrap_checks.py` — corrected mobile-cation labels, CAVD ionic-radius ablation, per-protocol bootstrap CIs, CAVD duplicate-structure check, UQ bootstrap
12. `distribution_and_heterogeneity_checks.py` — target-distribution comparison across grouped/ungrouped splits, migration-barrier dataset heterogeneity by DFT functional type
13. `cross_valence_robustness_checks.py` — NEB-corrected-label reruns of the model benchmark, descriptor ablation, SHAP, and Gaussian-process comparisons; a GGA-only functional-matched subset test; composition-grouped training resampling and within-valence-group splits; a CAVD matched-host subset check; an electronegativity-difference bonding descriptor; and screening-retention rates under cross-valence extrapolation

## Figures

- `make_figures.py` — main manuscript and SI figures

## Requirements

Python 3, pandas, numpy, scikit-learn, scipy, xgboost, matminer, pymatgen, matplotlib, shap.

## Data

Raw data (CAVD `Descriptors.xlsx` from Zhang/Deng/Ong et al. 2020, and the Devi et al. 2025 migration-barrier dataset) are not included here; scripts expect them in a local `data/` directory. See the manuscript's Data Availability statement.
