# Builds composition/symmetry-only features from the CAVD ion-transport descriptor dataset
# (Zhang, Deng, Ong et al., Scientific Data 2020); CAVD's own geometric outputs (IND, R_T,
# Acc_*) are excluded as predictors since IND is an exact restatement of Acc_a+Acc_b+Acc_c.
import ast
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core.periodic_table import Element

DATA_DIR = Path("/glade/work/ajalali/Multivalent-Ion-Conductor-ML-project/data")
CATIONS = {"Li": "Li", "Na": "Na", "Mg": "Mg", "Al": "Al"}
CRYSTAL_SYSTEM_RANGES = [
    (1, 2, "triclinic"), (3, 15, "monoclinic"), (16, 74, "orthorhombic"),
    (75, 142, "tetragonal"), (143, 167, "trigonal"), (168, 194, "hexagonal"),
    (195, 230, "cubic"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def crystal_system(sg):
    for lo, hi, name in CRYSTAL_SYSTEM_RANGES:
        if lo <= sg <= hi:
            return name
    return "unknown"


def strip_site_index(label):
    return re.sub(r"[0-9+\-.]+$", "", label)


_elem_prop_cache = {}


def elem_props(symbol):
    if symbol not in _elem_prop_cache:
        try:
            e = Element(symbol)
            _elem_prop_cache[symbol] = {
                "Z": float(e.Z),
                "X": float(e.X) if e.X else np.nan,
                "mass": float(e.atomic_mass),
                "group": float(e.group),
                "row": float(e.row),
            }
        except Exception:
            _elem_prop_cache[symbol] = {"Z": np.nan, "X": np.nan, "mass": np.nan, "group": np.nan, "row": np.nan}
    return _elem_prop_cache[symbol]


def featurize_row(row, cation_symbol):
    radii = ast.literal_eval(row["Ionic_radii"])
    mobile_radii = [v for k, v in radii.items() if strip_site_index(k) == cation_symbol]
    framework = {k: v for k, v in radii.items() if strip_site_index(k) != cation_symbol}

    feat = {}
    feat["mobile_ion_radius_mean"] = float(np.mean(mobile_radii)) if mobile_radii else np.nan
    feat["n_mobile_sites"] = len(mobile_radii)
    feat["n_framework_sites"] = len(framework)

    fw_radii = list(framework.values())
    feat["framework_radius_mean"] = float(np.mean(fw_radii)) if fw_radii else np.nan
    feat["framework_radius_std"] = float(np.std(fw_radii)) if fw_radii else np.nan
    feat["framework_radius_min"] = float(np.min(fw_radii)) if fw_radii else np.nan
    feat["framework_radius_max"] = float(np.max(fw_radii)) if fw_radii else np.nan

    fw_elements = sorted(set(strip_site_index(k) for k in framework.keys()))
    feat["n_distinct_framework_elements"] = len(fw_elements)
    for prop in ("Z", "X", "mass", "group", "row"):
        vals = [elem_props(el)[prop] for el in fw_elements]
        vals = [v for v in vals if not np.isnan(v)]
        feat[f"framework_{prop}_mean"] = float(np.mean(vals)) if vals else np.nan
        feat[f"framework_{prop}_std"] = float(np.std(vals)) if vals else np.nan

    mob_props = elem_props(cation_symbol)
    feat["mobile_Z"] = mob_props["Z"]
    feat["mobile_X"] = mob_props["X"]
    feat["mobile_group"] = mob_props["group"]
    feat["mobile_row"] = mob_props["row"]

    sg = int(row["Symmetry_Int_Tables_number"])
    feat["space_group_number"] = sg
    feat["crystal_system"] = crystal_system(sg)

    feat["has_oxygen"] = int("O" in fw_elements)
    feat["has_halide"] = int(any(el in ("F", "Cl", "Br", "I") for el in fw_elements))
    feat["has_sulfur"] = int("S" in fw_elements)
    feat["has_phosphorus"] = int("P" in fw_elements)

    return feat


def main():
    t0 = time.time()
    all_rows = []
    for sheet, symbol in CATIONS.items():
        df = pd.read_excel(DATA_DIR / "Descriptors.xlsx", sheet_name=sheet)
        log(f"{sheet}: {len(df)} rows loaded")
        for _, row in df.iterrows():
            feat = featurize_row(row, symbol)
            feat["cation"] = sheet
            feat["filename"] = row["Filename"]
            feat["IND"] = int(row["IND"])
            feat["percolates"] = int(row["IND"] > 0)
            all_rows.append(feat)
        log(f"{sheet}: featurized")

    out = pd.DataFrame(all_rows)
    out = pd.get_dummies(out, columns=["crystal_system"], prefix="cs")
    out_path = DATA_DIR / "feature_table.parquet"
    out.to_parquet(out_path, index=False)
    log(f"Saved {out.shape} to {out_path}")
    log(f"IND distribution overall:\n{out['IND'].value_counts()}")
    log(f"Cation counts:\n{out['cation'].value_counts()}")
    log(f"DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
