# Builds a clean, ML-ready table from the Devi et al. 2025 (Scientific Data) migration-barrier
# dataset, resolving the mobile cation from the full-cell formula (preferring an alkali metal
# over an alkaline-earth/Al/Zn candidate when both appear, per known crystal chemistry).
import json
import re
from pathlib import Path

import pandas as pd
from pymatgen.core.composition import Composition

DATA_DIR = Path("/glade/work/ajalali/Multivalent-Ion-Conductor-ML-project/data/migration_barriers")
OUT_DIR = Path("/glade/work/ajalali/Multivalent-Ion-Conductor-ML-project/data")

CANDIDATES = ["Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Zn", "Al"]
ALKALI = {"Li", "Na", "K", "Rb", "Cs"}
VALENCE = {"Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
           "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2, "Zn": 2, "Al": 3}


def assign_mobile_cation(formula_str):
    comp = Composition(formula_str.replace(" ", ""))
    present = [el.symbol for el in comp.elements if el.symbol in CANDIDATES]
    if len(present) == 0:
        return None, present
    if len(present) == 1:
        return present[0], present
    alkalis = [e for e in present if e in ALKALI]
    if len(alkalis) == 1:
        return alkalis[0], present
    return None, present  # genuinely ambiguous, flagged not guessed


def main():
    with open(DATA_DIR / "EM-COMPLETE-DATASET.json") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} JSON records")

    rows = []
    for r in records:
        mobile, present = assign_mobile_cation(r["formula"])
        rows.append({
            "jid": r["jid"],
            "formula": r["formula"],
            "crystal_class": r["crystal_class"],
            "sys_name": r["sys_name"],
            "space_group": r["space_group"],
            "XC": r["XC"],
            "target_Em": r["target"],
            "mobile_cation": mobile,
            "candidate_cations_present": ",".join(present),
            "n_candidates": len(present),
        })
    df = pd.DataFrame(rows)
    df["valence"] = df["mobile_cation"].map(VALENCE)
    df["is_multivalent"] = df["valence"] >= 2

    n_unresolved = df["mobile_cation"].isna().sum()
    print(f"Mobile cation resolved for {len(df) - n_unresolved}/{len(df)} entries")
    print("Unresolved (ambiguous, 2+ non-alkali candidates or none):")
    print(df[df["mobile_cation"].isna()][["formula", "candidate_cations_present"]].to_string())

    print()
    print("Mobile-cation counts:")
    print(df["mobile_cation"].value_counts(dropna=False))
    print()
    print("Monovalent vs multivalent counts (resolved only):")
    print(df.dropna(subset=["mobile_cation"])["is_multivalent"].value_counts())

    df.to_parquet(OUT_DIR / "migration_barrier_table.parquet")
    df.to_csv(OUT_DIR / "migration_barrier_table.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'migration_barrier_table.parquet'}")


if __name__ == "__main__":
    main()
