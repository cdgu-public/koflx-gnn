"""141 reproduction error (repro_rel_diff = |sum dv^2 from stored float32 fluxes - moma_objective| / moma_objective) by quintile of
moma_objective, per model (2026-11-03, POC normals 142 check). Diagnostic only, no threshold.
Usage: python 159_repro_error_by_objective.py <141 set dir> [model ...]
"""
import sys
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1])
for m in sys.argv[2:] or sorted(p.name for p in root.iterdir() if p.is_dir()):
    o = pd.read_csv(root / m / "overall_distance.tsv", sep="\t", comment="#")
    o = o[o.status == "optimal"].copy()
    o["objective_quintile"] = pd.qcut(o.moma_objective, 5)
    print(f"== {m}")
    print(o.groupby("objective_quintile", observed=True).repro_rel_diff.agg(["median", "max", "count"]).to_string())
    big = o[o.repro_rel_diff > 0.05]
    print(f"rows > 0.05: {len(big)}; by gene {big.gene.value_counts().head(6).to_dict()}")
