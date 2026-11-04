"""Validity checks on 141 outputs for given model dirs (computation only, no expected result).
Usage: python 148_check_fluxome_distance.py <out_dir_of_set> [model ...]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

root = Path(sys.argv[1])
models = sys.argv[2:] or sorted(p.name for p in root.iterdir() if p.is_dir())
for m in models:
    ov = pd.read_csv(root / m / "overall_distance.tsv", sep="\t", comment="#")
    pw = pd.read_csv(root / m / "pathway_distance.tsv.gz", sep="\t", comment="#")
    sz = pd.read_csv(root / m / "group_size.tsv", sep="\t", comment="#")
    o = ov[ov.status == "optimal"]
    print(f"== {m}: knockouts {len(ov)} (optimal {len(o)}); negative moma_objective {int((o.moma_objective < 0).sum())} "
          f"(min {o.moma_objective.min():.3g})")
    rd = o.repro_rel_diff[o.moma_objective > 1e-12]
    print(f"  reproduction |sum dv^2 - obj|/obj (obj > 1e-12, n {len(rd)}): median {rd.median():.3g}, max {rd.max():.3g}; "
          f"d_all vs d_all_parquet max abs diff {np.nanmax(np.abs(o.d_all - o.d_all_parquet)):.3g}")
    print(f"  d_all median {o.d_all.median():.3g}; d_all_t median {o.d_all_t.median():.3g}; n_active_ref {o.n_active_ref.iloc[0]}; "
          f"n_changed median {o.n_changed.median():.0f}")
    for route, s in sz.groupby("route"):
        ref = s[s.reference_id == s.reference_id.iloc[0]]
        km = ref[ref.group_kind == "kegg_map"]
        nm = ref[ref.group_kind != "kegg_map"]
        p = pw[(pw.route == route) & (pw.group_kind == "kegg_map")]
        print(f"  [{route}] maps in model {int((km.n_in_model > 0).sum())}/{len(km)}; distinct-reaction coverage needs mapping file; "
              f"reactions in non-KEGG groups {int(nm.n_in_model.sum())}; maps with no active reference reaction "
              f"{int(((km.n_in_model > 0) & (km.n_active_ref == 0)).sum())}; d_rel_ref NA {p.d_rel_ref.isna().mean():.3f}, "
              f"d_sym NA {p.d_sym.isna().mean():.3f}; d_sym in [0,1] {bool(p.d_sym.dropna().between(0, 1 + 1e-9).all())}")
    assert len(pw) == len(o) * len(sz[(sz.n_in_model > 0)]) // sz.reference_id.nunique(), "row count mismatch"
print("checks done")
