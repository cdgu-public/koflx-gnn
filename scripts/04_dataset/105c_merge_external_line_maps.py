"""Merge the cell-line map and the requirement-cap overrides of the two 2026-11-03 external sets into one input pair for 64d
(design result/external_plan_20261103/design_external_gi_normal_20261103.md section 6(d); BPH-1 was run in its own root after the
D3 ruling, so the one-dataset build needs the union). Rows are copied, never edited; an empty (comment-only) override file
contributes no rows (as 58:45-48 and 64d). Refuses to overwrite.
Usage: python 105c_merge_external_line_maps.py <out_dir>
"""
import os, sys, time
import pandas as pd

R = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
SETS = [("external_gi_normal_20261103", f"{R}/data/depmap/external_gi_normal_20261103/lines.tsv",
         f"{R}/result/external_gi_normal_20261103/line_requirement_caps/lower_bound_overrides.tsv"),
        ("external_normal_bph1_20261103", f"{R}/data/depmap/external_normal_bph1_20261103/lines.tsv",
         f"{R}/result/external_normal_bph1_20261103/line_requirement_caps/lower_bound_overrides.tsv")]
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
f_lm, f_ov = f"{out}/lines_merged.tsv", f"{out}/lower_bound_overrides_merged.tsv"
assert not os.path.exists(f_lm) and not os.path.exists(f_ov), "outputs exist (no overwrite)"
lms, ovs = [], []
for name, lm_p, ov_p in SETS:
    lm = pd.read_csv(lm_p, sep="\t", comment="#"); lm["source_set"] = name; lms.append(lm)
    try:
        ov = pd.read_csv(ov_p, sep="\t", comment="#")
    except pd.errors.EmptyDataError:
        ov = pd.DataFrame()
    assert ov.empty or set(ov.line) <= set(lm.ModelID), f"{ov_p}: override rows for cell lines outside {lm_p}"
    ovs.append(ov); print(f"{name}: {len(lm)} cell lines, {len(ov)} override rows")
LM = pd.concat(lms, ignore_index=True); OV = pd.concat([o for o in ovs if len(o)], ignore_index=True)
assert LM.ModelID.is_unique, "a cell line occurs in both sets"
assert len(LM) == sum(len(x) for x in lms) and len(OV) == sum(len(x) for x in ovs)
hdr = (f"# source: {'; '.join(f'{a} (lines {b}; overrides {c})' for a, b, c in SETS)}\n"
       f"# rows copied unchanged; comment-only override file = 0 rows; script_python/105c_merge_external_line_maps.py; {time.ctime()}\n")
with open(f_lm, "w") as fh:
    fh.write(hdr + f"# n cell lines {len(LM)}\n"); LM.to_csv(fh, sep="\t", index=False)
with open(f_ov, "w") as fh:
    fh.write(hdr + f"# n override rows {len(OV)}\n"); OV.to_csv(fh, sep="\t", index=False)
print(f"merged: {len(LM)} cell lines, {len(OV)} override rows -> {out}")
