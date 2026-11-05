"""Write the protocol v3 lock for the GNN on one common-medium dataset (2026-10-31; design
result/common_medium_20261030/gnn_design_20261031.md section 2 (4); user ruling decision log 2026-10-31 17:57 entry, item 4).
Copy of result/gnn_20261030_tolpatch/protocol_lock_v3_tolpatch_20261030.json with common.dataset / out_root / tag of the medium,
the dataset md5, grid arm only (graph A' only unless --with-b), control arms removed, runner 116 added to the allowed readers.
Refuses to overwrite an existing lock. Usage: 117_make_lock_common_medium.py <DMEMF12_Gibco11320|RPMI1640_Gibco11875> [--with-b]
--with-b (2026-11-01, user ruling: graph B on both common media) writes protocol_lock_v3_common_<M>_AB_20261101.json (graphs A and B).
"""
import hashlib, json, os, sys, datetime

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
M = sys.argv[1]; assert M in ("DMEMF12_Gibco11320", "RPMI1640_Gibco11875"), M
T = f"{ROOT}/result/gnn_fixed_{M}_20261030_tolpatch"; DS = f"{T}/dataset"
out = f"{T}/protocol_lock_v3_common_{M}_AB_20261101.json" if "--with-b" in sys.argv else f"{T}/protocol_lock_v3_common_{M}_20261031.json"
if os.path.exists(out):
    sys.exit(f"exists, not overwritten: {out}")
src = f"{ROOT}/result/gnn_20261030_tolpatch/protocol_lock_v3_tolpatch_20261030.json"
L = json.load(open(src))
md5 = hashlib.md5(open(f"{DS}/samples.tsv", "rb").read()).hexdigest()
L["name"] = f"protocol_v3_fixed_budget_cosine_common_{M}"
L["written"] = f"{datetime.datetime.now():%Y-%m-%d %H:%M} KST, before the first launch on the common-medium dataset {M}"
L["authority"] = [src + " (training rule unchanged)", "the project decision log entry 2026-10-31 17:57 KST (user: GNN on both common-medium datasets, RUN)",
                  "result/common_medium_20261030/gnn_design_20261031.md (design)"]
L["statement"] = (f"Same training rule as {os.path.basename(src)}, applied to {DS} (MOMA under a common medium, {M}; script 64d). "
                  "Gene-disjoint folds are the default group_folds of 81b on this dataset (not pinned to the own-medium folds; design section 1); "
                  "cell-line-disjoint folds are identical to the own-medium grid. Grid arm only.")
L["dataset"] = DS
L["dataset_md5_samples_tsv"] = f"{md5} ({DS}/samples.tsv)"
for k in ("dataset_v2_note", "falsifier_dry_test", "controls_statement"):
    L.pop(k, None)
L["common"].update(dataset=DS, out_root=f"{T}/runs_v2", tag=f"_cm{M}")
grid = L["arms"]["grid"]
if "--with-b" not in sys.argv:
    grid["graphs"] = {"A": grid["graphs"]["A"]}
L["arms"] = {"grid": grid}
L["monitoring"]["allowed_readers_of_monitor_or_ckpt"].append("script_bash/116_run_protocol_v3_common_medium.sh")
json.dump(L, open(out, "w"), indent=1, ensure_ascii=False)
print(out, md5, sorted(grid["graphs"]))
