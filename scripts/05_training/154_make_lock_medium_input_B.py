"""Write the protocol v3 lock that adds the graph B medium-input entry on dataset_v2 (2026-11-03; user approval 2026-11-03 ~16:30
"GPU 실행안 승인. medium input부터": GPU 2 only, seed 0, folds 0-4). New file; protocol_lock_v3_tolpatch_medinput_20261102.json
is not modified. The new lock = that lock verbatim (common block and all entries unchanged, so every existing config still matches)
plus one entry:
  medium_input_B  script 81b_gnn_train_v2.py without --no-medium, graph B (mean_max), gene-disjoint split, seed 0, folds 0-4,
                  30 epochs, common_override no_medium false and out_root result/gnn_20261030_tolpatch/runs_v2_med_B.
Comparator: grid entry B_gene_nomed_meanmax_fixedcos30_s0_tolpatch, paired on fold. Only no_medium differs from it.
Refuses to overwrite.
Usage: python 154_make_lock_medium_input_B.py
"""
import copy, json, os, sys, datetime

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
T = f"{ROOT}/result/gnn_20261030_tolpatch"
src = f"{T}/protocol_lock_v3_tolpatch_medinput_20261102.json"
out = f"{T}/protocol_lock_v3_tolpatch_medinput_B_20261103.json"
if os.path.exists(out):
    sys.exit(f"exists, not overwritten: {out}")
L = json.load(open(src))
assert L["common"]["no_medium"] is True and "medium_input_B" not in L["arms"]
e = copy.deepcopy(L["arms"]["grid"])
e["graphs"] = {"B": e["graphs"]["B"]}
e["splits"] = ["gene"]
e["seeds"] = [0]
e["common_override"] = {"no_medium": False, "out_root": f"{T}/runs_v2_med_B"}
e["note"] = ("grid B gene-disjoint seed 0 with the medium branch fed the cell line's medium (81b without --no-medium); only no_medium "
             "differs from the grid entry. Comparator B_gene_nomed_meanmax_fixedcos30_s0_tolpatch, paired on fold. Seed 0 only: "
             "one GPU (GPU 2) and the 2026-11-05 deadline; B takes one run per 24 GB GPU (peak 15.2 GB).")
L["arms"]["medium_input_B"] = e
L["name"] = "protocol_v3_fixed_budget_cosine_tolpatch_disflag_medinput_medinputB"
L["written"] = f"{datetime.datetime.now():%Y-%m-%d %H:%M} KST, before the first launch of the graph B medium-input entry"
L["authority"] = [src + " (verbatim: common block and all entries; unchanged)",
                  "user approval 2026-11-03: graph B medium input on GPU 2, seed 0, folds 0-4, run first"]
L["controls_statement"] += (" Added 2026-11-03: medium_input_B (B gene-disjoint seed 0, medium branch on), out root runs_v2_med_B; "
                            "the other entries are unchanged and their existing runs match this lock as before.")
json.dump(L, open(out, "w"), indent=1, ensure_ascii=False)
print(out)
print(json.dumps(e, indent=1))
