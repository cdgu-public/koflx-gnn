"""Write the protocol v3 lock that adds the medium-input arm on dataset_v2 (2026-11-02; user ruling 2026-11-02 "rerun the medium-input GNN
at low priority" (translated): runs after the main-flow items, on idle GPUs). New file; protocol_lock_v3_tolpatch_disflag_20261101.json is not modified.
The new lock = that lock verbatim (common block and all arms unchanged, so every existing config still matches) plus one arm:
  medium_input  script 81b_gnn_train_v2.py without --no-medium, graph A' (attn_max), gene-disjoint split, seeds 0-2, folds 0-4,
                30 epochs, common_override no_medium false and out_root result/gnn_20261030_tolpatch/runs_v2_med.
Comparator: grid arm A_gene_nomed_fixedcos30_s<seed>_tolpatch, paired on (seed, fold). Scope = the pre-patch medium ablation
(A' gene-disjoint, 3 seeds; result/gnn_20261028/eval_medium_ablation_prepatch_20261102_1457). Refuses to overwrite.
Usage: python 134_make_lock_medium_input.py
"""
import copy, json, os, sys, datetime

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
T = f"{ROOT}/result/gnn_20261030_tolpatch"
src = f"{T}/protocol_lock_v3_tolpatch_disflag_20261101.json"
out = f"{T}/protocol_lock_v3_tolpatch_medinput_20261102.json"
if os.path.exists(out):
    sys.exit(f"exists, not overwritten: {out}")
L = json.load(open(src))
assert L["common"]["no_medium"] is True
arm = copy.deepcopy(L["arms"]["grid"])
arm["graphs"] = {"A": arm["graphs"]["A"]}
arm["splits"] = ["gene"]
arm["common_override"] = {"no_medium": False, "out_root": f"{T}/runs_v2_med"}
arm["note"] = ("grid A' gene-disjoint with the medium branch fed the cell line's medium (81b without --no-medium); only no_medium differs "
               "from the grid arm. Comparator A_gene_nomed_fixedcos30_s<seed>_tolpatch, paired on (seed, fold).")
L["arms"]["medium_input"] = arm
L["name"] = "protocol_v3_fixed_budget_cosine_tolpatch_disflag_medinput"
L["written"] = f"{datetime.datetime.now():%Y-%m-%d %H:%M} KST, before the first launch of the medium-input arm"
L["authority"] = [src + " (verbatim: common block and all arms; unchanged)",
                  "user ruling 2026-11-02: rerun the medium-input GNN at low priority, after the main-flow items, on idle GPUs"]
L["controls_statement"] += (" Added 2026-11-02: medium_input (A' gene-disjoint, medium branch on), out root runs_v2_med; "
                            "the other arms are unchanged and their existing runs match this lock as before.")
json.dump(L, open(out, "w"), indent=1, ensure_ascii=False)
print(out)
print(json.dumps(arm, indent=1))
