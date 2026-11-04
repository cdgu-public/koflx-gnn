"""Write the protocol v3 lock that adds the baseline-plus-flag control arm (2026-11-01; user ruling "바로 시작", design
result/gnn_20261030_tolpatch/design_flag_arm_20261101.md). New file; result/gnn_20261030_tolpatch/protocol_lock_v3_tolpatch_20261030.json
is not modified. The new lock = that lock verbatim (common block, grid and the five control arms unchanged, so every existing
config still matches) plus one arm:
  control_disabled_flag  script 90_gnn_e2e3_v2.py --disabled-flag, graph A' (attn_max), gene-disjoint split, seeds 0-2, folds 0-4,
                         30 epochs, flags as control_wt_only with wt_only false and disabled_flag true,
                         out_root result/gnn_20261030_tolpatch/runs_v2_disflag (common_override; separate root so the 2026-10-30 lock's
                         sweep of runs_v2 is unchanged), runner script_bash/122_run_flag_arm_tolpatch.sh added to the allowed readers.
Refuses to overwrite. Usage: python 121_make_lock_flag_arm.py
"""
import copy, json, os, sys, datetime

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
T = f"{ROOT}/result/gnn_20261030_tolpatch"
src = f"{T}/protocol_lock_v3_tolpatch_20261030.json"
out = f"{T}/protocol_lock_v3_tolpatch_disflag_20261101.json"
if os.path.exists(out):
    sys.exit(f"exists, not overwritten: {out}")
L = json.load(open(src))
wt = L["arms"]["control_wt_only"]
assert wt["flags"]["wt_only"] is True and "disabled_flag" not in wt["flags"]
arm = copy.deepcopy(wt)
arm["flags"]["wt_only"] = False
arm["flags"]["disabled_flag"] = True
arm["common_override"] = {"out_root": f"{T}/runs_v2_disflag"}
arm["note"] = ("baseline A' input plus one node-feature column = the WT-only arm's disabled-reaction flag; node set, edges, KO flux and "
               "currency columns unchanged (90 build(), F_IN + 1). Comparators: grid arm A_gene_nomed_fixedcos30_s<seed>_tolpatch (baseline) and "
               "control_wt_only, paired on (seed, fold) by script_python/118. Design: result/gnn_20261030_tolpatch/design_flag_arm_20261101.md")
L["arms"]["control_disabled_flag"] = arm
L["name"] = "protocol_v3_fixed_budget_cosine_tolpatch_disflag"
L["written"] = f"{datetime.datetime.now():%Y-%m-%d %H:%M} KST, before the first launch of the baseline-plus-flag arm"
L["authority"] = [src + " (verbatim: common block, grid, five control arms; unchanged)",
                  "documentation/status.md entry 2026-10-31 23:55 KST (weakest point: WT-only confounded by the disabled-reaction flag; separating arm = option)",
                  "user ruling 2026-11-01 ('바로 시작'): run the baseline-plus-flag arm",
                  f"{T}/design_flag_arm_20261101.md (design, written before the launch)"]
L["controls_statement"] += (" Added 2026-11-01: control_disabled_flag (baseline input plus the disabled-reaction flag), out root runs_v2_disflag, "
                            "same split, seeds, folds, tag; the other arms are unchanged and their existing runs match this lock as before.")
L["monitoring"]["allowed_readers_of_monitor_or_ckpt"].append("script_bash/122_run_flag_arm_tolpatch.sh")
json.dump(L, open(out, "w"), indent=1, ensure_ascii=False)
print(out)
print(json.dumps(arm, indent=1))
