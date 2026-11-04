"""Minimal additions that let each non-growing medium assembly support WT growth.

45_universal_exchange_checks.py found zero WT growth on 11 of 13 assemblies and no single
added exchange that restores it. Here the candidate pool is every exchange of the F-12
assembly (which grows) plus Human-GEM's Ham's task list. For each failing assembly the
pool is added, growth is confirmed, then pool members are removed one at a time (fixed
order: pool sorted by exchange id, so the result is deterministic) while growth stays
positive. The survivors form a minimal (irreducible, not necessarily minimum) rescue set.
Rules match script 45 (D10 open set with sulfite rejected; B12 and the serum set open).
"""
import importlib.util
import json

import cobra
import pandas as pd

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
OUT = f"{ROOT}/result/medium_checks_20261026"
TOL = 1e-6
spec = importlib.util.spec_from_file_location("ko", f"{ROOT}/script_python/30_ko_screen_medium.py")
ko = importlib.util.module_from_spec(spec); spec.loader.exec_module(ko)

prev = json.load(open(f"{OUT}/universal_exchange_checks.json"))
cobra.Configuration().solver = "gurobi"
m = cobra.io.read_sbml_model(ko.HG)
m.solver.configuration.verbosity = 0
NAME = {r.id: list(r.metabolites)[0].name for r in m.exchanges}
by_name = {}
for rid, n in NAME.items():
    by_name.setdefault(n, []).append(rid)

ALWAYS = {"MAR09048", "MAR09047", "MAR09079"} | {
    rid for n in prev["part1_inorganic"]["accepted"] for rid in by_name[n]}
UNIVERSAL = {"MAR09269", "MAR09036", "MAR09151", "MAR09153"} | set(by_name.get("retinoate", []))

asm = pd.read_csv(f"{ROOT}/data/medium/assembled_media_exchange_mM.tsv", sep="\t")
asm = asm[asm.supplement_rule.isin(["final_concentration"]) | asm.supplement_rule.isna()]
MEDIA = {a: set(g.exchange_rxn_id) for a, g in asm.groupby("assembly_id")}
POOL = MEDIA["F12_Gibco11765"] | ko.ham_exchanges(m)


def growth(ids):
    for r in m.exchanges:
        r.lower_bound = -1000.0 if r.id in ids else 0.0
    v = m.slim_optimize()
    return 0.0 if v is None or v != v else float(v)


res = {}
for a, d in prev["part3_media"].items():
    if d["wt_growth"] > TOL:
        continue
    base = MEDIA[a] | ALWAYS | UNIVERSAL
    add = sorted(POOL - base)
    with m:
        g_full = growth(base | set(add))
        keep = list(add)
        if g_full > TOL:
            for rid in sorted(add):
                trial = [k for k in keep if k != rid]
                if growth(base | set(trial)) > TOL:
                    keep = trial
        g_min = growth(base | set(keep))
    res[a] = {"n_lines": d["n_lines"], "growth_with_pool": g_full, "growth_with_minimal_set": g_min,
              "minimal_set": [f"{NAME[k]} ({k})" for k in keep]}
    print(a, round(g_full, 4), round(g_min, 4), res[a]["minimal_set"], flush=True)

json.dump(res, open(f"{OUT}/minimal_rescue_sets.json", "w"), indent=1)
