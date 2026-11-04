"""Final exchange lower bounds per medium assembly for pFBA/MOMA (not for tINIT, which keeps the
scaffold exchanges open; user 2026-10-27).

open at -1000: D10 inorganic set + O2/H2O/H+ (scripts 45/47), medium species that are not carbon
sources (vitamins/cofactors; basis open_not_carbon_source in the cap table), and biotin, lipoic acid,
B12, retinoate, alpha/gamma-tocopherol in every medium.
capped: medium carbon sources by data/medium/uptake_caps/carbon_uptake_caps_dw0.2ng.tsv;
linoleate and linolenate in every medium by their uptakes in the joint minimum of
(linoleate + linolenate) at biomass = mu = ln2/20.6 h in the RPMI context, replacing the concentration-scaled cap that F-12 media would otherwise get
(2.5e-5 for linoleate limited F-12 growth to 0.00035 1/h).
Everything else closed. Reports WT growth of the scaffold per assembly.
Env: crcgem. Output: data/medium/uptake_caps/exchange_lower_bounds_by_assembly.tsv
"""
import json
import numpy as np, pandas as pd, cobra
ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
U = f"{ROOT}/data/medium/uptake_caps"
cobra.Configuration().solver = "gurobi"
m = cobra.io.read_sbml_model(f"{ROOT}/data/scaffold/Human-GEM.xml"); m.solver.configuration.verbosity = 0
prev = json.load(open(f"{ROOT}/result/medium_checks_20261026/universal_exchange_checks.json"))
NAME = {r.id: list(r.metabolites)[0].name for r in m.exchanges}
by_name = {}
for rid, n in NAME.items():
    by_name.setdefault(n, []).append(rid)
ALWAYS = {"MAR09048", "MAR09047", "MAR09079"} | {rid for n in prev["part1_inorganic"]["accepted"] for rid in by_name[n]}
OPEN_ALL = {by_name[n][0] for n in ["biotin", "lipoic acid", "aquacob(III)alamin", "retinoate",
                                     "alpha-tocopherol", "gamma-tocopherol"]}
MU = float(np.log(2) / 20.6)  # ln2 / median NCI-60 colon doubling time (CellMiner), 1/h
asm = pd.read_csv(f"{ROOT}/data/medium/assembled_media_exchange_mM.tsv", sep="\t")
asm = asm[asm.supplement_rule == "final_concentration"]
caps = pd.read_csv(f"{U}/carbon_uptake_caps_dw0.2ng.tsv", sep="\t", comment="#")
EFA_IDS = [by_name["linoleate"][0], by_name["linolenate"][0]]
# joint requirement: RPMI context (carbon caps, everything else of the rule open), biomass fixed to mu,
# minimize linoleate + linolenate uptake; the two caps are that solution's uptakes. Separately
# minimized requirements (each with the other open) halve growth when both are capped (0.01689 1/h).
c0 = caps[caps.assembly_id == "RPMI1640_Gibco11875"]
cap0 = dict(zip(c0.exchange_rxn_id, c0.cap_mmol_gDW_h))
med0 = set(asm[asm.assembly_id == "RPMI1640_Gibco11875"].exchange_rxn_id)
with m:
    for r in m.exchanges:
        if r.id in med0 | ALWAYS | OPEN_ALL | set(EFA_IDS):
            v = cap0.get(r.id, np.nan); r.lower_bound = -1000.0 if v != v else -float(v)
        else:
            r.lower_bound = 0.0
    m.reactions.get_by_id("MAR13082").bounds = (MU, MU)
    m.objective = {m.reactions.get_by_id(x): 1 for x in EFA_IDS}; m.objective_direction = "max"
    sol = m.optimize()
    EFA = {x: max(-float(sol.fluxes[x]), 0.0) for x in EFA_IDS}
print("EFA caps (joint min at mu):", EFA, flush=True)
bio = m.reactions.get_by_id("MAR13082")
rows, growth = [], {}
for a, g in asm.groupby("assembly_id"):
    c = caps[caps.assembly_id == a]
    capd = dict(zip(c.exchange_rxn_id, c.cap_mmol_gDW_h))
    lb = {}
    for rid in set(g.exchange_rxn_id) | ALWAYS | OPEN_ALL:
        v = capd.get(rid, np.nan)
        lb[rid] = (-1000.0, "open") if (v != v) else (-float(v), "carbon_cap")
    for rid, v in EFA.items():
        lb[rid] = (-v, "EFA_requirement_x_mu")
    for rid, (v, why) in lb.items():
        rows.append(dict(assembly_id=a, exchange_rxn_id=rid, name=NAME.get(rid), lower_bound=v, basis=why))
    with m:
        for r in m.exchanges:
            r.lower_bound = lb[r.id][0] if r.id in lb else 0.0
        gv = m.slim_optimize(); growth[a] = 0.0 if gv is None or gv != gv else float(gv)
    print(a, round(growth[a], 5), flush=True)
df = pd.DataFrame(rows)
with open(f"{U}/exchange_lower_bounds_by_assembly.tsv", "w") as fh:
    fh.write(f"# exchange lower bounds per assembly; see script_python/57_build_uptake_bounds.py; "
             f"scaffold WT growth: {json.dumps({k: round(v, 5) for k, v in growth.items()})}\n")
    df.to_csv(fh, sep="\t", index=False)
