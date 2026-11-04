"""Caps for the species opened in every medium without a medium concentration (scope section 5.1;
user ruling 2026-10-27): cap = biomass requirement x reference growth, reference growth
mu = ln2 / median NCI-60 colon doubling time (CellMiner NCI60_CELL_LINE_METADATA.txt, 20.6 h).

Computed per medium assembly in its own context, because the requirement depends on what else
is available (in the all-open scaffold it is 0 for linoleate; first attempt, discarded):
open exchanges = medium components + D10 inorganic set + O2/H2O/H+ + serum set + the 8 species
(as script 47); organic medium species capped by data/medium/uptake_caps/carbon_uptake_caps_dw0.2ng.tsv;
biomass MAR13082 fixed to mu; minimize the uptake of each species with the other 7 open at 1000.
If mu is infeasible under the caps, the maximum growth is reported instead (status).
Env: crcgem (gurobi). Output: data/medium/uptake_caps/biomass_requirement_per_assembly.tsv
"""
import importlib.util, json
import numpy as np, pandas as pd, cobra
ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
DT = [23.8, 31.5, 17.4, 20.6, 19.5, 23.7, 20.4]  # h; COLO205 HCC_2998 HCT_116 HCT_15 HT29 KM12 SW_620
MU = float(np.log(2) / np.median(DT))
SPECIES = ["linoleate", "linolenate", "lipoic acid", "biotin", "aquacob(III)alamin", "retinoate",
           "alpha-tocopherol", "gamma-tocopherol"]
spec = importlib.util.spec_from_file_location("ko", f"{ROOT}/script_python/30_ko_screen_medium.py")
ko = importlib.util.module_from_spec(spec); spec.loader.exec_module(ko)
prev = json.load(open(f"{ROOT}/result/medium_checks_20261026/universal_exchange_checks.json"))
cobra.Configuration().solver = "gurobi"
m = cobra.io.read_sbml_model(ko.HG); m.solver.configuration.verbosity = 0
NAME = {r.id: list(r.metabolites)[0].name for r in m.exchanges}
by_name = {}
for rid, n in NAME.items():
    by_name.setdefault(n, []).append(rid)
ALWAYS = {"MAR09048", "MAR09047", "MAR09079"} | {rid for n in prev["part1_inorganic"]["accepted"] for rid in by_name[n]}
SP = {s: by_name[s][0] for s in SPECIES}
asm = pd.read_csv(f"{ROOT}/data/medium/assembled_media_exchange_mM.tsv", sep="\t")
asm = asm[asm.supplement_rule == "final_concentration"]
MEDIA = {a: set(g.exchange_rxn_id) for a, g in asm.groupby("assembly_id")}
caps = pd.read_csv(f"{ROOT}/data/medium/uptake_caps/carbon_uptake_caps_dw0.2ng.tsv", sep="\t", comment="#")
bio = m.reactions.get_by_id("MAR13082")

rows = []
for a in sorted(MEDIA):
    c = caps[(caps.assembly_id == a) & caps.cap_mmol_gDW_h.notna() & caps.exchange_rxn_id.notna()]
    capd = dict(zip(c.exchange_rxn_id, c.cap_mmol_gDW_h))
    opened = MEDIA[a] | ALWAYS | set(SP.values())
    with m:
        for r in m.exchanges:
            r.lower_bound = (-capd[r.id] if r.id in capd else -1000.0) if r.id in opened else 0.0
        gmax = m.slim_optimize()
        for s, rid in SP.items():
            with m:
                bio.bounds = (MU, MU)
                m.objective = {m.reactions.get_by_id(rid): 1}; m.objective_direction = "max"
                v = m.slim_optimize()
                ok = v is not None and v == v
                rows.append(dict(assembly_id=a, species=s, exchange_rxn_id=rid, growth_max_capped=gmax,
                                 feasible_at_mu=ok, min_uptake_at_mu=(max(-v, 0.0) if ok else np.nan)))
    print(a, "gmax", round(gmax, 5), flush=True)
df = pd.DataFrame(rows)
hdr = (f"# cap = min uptake (mmol/gDW/h) at biomass = mu = ln2/{np.median(DT)} h = {MU:.6f} 1/h, per assembly "
       f"context (see script docstring); script script_python/53_biomass_requirement.py\n")
with open(f"{ROOT}/data/medium/uptake_caps/biomass_requirement_per_assembly.tsv", "w") as fh:
    fh.write(hdr); df.to_csv(fh, sep="\t", index=False)
print(df.pivot(index="species", columns="assembly_id", values="min_uptake_at_mu").round(6).T.to_string())
