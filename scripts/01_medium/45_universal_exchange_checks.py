"""Checks every exchange that is opened in all media, and WT growth per medium assembly.

Rule under test (D10 as ruled 2026-10-26): each line's medium components are open;
O2, H2O, H+, HCO3-, CO2, NH3/NH4+ and the non-toxic inorganic salts and metals are open
in every medium; vitamin B12 (aquacob(III)alamin, D8) and the FBS option B serum set
(linolenate, retinoate, alpha-/gamma-tocopherol) are open in every medium. Uptake bounds
are binary (-1000) here; the uptake caps (option a) come later.

Part 1. Inorganic candidates: an inorganic species is opened everywhere only if, with no
carbon source, its uptake cannot drive ATP hydrolysis (MAR03964), alone or together with
the other accepted species (an energy source without carbon is a modelling artefact).
Part 2. Organic species opened everywhere: (i) can the model make it (max secretion with
uptake closed, on the RPMI assembly)? (ii) can it serve as carbon or energy source alone
(max ATP and max biomass with only it and the accepted inorganic set)? (iii) per medium
assembly: growth with it open vs closed, and its maximum uptake at optimal growth (FVA)
against what the biomass reaction itself consumes.
Part 3. WT growth for each of the 13 medium assemblies used by the 59 lines; for any with
zero growth, every single exchange whose addition restores growth.
"""
import json
import re

import cobra
import pandas as pd
from cobra.flux_analysis import flux_variability_analysis

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
OUT = f"{ROOT}/result/medium_checks_20261026"
TOL = 1e-6

cobra.Configuration().solver = "gurobi"
m = cobra.io.read_sbml_model(f"{ROOT}/data/scaffold/Human-GEM.xml")
m.solver.configuration.verbosity = 0
EX = {r.id: r for r in m.exchanges}
NAME = {r.id: list(r.metabolites)[0].name for r in m.exchanges}
ATPM = m.reactions.get_by_id("MAR03964")
BIOMASS = m.reactions.get_by_id("MAR13082")

ALWAYS_BASE = ["MAR09048", "MAR09047", "MAR09079"]           # O2, H2O, H+
INORG_CAND = {  # carbon-free exchanges that are salts/metals or the agreed inorganic set
    "MAR09078": "HCO3-", "MAR09058": "CO2", "MAR09073": "NH3", "MAR11420": "NH4+",
    "MAR09082": "Ca2+", "MAR13073": "Cu2+", "MAR09076": "Fe2+", "MAR09096": "Fe3+",
    "MAR09081": "K+", "MAR13072": "Mg2+", "MAR09077": "Na+", "MAR09150": "chloride",
    "MAR09072": "Pi", "MAR11405": "PPi", "MAR09074": "sulfate", "MAR09088": "sulfite",
    "MAR09432": "thiosulfate", "MAR09148": "iodide", "MAR09080": "zinc", "MAR09386": "hydroxide",
}
# Not candidates: H2S, H2O2, NO, O2- (ruled closed), nitrite, hypochlorite, nitryl-chloride,
# peroxynitrite (reactive, not medium components), Li+ (not a medium component),
# selenate (open only if in the medium, D7), lipoproteins and placeholders (not inorganic).
ORGANIC_UNIVERSAL = {"MAR09269": "aquacob(III)alamin (vitamin B12)",
                     "MAR09036": "linolenate", "MAR09151": "alpha-tocopherol",
                     "MAR09153": "gamma-tocopherol"}


def find_ex(name):
    return [rid for rid, n in NAME.items() if n == name]


ret = find_ex("retinoate")
ORGANIC_UNIVERSAL.update({rid: "retinoate" for rid in ret})


def open_only(ids, cap=1000.0):
    for r in m.exchanges:
        r.lower_bound = -cap if r.id in ids else 0.0


def opt(obj=None):
    if obj is not None:
        m.objective = obj
    v = m.slim_optimize()
    return 0.0 if v is None or v != v else float(v)


res = {"part1_inorganic": {}, "part2_organic": {}, "part3_media": {}}

# ---------- Part 1
base = set(ALWAYS_BASE)
with m:
    open_only(base)
    res["part1_inorganic"]["baseline_atp"] = opt(ATPM)
    for rid, nm in INORG_CAND.items():
        open_only(base | {rid})
        res["part1_inorganic"][f"{nm} ({rid})"] = opt(ATPM)
accepted = {rid for rid, nm in INORG_CAND.items()
            if res["part1_inorganic"][f"{nm} ({rid})"] <= TOL}
with m:
    open_only(base | accepted)
    res["part1_inorganic"]["ALL_ACCEPTED_TOGETHER"] = opt(ATPM)
res["part1_inorganic"]["accepted"] = sorted(NAME[r] for r in accepted)
res["part1_inorganic"]["rejected"] = sorted(INORG_CAND[r] for r in set(INORG_CAND) - accepted)
ALWAYS = base | accepted
print(json.dumps(res["part1_inorganic"], indent=1), flush=True)

# ---------- media assemblies (D3 default: final_concentration)
asm = pd.read_csv(f"{ROOT}/data/medium/assembled_media_exchange_mM.tsv", sep="\t")
lm = pd.read_csv(f"{ROOT}/data/medium/line_medium_map.tsv", sep="\t")
used = sorted(lm.assembly_id.unique())
asm = asm[asm.supplement_rule.isin(["final_concentration"]) | asm.supplement_rule.isna()]
MEDIA = {a: set(asm.loc[asm.assembly_id == a, "exchange_rxn_id"]) for a in used}
missing = [a for a in used if not MEDIA[a]]
assert not missing, f"no exchange rows for assemblies {missing}"


def medium_set(a, organic=True):
    # selenate stays medium-only (D7): it is opened only through MEDIA[a]
    s = MEDIA[a] | ALWAYS
    return s | set(ORGANIC_UNIVERSAL) if organic else s


# ---------- Part 2
rpmi = next(a for a in used if a.startswith("RPMI1640_Gibco11875"))
for rid, nm in ORGANIC_UNIVERSAL.items():
    d = {}
    with m:
        open_only(medium_set(rpmi) - {rid})
        EX[rid].upper_bound = 1000.0
        d["synthesis_max_secretion_RPMI"] = opt(EX[rid])          # export = positive flux
    with m:
        open_only(ALWAYS | {rid})
        d["sole_organic_max_ATP"] = opt(ATPM)
        d["sole_organic_max_biomass"] = opt(BIOMASS)
    met_e = list(EX[rid].metabolites)[0]
    per = {}
    for a in used:
        with m:
            open_only(medium_set(a))
            g_open = opt(BIOMASS)
            fva = flux_variability_analysis(m, [EX[rid]], fraction_of_optimum=1.0) if g_open > TOL else None
            open_only(medium_set(a) - {rid})
            g_closed = opt(BIOMASS)
        per[a] = {"growth_open": g_open, "growth_closed": g_closed,
                  "max_uptake_at_optimum": (None if fva is None else float(-fva.loc[rid, "minimum"]))}
    d["per_medium"] = per
    res["part2_organic"][f"{nm} ({rid})"] = d
    print(nm, json.dumps({k: v for k, v in d.items() if k != "per_medium"}), flush=True)

# biomass demand of cobalt cofactor, for comparison with B12 uptake
res["part2_organic"]["note_cofactor_pool_cobamide_coeff"] = float(
    -m.reactions.get_by_id("MAR10065").metabolites[m.metabolites.get_by_id("MAM01600m")])

# ---------- Part 3
for a in used:
    with m:
        open_only(medium_set(a))
        g = opt(BIOMASS)
        rescue = []
        if g <= TOL:
            for rid in EX:
                if rid in medium_set(a):
                    continue
                EX[rid].lower_bound = -1000.0
                if opt() > TOL:
                    rescue.append(f"{NAME[rid]} ({rid})")
                EX[rid].lower_bound = 0.0
    res["part3_media"][a] = {"wt_growth": g, "n_lines": int((lm.assembly_id == a).sum()),
                             "single_rescues": rescue}
    print(a, round(g, 4), (lm.assembly_id == a).sum(), rescue[:15], flush=True)

json.dump(res, open(f"{OUT}/universal_exchange_checks.json", "w"), indent=1, default=str)
