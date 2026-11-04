"""Carbon-source uptake caps per medium assembly (user ruling 2026-10-26, option a).

Rule (scope_overall_vs_recomb2027.md section 5): per-metabolite consumption rate measured in the
NCI-60 colon lines (Jain et al. 2012 [ref: PMID 22628656]), median across lines, as the uptake cap;
components not measured are scaled to glucose by concentration.

Implementation:
 - CORE values (fmol/cell/h, negative = consumption) of the 7 NCI-60 colon lines (Figure S2 sheet);
   replicate columns averaged per line, then the median across the 7 lines.
 - Organic medium species (formula contains C in Human-GEM v2.0.0) of each assembly
   (assembled_media_exchange_mM.tsv, supplement_rule = final_concentration, the D3 default).
 - measured and median < 0  -> cap = -median (basis "jain_measured")
   measured and median >= 0 -> net release in the colon lines; glucose-scaled as below
                               ("glucose_scaled_net_release"; user ruling 2026-10-27)
   not measured             -> cap = glucose cap * [species]_medium / [glucose]_RPMI, the reference being
                               the glucose of Jain's own medium RPMI-1640 (user ruling 2026-10-27;
                               also covers L-15, which has no glucose) ("glucose_scaled")
 - Species opened in every medium regardless of the basal formulation (scope section 5.1:
   linoleate, linolenate, lipoic acid, biotin, B12, retinoate, tocopherols) are listed with their
   status when the assembly has no concentration for them ("FLAG_no_concentration").
 - mmol/gDW/h with dry weight 0.2 ng/cell (Zielinski et al. 2017 [ref: PMID 28120890], user ruling
   2026-10-27); --dw-ng-per-cell, written into the output name.
Env: crcgem. Output: data/medium/uptake_caps/
"""
import argparse, os, re
import numpy as np, pandas as pd, cobra

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
J = f"{ROOT}/data/medium/sources/jain2012"
OUT = f"{ROOT}/data/medium/uptake_caps"
ap = argparse.ArgumentParser()
ap.add_argument("--dw-ng-per-cell", type=float, default=0.2)
args = ap.parse_args()
os.makedirs(OUT, exist_ok=True)

# Jain name -> Human-GEM metabolite name (only where the names differ; identical names map as is)
SYN = {"thiamine": "thiamin", "niacinamide": "nicotinamide", "niacin": "nicotinate",
       "cis-hydroxyproline/trans-hydroxyproline": "trans-4-hydroxy-L-proline"}
ALWAYS_OPEN = ["linoleate", "linolenate", "lipoic acid", "biotin", "aquacob(III)alamin", "retinoate",
               "alpha-tocopherol", "gamma-tocopherol"]
INORGANIC_C = {"HCO3-", "CO2"}  # opened under D10 (scope section 5), not carbon-source caps

# ---- Jain colon medians ----
core = pd.read_csv(f"{J}/DatabaseS1_CORE_fmol_per_cell_per_h.tsv", sep="\t")
tt = pd.read_csv(f"{J}/DatabaseS1_FigureS2_tumor_type.tsv", sep="\t")
colon = tt.loc[tt["Tumor type"] == "Colon", "Cell line"].tolist()
vals = core.set_index("Metabolite (b)").iloc[:, 2:]
cal = core.set_index("Metabolite (b)")["Calibrated (e)"]
base = pd.Series([re.sub(r"\.\d+$", "", c) for c in vals.columns], index=vals.columns)
per_line = pd.DataFrame({ln: vals.loc[:, base[base == ln].index].astype(float).mean(axis=1) for ln in colon})
assert per_line.shape[1] == 7, per_line.columns
jain = pd.DataFrame({"calibrated": cal, "median_fmol_cell_h": per_line.median(axis=1),
                     "n_lines": per_line.notna().sum(axis=1)})
jain = jain[jain.calibrated == 1]
jain["hg_name"] = [SYN.get(n, n) for n in jain.index]
jain.to_csv(f"{OUT}/jain2012_colon_median_core.tsv", sep="\t")
glc_cap = -jain.loc["glucose", "median_fmol_cell_h"]
print(f"colon lines: {colon}; glucose median CORE {-glc_cap:.4g} fmol/cell/h")

# ---- media ----
a = pd.read_csv(f"{ROOT}/data/medium/assembled_media_exchange_mM.tsv", sep="\t")
a = a[a.supplement_rule == "final_concentration"]
m = cobra.io.read_sbml_model(f"{ROOT}/data/scaffold/Human-GEM.xml")
formula = {x.id: (x.formula or "") for x in m.metabolites}
name2met = {}
for x in m.metabolites:
    if x.compartment == "e":
        name2met.setdefault(x.name, x.id)
has_c = lambda f: bool(re.search(r"C(?![a-z])", f))
jmap = jain.reset_index().set_index("hg_name")
GLC_REF_MM = float(a.loc[(a.assembly_id == "RPMI1640_Gibco11875") & (a.humangem_met_name == "glucose"),
                         "total_species_mM"].iloc[0])  # Jain medium: RPMI-1640 (GIBCO)
print(f"reference glucose (RPMI1640_Gibco11875): {GLC_REF_MM} mM")

rows = []
for asm, g in a.groupby("assembly_id"):
    conc = dict(zip(g.humangem_met_name, g.total_species_mM))
    ex = dict(zip(g.humangem_met_name, g.exchange_rxn_id))
    names = sorted(set(g.humangem_met_name) | set(ALWAYS_OPEN))
    for n in names:
        mid = g.loc[g.humangem_met_name == n, "humangem_met_id"].iloc[0] if n in conc else name2met.get(n)
        f = formula.get(mid, "")
        if not has_c(f) or n in INORGANIC_C:
            continue
        r = dict(assembly_id=asm, humangem_met_name=n, humangem_met_id=mid, exchange_rxn_id=ex.get(n),
                 formula=f, medium_mM=conc.get(n, np.nan), cap_fmol_cell_h=np.nan)
        if n in jmap.index:
            med = float(jmap.loc[n, "median_fmol_cell_h"])
            r["jain_median_fmol_cell_h"] = med
            if med < 0:
                r.update(basis="jain_measured", cap_fmol_cell_h=-med)
            else:
                r.update(basis="glucose_scaled_net_release",
                         cap_fmol_cell_h=glc_cap * conc[n] / GLC_REF_MM if n in conc else np.nan)
        elif n not in conc:
            r.update(basis="FLAG_no_concentration")
        else:
            r.update(basis="glucose_scaled", cap_fmol_cell_h=glc_cap * conc[n] / GLC_REF_MM)
        rows.append(r)
caps = pd.DataFrame(rows)

# Carbon source = the model can catabolize it: max ATP hydrolysis (MAR03964) > 0 with it as the only
# organic input plus the D10 inorganic set (same test as scripts 45/48). Vitamins and other cofactors
# fail this test; they are opened without a cap ("open_not_carbon_source"), because the measured
# vitamin uptake cannot meet the Human-GEM cofactor-pool requirement (pantothenate cap alone limited
# growth to 0.00115 1/h in every medium, 2026-10-27).
import json
prev = json.load(open(f"{ROOT}/result/medium_checks_20261026/universal_exchange_checks.json"))
NAME = {r.id: list(r.metabolites)[0].name for r in m.exchanges}
by_name = {}
for rid, nm in NAME.items():
    by_name.setdefault(nm, []).append(rid)
ALWAYS = {"MAR09048", "MAR09047", "MAR09079"} | {rid for nm in prev["part1_inorganic"]["accepted"] for rid in by_name[nm]}
caps["exchange_rxn_id"] = [e if isinstance(e, str) else (by_name.get(nm, [None])[0])
                           for e, nm in zip(caps.exchange_rxn_id, caps.humangem_met_name)]
m.solver = "gurobi"
atpm = m.reactions.get_by_id("MAR03964")
cat = {}
for rid in sorted(set(caps.exchange_rxn_id.dropna())):
    with m:
        for r in m.exchanges:
            r.lower_bound = -1000.0 if r.id in ALWAYS | {rid} else 0.0
        m.objective = atpm
        v = m.slim_optimize()
        cat[rid] = 0.0 if v is None or v != v else float(v)
caps["sole_organic_max_ATP"] = caps.exchange_rxn_id.map(cat)
nc = caps.sole_organic_max_ATP <= 1e-6
caps.loc[nc, "basis"] = "open_not_carbon_source"
caps.loc[nc, "cap_fmol_cell_h"] = np.nan
tag = "fmol_per_cell_h"
if args.dw_ng_per_cell:
    # fmol/cell/h / (ng/cell * 1e-9 g/ng) * 1e-12 mmol/fmol = mmol/gDW/h
    caps["cap_mmol_gDW_h"] = caps.cap_fmol_cell_h * 1e-12 / (args.dw_ng_per_cell * 1e-9)
    tag = f"dw{args.dw_ng_per_cell:g}ng"
hdr = (f"# Carbon-source uptake caps. Source Jain 2012 Database S1 CORE, colon lines {colon}, median of "
       f"per-line replicate means; glucose cap {glc_cap:.6g} fmol/cell/h; medium = assembled_media_exchange_mM.tsv "
       f"(final_concentration); rule: scope_overall_vs_recomb2027.md section 5 option a; "
       f"dry weight per cell: {args.dw_ng_per_cell if args.dw_ng_per_cell else 'PENDING (no mmol/gDW/h column)'}; "
       f"script script_python/52_jain_uptake_caps.py; built {pd.Timestamp.now():%Y-%m-%d %H:%M}\n")
path = f"{OUT}/carbon_uptake_caps_{tag}.tsv"
with open(path, "w") as fh:
    fh.write(hdr); caps.to_csv(fh, sep="\t", index=False)
print("wrote", path)
print(caps.groupby("basis").size().to_string())
print(caps[caps.basis.str.startswith("FLAG")].groupby(["basis", "humangem_met_name"]).assembly_id.count().to_string())
