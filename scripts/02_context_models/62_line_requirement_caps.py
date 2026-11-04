"""Per-line uptake caps at mu on the context model (user ruling 2026-10-27 20:58, after QC 61 found WT
growth below mu in 30/53 lines: caps sized on the scaffold limit context models that lack the
synthesis route).

Rule. Eligible caps = those not measured by Jain 2012: basis glucose_scaled / glucose_scaled_net_release
in data/medium/uptake_caps/carbon_uptake_caps_dw0.2ng.tsv, and EFA_requirement_x_mu (linoleate,
linolenate) in exchange_lower_bounds_by_assembly.tsv. Jain-measured caps are kept. Per line, on its
task-repaired tINIT model with its assembly's bounds B0:
 1. FBA growth >= mu(1-1e-6): no change.
 2. Otherwise open the eligible caps (B1). If growth on B1 is still < mu, gap-fill from the scaffold
    with B1 bounds (same LP as script 58: biomass = mu, minimize total flux, weight 1 on context
    reactions and 1000 on absent ones) and add the absent reactions carrying flux.
 3. Biomass fixed to mu on B1, minimize the summed uptake (reverse variables) of the eligible
    exchanges; new cap = max(B0 cap, that uptake) for each eligible exchange.
 4. Check FBA growth >= mu(1-1e-6) with the new bounds.
mu = ln2/20.6 h (CellMiner NCI-60 colon median doubling time).
Usage: python 62_line_requirement_caps.py [n_proc] [lines...]   Env: crcgem.
Output: data/medium/uptake_caps/line_requirement_caps/{lower_bound_overrides.tsv, gapfill.json, lines.tsv}
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd, cobra

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
U = f"{ROOT}/data/medium/uptake_caps"
# out-of-bag lines (script 76/77, 2026-10-28): env LRC_OUT, LRC_MODELS, LRC_LINE_MAP (ModelID, assembly_id)
# redirect the output, the models and the line -> assembly map; defaults are the CRC files
OUT = os.environ.get("LRC_OUT", f"{U}/line_requirement_caps")
MODELS = os.environ.get("LRC_MODELS", f"{ROOT}/result/tinit_localT2_20261027/models")
MU = float(np.log(2) / 20.6)
TOL = 1e-6
lm = pd.read_csv(os.environ.get("LRC_LINE_MAP", f"{ROOT}/data/medium/line_medium_map.tsv"), sep="\t").set_index("ModelID")
LB = pd.read_csv(f"{U}/exchange_lower_bounds_by_assembly.tsv", sep="\t", comment="#")
CAP = pd.read_csv(f"{U}/carbon_uptake_caps_dw0.2ng.tsv", sep="\t", comment="#")
SCALED = set(zip(CAP.assembly_id[CAP.basis.str.startswith("glucose_scaled")],
                 CAP.exchange_rxn_id[CAP.basis.str.startswith("glucose_scaled")]))


def growth(m):
    v = m.slim_optimize()
    return 0.0 if v is None or v != v else float(v)


def run(line):
    t0 = time.time()
    a = lm.loc[line, "assembly_id"]
    tb = LB[LB.assembly_id == a].set_index("exchange_rxn_id")
    b0 = tb.lower_bound.to_dict()
    elig = [r for r in tb.index if (a, r) in SCALED or tb.basis[r] == "EFA_requirement_x_mu"]
    m = cobra.io.read_sbml_model(f"{MODELS}/{line}/{line}.repaired.xml")
    m.solver = "gurobi"; m.solver.configuration.verbosity = 0; m.solver.problem.Params.Threads = 2
    for r in m.exchanges:
        r.lower_bound = b0.get(r.id, 0.0)
    m.objective = "MAR13082"
    g0 = growth(m)
    row = dict(line=line, assembly=a, growth_B0=g0, n_eligible=len(elig))
    if g0 >= MU * (1 - TOL):
        row.update(changed=False, growth_new=g0, gapfill=[], seconds=round(time.time() - t0))
        return row, []
    b1 = dict(b0)
    for r in elig:
        b1[r] = -1000.0
    for r in m.exchanges:
        r.lower_bound = b1.get(r.id, 0.0)
    g1 = growth(m)
    add = []
    if g1 < MU * (1 - TOL):
        scaf = cobra.io.read_sbml_model(f"{ROOT}/data/scaffold/Human-GEM.xml")
        scaf.solver = "gurobi"; scaf.solver.configuration.verbosity = 0; scaf.solver.problem.Params.Threads = 2
        for r in scaf.exchanges:
            r.lower_bound = b1.get(r.id, 0.0)
        present = {r.id for r in m.reactions}
        scaf.reactions.get_by_id("MAR13082").bounds = (MU, MU)
        scaf.objective = scaf.problem.Objective(0, direction="min")
        scaf.objective.set_linear_coefficients(
            {v: (1.0 if r.id in present else 1000.0)
             for r in scaf.reactions for v in (r.forward_variable, r.reverse_variable)})
        gs = scaf.optimize()
        add = [r.id for r in scaf.reactions if r.id not in present and abs(gs.fluxes[r.id]) > 1e-9] \
            if gs.status == "optimal" else []
        m.add_reactions([scaf.reactions.get_by_id(x).copy() for x in add])
        for r in m.exchanges:
            r.lower_bound = b1.get(r.id, 0.0)
        m.objective = "MAR13082"
    g1b = growth(m)
    over = []
    with m:
        m.reactions.get_by_id("MAR13082").bounds = (MU, MU)
        ex = [m.reactions.get_by_id(r) for r in elig if r in m.reactions]
        m.objective = m.problem.Objective(0, direction="min")
        m.objective.set_linear_coefficients({r.reverse_variable: 1.0 for r in ex})
        s = m.optimize()
        req = {r.id: max(-float(s.fluxes[r.id]), 0.0) for r in ex} if s.status == "optimal" else {}
    new = dict(b0)
    for r, u in req.items():
        if -u < b0[r]:
            new[r] = -u
            over.append(dict(line=line, assembly=a, exchange_rxn_id=r, name=tb.name[r], basis=tb.basis[r],
                             lower_bound_B0=b0[r], lower_bound_new=-u))
    for r in m.exchanges:
        r.lower_bound = new.get(r.id, 0.0)
    m.objective = "MAR13082"
    gn = growth(m)
    row.update(changed=True, growth_B1=g1, gapfill=add, growth_B1_after_gapfill=g1b,
               requirement_status=s.status, n_caps_raised=len(over), growth_new=gn,
               reaches_mu=gn >= MU * (1 - TOL), seconds=round(time.time() - t0))
    return row, over


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    lines = sys.argv[2:] or sorted(lm.index)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p:
        res = p.map(run, lines)
    rows = pd.DataFrame([r for r, _ in res])
    over = pd.DataFrame([o for _, ov in res for o in ov])
    hdr = (f"# per-line requirement caps at mu={MU:.6f}; script_python/62_line_requirement_caps.py; "
           f"user ruling 2026-10-27; {len(lines)} lines; {time.ctime()}\n")
    for name, df in (("lines.tsv", rows.drop(columns="gapfill")), ("lower_bound_overrides.tsv", over)):
        with open(f"{OUT}/{name}", "w") as fh:
            fh.write(hdr); df.to_csv(fh, sep="\t", index=False)
    json.dump({r["line"]: r["gapfill"] for r in rows.to_dict("records")}, open(f"{OUT}/gapfill.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(rows.drop(columns="gapfill").to_string(index=False))
    print(over.groupby("name").size().to_string() if len(over) else "no overrides")
