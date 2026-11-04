"""pFBA reference + single-gene MOMA fluxomes for one DepMap line on its task-repaired tINIT model.

Model: result/tinit_localT2_20261027/models/<line>/<line>.repaired.xml (scripts 22 + 56).
Medium: the line's assembly (data/medium/line_medium_map.tsv) with exchange lower bounds from
data/medium/uptake_caps/exchange_lower_bounds_by_assembly.tsv (script 57); exchanges not listed
are closed for uptake; upper bounds unchanged.
Reference: pFBA at the FBA optimum (config crc_depmap_celllines.yaml: proliferative_reference pfba).
Screen: quadratic MOMA (cobra add_moma built once; see comment in the body; GPR-aware deletion) over every gene whose
knockout disables >= 1 reaction of the context model. Full flux matrix kept.
Usage: python 58_moma_line.py <ModelID> [--threads 2]   Env: crcgem.
Output: result/moma_20261027/<line>/{summary.tsv, flux_matrix.parquet, reference.parquet, meta.json}
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, cobra

ROOT = Path("/data/workspace/cdgu/Winery/seeding/CRC_GEM")
sys.path.insert(0, str(ROOT / "module"))
from crcgem import moma, reference_flux  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("line")
ap.add_argument("--threads", type=int, default=2)
# out-of-bag lines (script 76, 2026-10-28): own model and output roots, assembly given explicitly
ap.add_argument("--models-root", default="result/tinit_localT2_20261027/models")
ap.add_argument("--out-root", default="result/moma_20261027")
ap.add_argument("--assembly")
ap.add_argument("--caps-dir", default="data/medium/uptake_caps/line_requirement_caps")
args = ap.parse_args()
out = ROOT / args.out_root / args.line
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()

lm = pd.read_csv(ROOT / "data/medium/line_medium_map.tsv", sep="\t").set_index("ModelID")
assembly = args.assembly or lm.loc[args.line, "assembly_id"]
lb = pd.read_csv(ROOT / "data/medium/uptake_caps/exchange_lower_bounds_by_assembly.tsv", sep="\t", comment="#")
lb = lb[lb.assembly_id == assembly].set_index("exchange_rxn_id").lower_bound.to_dict()

m = cobra.io.read_sbml_model(str(ROOT / args.models_root / args.line / f"{args.line}.repaired.xml"))
m.solver = "gurobi"
m.solver.configuration.verbosity = 0
m.solver.problem.Params.Threads = args.threads
# per-line requirement caps at mu and their gap-fill (script 62, user ruling 2026-10-27 20:58)
LRC = ROOT / args.caps_dir
try:
    ov = pd.read_csv(LRC / "lower_bound_overrides.tsv", sep="\t", comment="#")
except pd.errors.EmptyDataError:   # 2026-11-03: 62 writes only the comment line when no line needs an override (BPH-1)
    ov = pd.DataFrame(columns=["line", "exchange_rxn_id", "lower_bound_new"])
ov = ov[ov.line == args.line]
lb.update(dict(zip(ov.exchange_rxn_id, ov.lower_bound_new)))
gf62 = json.load(open(LRC / "gapfill.json")).get(args.line, [])
if gf62:
    scaf62 = cobra.io.read_sbml_model(str(ROOT / "data/scaffold/Human-GEM.xml"))
    m.add_reactions([scaf62.reactions.get_by_id(x).copy() for x in gf62 if x not in m.reactions])
for r in m.exchanges:
    r.lower_bound = lb.get(r.id, 0.0)
bio = m.reactions.get_by_id("MAR13082")
m.objective = bio
wt = m.slim_optimize()
wt = 0.0 if wt is None or wt != wt else float(wt)
meta = dict(line=args.line, assembly=assembly, n_reactions=len(m.reactions), n_genes=len(m.genes), wt_growth=wt,
            caps_rule="per-line requirement at mu (script 62)",
            caps_raised=dict(zip(ov.exchange_rxn_id, ov.lower_bound_new)), gapfill_script62=gf62)
if wt <= 1e-9:
    # Gap-fill for growth in the line's own medium (config extraction.gap_filling: true): on the
    # scaffold with the same exchange bounds, biomass fixed to mu (ln2/20.6 h), minimize total
    # absolute flux with weight 1 on context reactions and 1000 on reactions absent from the
    # context model; the absent reactions carrying flux are added from the scaffold.
    scaf = cobra.io.read_sbml_model(str(ROOT / "data/scaffold/Human-GEM.xml"))
    scaf.solver = "gurobi"; scaf.solver.configuration.verbosity = 0
    for r in scaf.exchanges:
        r.lower_bound = lb.get(r.id, 0.0)
    present = {r.id for r in m.reactions}
    with scaf:
        scaf.reactions.get_by_id("MAR13082").bounds = (np.log(2) / 20.6, np.log(2) / 20.6)
        # same objective set by coefficients: building it as a symbolic sum took ~6 h on the
        # scaffold (2026-10-27: 1,000 terms 94 s, 2,000 terms 462 s); the LP solves in < 1 s
        scaf.objective = scaf.problem.Objective(0, direction="min")
        scaf.objective.set_linear_coefficients(
            {v: (1.0 if r.id in present else 1000.0)
             for r in scaf.reactions for v in (r.forward_variable, r.reverse_variable)})
        gsol = scaf.optimize()
        add = [r.id for r in scaf.reactions if r.id not in present and abs(gsol.fluxes[r.id]) > 1e-9] \
            if gsol.status == "optimal" else []
    m.add_reactions([scaf.reactions.get_by_id(x).copy() for x in add])
    for r in m.exchanges:
        r.lower_bound = lb.get(r.id, 0.0)
    m.objective = m.reactions.get_by_id("MAR13082")
    wt = m.slim_optimize(); wt = 0.0 if wt is None or wt != wt else float(wt)
    meta.update(gapfill_added=add, wt_growth_after_gapfill=wt, n_reactions=len(m.reactions))
    if wt <= 1e-9:
        meta["status"] = "WT growth 0 in its medium after gap-fill; no screen"
        json.dump(meta, open(out / "meta.json", "w"), indent=1); print(meta); raise SystemExit(0)

try:
    ref = reference_flux.compute(m, "pfba")
except cobra.exceptions.Infeasible:
    # ACH-000950 (2026-10-27): pFBA with biomass >= wt*(1-1e-9) was infeasible after the diagnose
    # FVA (same pFBA on a fresh model was feasible). Retry once from a reset solver with biomass
    # capped at wt*(1-1e-6); recorded in meta.json.
    m.solver.problem.reset()
    bio.upper_bound = wt * (1 - 1e-6)
    ref = reference_flux.compute(m, "pfba", diagnose=False)
    meta.update(reference_fallback="biomass upper bound wt*(1-1e-6) after Infeasible pFBA")
genes = []
for g in m.genes:
    if any(r.gpr.eval({g.id}) is False for r in g.reactions):
        genes.append(g.id)
# Quadratic MOMA built ONCE on the WT model (cobra add_moma, same formulation as
# crcgem.moma.single_gene_screen); per gene only the bounds of the reactions its GPR disables are
# set to 0 and restored. gurobipy Model.reset() before every solve is REQUIRED: without it the
# re-solve returned the previous knockout's solution (2026-10-27 test, ACH-000009). Verified equal to
# crcgem.moma.single_gene_screen on 5 genes (growth ratio and MOMA objective), 4-7x faster.
from cobra.flux_analysis.moma import add_moma
from cobra.core import Solution
from cobra.core.solution import get_solution
rows, flux_rows = [], {}
with m:
    add_moma(m, solution=Solution(objective_value=np.nan, status="optimal", fluxes=ref.fluxes), linear=False)
    rids = [r.id for r in m.reactions if not r.id.startswith("moma_")]
    m.solver.problem.Params.DualReductions = 0
    for i, gid in enumerate(genes):
        dis = [r for r in m.genes.get_by_id(gid).reactions if r.gpr.eval({gid}) is False]
        saved = [(r, r.bounds) for r in dis]
        for r in dis:
            r.bounds = (0.0, 0.0)
        m.solver.problem.reset()
        # solve via the solver directly: cobra raises on 'unbounded' even with raise_error=False.
        # DualReductions=0 (set below add_moma) makes Gurobi report infeasible vs unbounded exactly.
        status = m.solver.optimize()
        sol = get_solution(m) if status == "optimal" else None
        if status == "optimal":
            gr = float(sol.fluxes["MAR13082"])
            rows.append(dict(gene=gid, status=sol.status, growth=gr, growth_ratio=gr / ref.growth,
                             moma_objective=float(sol.objective_value), n_disabled=len(dis)))
            flux_rows[gid] = sol.fluxes[rids]
        else:
            rows.append(dict(gene=gid, status=status, growth=np.nan, growth_ratio=np.nan,
                             moma_objective=np.nan, n_disabled=len(dis)))
        for r, b in saved:
            r.bounds = b
        if i % 50 == 0:
            print(f"{i}/{len(genes)} {time.time() - t0:.0f}s", flush=True)
summary = pd.DataFrame(rows).set_index("gene")
flux = pd.DataFrame(flux_rows).T
summary.to_csv(out / "summary.tsv", sep="\t")
flux.astype("float32").to_parquet(out / "flux_matrix.parquet")
ref.fluxes.to_frame("reference").to_parquet(out / "reference.parquet")
meta.update(status="done", reference_growth=float(ref.growth), n_screened=len(genes),
            n_optimal=int((summary.status == "optimal").sum()),
            n_lethal_0p05=int((summary.growth_ratio < 0.05).sum()), seconds=round(time.time() - t0))
json.dump(meta, open(out / "meta.json", "w"), indent=1)
print(meta)
