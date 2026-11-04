"""Diagnosis of the 14 trace knockouts whose re-solve under the patch condition (script 104: FeasibilityTol 1e-8, Presolve -1 then 0,
OptimalityTol 1e-6) was non-optimal and whose shipped row was kept (summary.tsv patch_status kept_shipped_resolve_*), 2026-10-30.
User question (verbatim): "Optimal이 안 나온건 multiple solution 인지 infeasible인지? Infeasible이면 ungrowth로 보고 ATPM 같은 maintenance
function들 꺼봐야 할 듯". Diagnostic only: no dataset or patch file is changed.

Per knockout (model rebuilt exactly as 104 / 58: repaired tINIT model, assembly lower bounds, per-cell-line caps + script-62 gap-fill,
reference-fallback biomass bound), with the GPR-disabled reactions closed:
  A. forced fluxes in the model: every reaction with lower bound > 0 or upper bound < 0 (internal, demand, maintenance), with ids/names
     (the ATP maintenance reaction of Human-GEM is MAR03964 "ATP phosphohydrolase"; its bounds are reported whatever they are)
  B. feasibility of the knockout polytope: LP with objective 0 at FeasibilityTol 1e-8 (Presolve -1 and 0) and at the production path
     (1e-7, Presolve 0); raw Gurobi Status code and optlang status. If infeasible: Gurobi IIS (constraints = metabolites, variable
     bounds = reactions) and, one at a time, every forced bound of A relaxed to 0 (plus all together): feasibility, FBA growth, MOMA growth
  C. FBA knockout optimum at 1e-8 and at the production path (status, growth ratio)
  D. MOMA QP (cobra add_moma, linear=False, on the shipped reference): check that the objective covers every reaction (NumQNZs == number
     of reactions == moma_dist variables; strictly convex in the distance variables, so no multiple optima), then raw Gurobi Status under
     Presolve -1 / Presolve 0 / barrier (Method 2, BarHomogeneous 1) / NumericFocus 3, all at FeasibilityTol 1e-8, OptimalityTol 1e-6
  E. membership of the knockout in the GNN sample set of the patched dataset (81b fluxome-change filter) and its new gene-disjoint fold
Usage: python 104c_diag_nonoptimal_tolpatch.py <line> [--threads 2] [--out result/moma_20261030_tolpatch/_nonoptimal_diag]
       python 104c_diag_nonoptimal_tolpatch.py --aggregate   (concatenate the per-cell-line tables into rows.tsv, forced_bounds.tsv, iis.tsv, relaxation.tsv)
Env: crcgem.
"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/data/workspace/cdgu/Winery/seeding/CRC_GEM")
ap = argparse.ArgumentParser()
ap.add_argument("line", nargs="?")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--patch-root", default="result/moma_20261030_tolpatch")
ap.add_argument("--dataset", default="result/gnn_20261030_tolpatch/dataset")
ap.add_argument("--out", default="result/moma_20261030_tolpatch/_nonoptimal_diag")
ap.add_argument("--aggregate", action="store_true")
args = ap.parse_args()
OUT = ROOT / args.out; OUT.mkdir(parents=True, exist_ok=True)
HDR = (f"# script_python/104c_diag_nonoptimal_tolpatch.py; patch root {args.patch_root}; dataset {args.dataset}; knockouts = summary.tsv rows with patch_status kept_shipped_resolve_*; "
       f"model/caps/reference as script 104 (= 58); FeasibilityTol 1e-8 / OptimalityTol 1e-6 unless stated; production path = FeasibilityTol 1e-7, OptimalityTol 1e-7, Presolve 0, Method 0; "
       f"lethal = growth_ratio < 0.05; date {time.strftime('%Y-%m-%d %H:%M %Z')}\n")

if args.aggregate:
    for name in ("rows", "forced_bounds", "iis", "relaxation"):
        parts = []
        for f in sorted(glob.glob(str(OUT / f"ACH-*_{name}.tsv"))):
            try:
                parts.append(pd.read_csv(f, sep="\t", comment="#"))
            except pd.errors.EmptyDataError:   # per-cell-line table with no rows (e.g. no forced bounds, no infeasible polytope)
                pass
        assert not (OUT / f"{name}.tsv").exists(), f"{name}.tsv exists (no overwrite)"
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        with open(OUT / f"{name}.tsv", "w") as fh:
            fh.write(HDR + f"# aggregated from {len(parts)} per-cell-line tables; n rows {len(df)}\n"); df.to_csv(fh, sep="\t", index=False)
        print(name, len(df))
    sys.exit(0)

import cobra, gurobipy  # noqa: E402
from cobra.flux_analysis.moma import add_moma  # noqa: E402
from cobra.core import Solution  # noqa: E402
ln = args.line; t0 = time.time()
PR = ROOT / args.patch_root / ln
for f in ("rows", "forced_bounds", "iis", "relaxation"):
    assert not (OUT / f"{ln}_{f}.tsv").exists(), f"{OUT}/{ln}_{f}.tsv exists (no overwrite)"
meta = json.load(open(PR / "meta.json")); mu = meta["reference_growth"]
src = Path(meta["tolpatch"]["source"])
sm = pd.read_csv(PR / "summary.tsv", sep="\t", index_col=0, comment="#")
kept = sm[sm.patch_status.astype(str).str.startswith("kept")]
assert len(kept), f"{ln}: no kept rows"
ref = pd.read_parquet(PR / "reference.parquet")["reference"]

# ---- model build (104 / 58) ----
lm = pd.read_csv(ROOT / "data/medium/line_medium_map.tsv", sep="\t").set_index("ModelID")
assembly = meta["assembly"]
lb = pd.read_csv(ROOT / "data/medium/uptake_caps/exchange_lower_bounds_by_assembly.tsv", sep="\t", comment="#")
lb = lb[lb.assembly_id == assembly].set_index("exchange_rxn_id").lower_bound.to_dict()
models_root = "result/tinit_localT2_20261027/models" if "moma_20261027" in str(src) else "result/oob_20261028/models"
caps_dir = "data/medium/uptake_caps/line_requirement_caps" if "moma_20261027" in str(src) else "result/oob_20261028/line_requirement_caps"
m = cobra.io.read_sbml_model(str(ROOT / models_root / ln / f"{ln}.repaired.xml"))
m.solver = "gurobi"; m.solver.configuration.verbosity = 0; m.solver.problem.Params.Threads = args.threads
ov = pd.read_csv(ROOT / caps_dir / "lower_bound_overrides.tsv", sep="\t", comment="#"); ov = ov[ov.line == ln]
lb.update(dict(zip(ov.exchange_rxn_id, ov.lower_bound_new)))
gf62 = json.load(open(ROOT / caps_dir / "gapfill.json")).get(ln, [])
if gf62:
    scaf62 = cobra.io.read_sbml_model(str(ROOT / "data/scaffold/Human-GEM.xml"))
    m.add_reactions([scaf62.reactions.get_by_id(x).copy() for x in gf62 if x not in m.reactions])
for r in m.exchanges:
    r.lower_bound = lb.get(r.id, 0.0)
bio = m.reactions.get_by_id("MAR13082"); m.objective = bio
wt = m.slim_optimize(); assert abs(wt - meta["wt_growth"]) < 1e-9, (wt, meta["wt_growth"])
if meta.get("reference_fallback"):
    bio.upper_bound = meta["wt_growth"] * (1 - 1e-6)
P = m.solver.problem
PROD = dict(FeasibilityTol=1e-7, OptimalityTol=1e-7, Presolve=0, Method=0)
PATCH = dict(FeasibilityTol=1e-8, OptimalityTol=1e-6, Presolve=-1, Method=0)


def setp(**kw):
    for k, v in kw.items():
        P.setParam(k, v)


# ---- A. forced fluxes ----
forced = [r for r in m.reactions if r.lower_bound > 0 or r.upper_bound < 0]
atpm = m.reactions.get_by_id("MAR03964") if "MAR03964" in m.reactions else None
fb = pd.DataFrame([dict(line=ln, reaction=r.id, name=r.name, lower_bound=r.lower_bound, upper_bound=r.upper_bound, is_exchange=r in m.exchanges,
                        is_boundary=r.boundary, n_metabolites=len(r.metabolites), reference_flux=float(ref[r.id]), is_atp_maintenance=(r.id == "MAR03964")) for r in forced])
with open(OUT / f"{ln}_forced_bounds.tsv", "w") as fh:
    fh.write(HDR + f"# {ln}: reactions with lower bound > 0 or upper bound < 0 after the build (n {len(forced)}); ATP maintenance MAR03964 present {atpm is not None}, bounds {atpm.bounds if atpm else None}\n")
    fb.to_csv(fh, sep="\t", index=False)
print(f"{ln}: build {time.time()-t0:.0f}s; forced bounds {len(forced)}: {[(r.id, r.bounds) for r in forced][:12]}; MAR03964 {atpm.bounds if atpm else 'absent'}", flush=True)

# ---- E. GNN membership and new gene fold (81b filter and group_folds on the patched dataset) ----
D = ROOT / args.dataset
smeta = json.load(open(D / "scaffold_meta.json")); gene_rxns = smeta["gene_rxns"]
samp = []
for f in sorted((D / "lines").glob("*.npz")):
    z = np.load(f); th = 1e-6 / float(z["mu_ref"]); V = np.where(np.abs(z["V"]) > th, z["V"], 0); vw = np.where(np.abs(z["vwt"]) > th, z["vwt"], 0)
    keep = np.abs(V - vw).max(1) > th
    samp += [(f.stem, str(g)) for g in z["genes"][keep]]
S = pd.DataFrame(samp, columns=["line", "gene"]); S["gkey"] = [",".join(map(str, sorted(gene_rxns.get(g, [])))) for g in S.gene]
ug, inv = np.unique(S.gkey.to_numpy(), return_inverse=True); n = np.bincount(inv)
order = sorted(range(len(ug)), key=lambda j: (-n[j], ug[j])); fn = np.zeros(5); fo = np.zeros(len(ug), int)
for t, j in enumerate(order):
    f_ = t if t < 5 else int(np.argmin(fn)); fo[j] = f_; fn[f_] += n[j]
S["fold_gene"] = fo[inv]; S = S.set_index(["line", "gene"])


def ko_bounds(gid):
    dis = [r for r in m.genes.get_by_id(gid).reactions if r.gpr.eval({gid}) is False]
    return dis, [(r, r.bounds) for r in dis]


def status_raw():
    return int(P.Status)


def feas_lp(**params):
    """objective 0: status string, raw code"""
    with m:
        m.objective = m.problem.Objective(0, direction="min")
        setp(**params); P.reset(); st = m.solver.optimize()
        return st, status_raw()


def fba(**params):
    with m:
        m.objective = bio; setp(**params); P.reset(); st = m.solver.optimize()
        g = (bio.forward_variable.primal - bio.reverse_variable.primal) / mu if st == "optimal" else np.nan
        return st, status_raw(), g


rows, iis_rows, relax_rows = [], [], []
for gid in kept.index:
    dis, saved = ko_bounds(gid)
    for r in dis:
        r.bounds = (0.0, 0.0)
    row = dict(line=ln, gene=gid, patch_status=sm.patch_status[gid], shipped_status=sm.status[gid], shipped_growth_ratio=float(sm.growth_ratio[gid]),
               n_disabled=len(dis), disabled_reactions=";".join(r.id for r in dis), max_abs_ref_flux_disabled=max(abs(float(ref[r.id])) for r in dis))
    # B feasibility
    for tag, prm in (("feas_lp_patch_pre-1", PATCH), ("feas_lp_patch_pre0", dict(PATCH, Presolve=0)), ("feas_lp_prod", PROD)):
        st, code = feas_lp(**prm); row[tag] = st; row[tag + "_code"] = code
    # C FBA
    for tag, prm in (("fba_patch", PATCH), ("fba_prod", PROD)):
        st, code, g = fba(**prm); row[tag] = st; row[tag + "_code"] = code; row[tag + "_growth_ratio"] = g
    polytope_infeasible = row["feas_lp_patch_pre-1"] != "optimal" and row["feas_lp_patch_pre0"] != "optimal"
    row["polytope_feasible_1e-8"] = not polytope_infeasible
    if polytope_infeasible:
        with m:
            m.objective = m.problem.Objective(0, direction="min"); setp(**PATCH); P.reset(); m.solver.optimize()
            try:
                P.computeIIS()
                for c in P.getConstrs():
                    if c.IISConstr:
                        iis_rows.append(dict(line=ln, gene=gid, kind="metabolite_balance", name=c.ConstrName, lower_bound=np.nan, upper_bound=np.nan))
                for v in P.getVars():
                    if v.IISLB or v.IISUB:
                        rid = v.VarName.replace("_reverse_", "|").split("|")[0] if "_reverse_" in v.VarName else v.VarName
                        iis_rows.append(dict(line=ln, gene=gid, kind="variable_bound" + ("_lb" if v.IISLB else "") + ("_ub" if v.IISUB else ""), name=v.VarName,
                                             lower_bound=v.LB, upper_bound=v.UB, reaction=rid, reaction_name=m.reactions.get_by_id(rid).name if rid in m.reactions else "",
                                             reaction_bounds=str(m.reactions.get_by_id(rid).bounds) if rid in m.reactions else "", disabled_by_ko=rid in {r.id for r in dis}))
                row["iis_n_constraints"] = sum(1 for x in iis_rows if x["line"] == ln and x["gene"] == gid and x["kind"] == "metabolite_balance")
                row["iis_n_bounds"] = sum(1 for x in iis_rows if x["line"] == ln and x["gene"] == gid and x["kind"] != "metabolite_balance")
            except gurobipy.GurobiError as e:
                row["iis_error"] = str(e)[:80]
        # relaxations: each forced bound to 0, then all together
        trials = [(r.id, [r]) for r in forced] + ([("ALL_FORCED", forced)] if len(forced) > 1 else [])
        for name, rs in trials:
            saved2 = [(r, r.bounds) for r in rs]
            for r in rs:
                r.bounds = (min(r.lower_bound, 0.0), max(r.upper_bound, 0.0))
            st, code = feas_lp(**PATCH); stf, codef, g = fba(**PATCH)
            gm, stm = np.nan, "not_run"
            if st == "optimal":
                with m:
                    add_moma(m, solution=Solution(objective_value=np.nan, status="optimal", fluxes=ref), linear=False)
                    P2 = m.solver.problem; P2.Params.DualReductions = 0
                    for k, v in PATCH.items():
                        P2.setParam(k, v)
                    P2.reset(); stm = m.solver.optimize()
                    if stm != "optimal":
                        P2.Params.Presolve = 0; P2.reset(); stm = m.solver.optimize() + "(presolve0)"
                    gm = (bio.forward_variable.primal - bio.reverse_variable.primal) / mu if stm.startswith("optimal") else np.nan
            relax_rows.append(dict(line=ln, gene=gid, relaxed=name, relaxed_bounds_before=";".join(f"{r.id}:{b}" for r, b in saved2), feas_lp=st, feas_lp_code=code,
                                   fba_status=stf, fba_growth_ratio=g, moma_status=stm, moma_growth_ratio=gm, moma_lethal=(gm < 0.05) if gm == gm else np.nan))
            for r, b in saved2:
                r.bounds = b
    # D MOMA QP under several solver settings
    with m:
        add_moma(m, solution=Solution(objective_value=np.nan, status="optimal", fluxes=ref), linear=False)
        P2 = m.solver.problem; P2.Params.DualReductions = 0; P2.update()   # gurobipy attributes are lazy; update before reading NumQNZs
        row["moma_n_reactions"] = len([r for r in m.reactions if not r.id.startswith("moma_")]); row["moma_num_qnz"] = int(P2.NumQNZs)
        row["moma_n_dist_vars"] = sum(1 for v in m.solver.variables if v.name.startswith("moma_dist_"))
        row["moma_objective_covers_all_reactions"] = row["moma_num_qnz"] == row["moma_n_reactions"] == row["moma_n_dist_vars"]
        for tag, prm in (("moma_pre-1", PATCH), ("moma_pre0", dict(PATCH, Presolve=0)), ("moma_barrier_homog", dict(PATCH, Method=2, BarHomogeneous=1)),
                         ("moma_numfocus3", dict(PATCH, NumericFocus=3)), ("moma_numfocus3_pre0", dict(PATCH, NumericFocus=3, Presolve=0)),
                         ("moma_feas1e-7_pre-1", dict(PATCH, FeasibilityTol=1e-7)), ("moma_feas1e-7_numfocus3", dict(PATCH, FeasibilityTol=1e-7, NumericFocus=3)),
                         ("moma_prod", PROD)):
            for k, v in dict(Method=0, BarHomogeneous=-1, NumericFocus=0).items():
                P2.setParam(k, v)
            for k, v in prm.items():
                P2.setParam(k, v)
            P2.reset(); ts = time.time()
            try:
                st = m.solver.optimize(); code = int(P2.Status)
                g = (bio.forward_variable.primal - bio.reverse_variable.primal) / mu if st == "optimal" else np.nan
            except Exception as e:  # noqa: BLE001
                st, code, g = f"error:{getattr(e, 'errno', '')}", -1, np.nan
            row[tag] = st; row[tag + "_code"] = code; row[tag + "_growth_ratio"] = g; row[tag + "_s"] = round(time.time() - ts, 1)
            print(f"{ln} {gid} {tag}: {st} code {code} G={g} {time.time()-ts:.1f}s", flush=True)
    for r, b in saved:
        r.bounds = b
    key = (ln, gid); row["in_gnn_sample_set_new"] = key in S.index; row["new_gene_fold"] = int(S.loc[key, "fold_gene"]) if key in S.index else -1
    rows.append(row)
    print(f"{ln} {gid}: {json.dumps({k: (v if not isinstance(v, float) else round(v, 6)) for k, v in row.items() if k not in ('disabled_reactions',)})}", flush=True)

for name, df, note in (("rows", pd.DataFrame(rows), "one row per kept knockout; Gurobi Status codes: 2 optimal, 3 infeasible, 4 inf_or_unbd, 5 unbounded, 12 numeric, 13 suboptimal"),
                       ("iis", pd.DataFrame(iis_rows), "Gurobi IIS of the knockout polytope feasibility LP (patch condition); only for infeasible polytopes"),
                       ("relaxation", pd.DataFrame(relax_rows), "forced bounds relaxed to 0 one at a time (and all together) for infeasible polytopes; feasibility, FBA, MOMA at the patch condition")):
    with open(OUT / f"{ln}_{name}.tsv", "w") as fh:
        fh.write(HDR + f"# {ln}: {note}; n {len(df)}\n"); df.to_csv(fh, sep="\t", index=False)
print(f"{ln}: done {time.time()-t0:.0f}s", flush=True)
