"""Solver-tolerance patch of the MOMA single-gene knockout screen for one cell line (user decision 2026-10-30 13:18 KST,
option A; evidence result/gnn_20261028/discussion_20261030/review_tolerance.md).

What is re-solved. Only the knockouts whose GPR-disabled reactions carry a small reference flux, 0 < max|v_ref| <= 1e-5
(code label "trace"; the flux is the forced biomass cofactor demand of Human-GEM, review_tolerance.md section 4). At the
production FeasibilityTol these calls flip between solver instances; at 1e-8 they are stable (sections 2-3). The quadratic
MOMA QP is rebuilt exactly as script_python/58_moma_line.py (same repaired tINIT model, medium assembly lower bounds,
per-cell-line requirement caps and their gap-fill of script 62, biomass upper bound of the reference fallback when the
production meta.json records one, cobra add_moma on the SHIPPED pFBA reference, gurobipy Model.reset() before every solve,
DualReductions 0, Threads as given). The reference fluxome is NOT re-solved (reference.parquet copied); whether it must be
is measured, not assumed: (i) mass-balance residual max|S v_ref| and bound violations of the shipped reference against the
model, (ii) a fresh pFBA at FeasibilityTol 1e-8 compared with the shipped reference (growth, max|dv|, the small-flux band).
Solver parameters. Every Gurobi parameter in force on the production path is read back from the model and written to
meta.json (observed, not assumed). Measured 2026-10-30 13:30 (ACH-000651, probe_prod_params.py): the production path is
FeasibilityTol 1e-7, OptimalityTol 1e-7, IntFeasTol 1e-7, Presolve 0, Method 0 (cobra 0.29.1 model.tolerance = 1e-7 and optlang
presolve False / lp_method primal), not Gurobi's defaults 1e-6 / Presolve -1 that review_tolerance.md assumed for "production
default". The patch sets FeasibilityTol (--feas, default 1e-8) and the --set parameters given by the launcher (script_bash/105:
Presolve=-1 OptimalityTol=1e-6, the verifier's condition under which the stable calls were measured; with Presolve 0 the first
trace QP at 1e-8 had not finished after 4 min). Everything else stays as on the production path.
Control. A seeded random sample of non-trace knockouts (--n-control, default 30: 10 shipped-lethal, 10 intermediate,
10 ratio > 0.99) is re-solved at the same tolerance and compared with the shipped rows; these rows are NOT patched (option A).
The change measures the mixed-tolerance risk of option A.
Outputs (new root, never the production root): <out-root>/<line>/
  summary.tsv          '#' header lines + the shipped columns (gene status growth growth_ratio moma_objective n_disabled)
                       with the trace rows replaced, plus feasibility_tol and patched columns
  flux_matrix.parquet  shipped matrix with the trace rows replaced (float32; rows of non-optimal re-solves dropped as in 58)
  reference.parquet    byte-identical copy of the shipped reference
  patch_diff.tsv       per re-solved gene (trace and control): old vs new status, growth, growth_ratio, lethal call
                       (< 0.05), call change, max|dv| over all reactions, solve time
  reference_check.tsv  the reference measurements (i) and (ii)
  meta.json            production meta + patch provenance (parameters, versions, counts, timings)
Usage: python 104_moma_tolpatch_line.py <ModelID> [--threads 2] [--feas 1e-8] [--n-control 30] [--seed 0]
       [--src-root result/moma_20261027] [--out-root result/moma_20261030_tolpatch]
       [--models-root result/tinit_localT2_20261027/models] [--caps-dir data/medium/uptake_caps/line_requirement_caps]
       [--assembly ID]  (external validation set: --src-root result/oob_20261028/moma --models-root result/oob_20261028/models
       --caps-dir result/oob_20261028/line_requirement_caps --assembly <id> --out-root result/oob_20261030_tolpatch/moma)
Env: crcgem.
"""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np, pandas as pd, cobra
from cobra.flux_analysis.moma import add_moma
from cobra.core import Solution
from cobra.core.solution import get_solution
from cobra.util.array import create_stoichiometric_matrix

ROOT = Path("/data/workspace/cdgu/Winery/seeding/CRC_GEM")
sys.path.insert(0, str(ROOT / "module"))
from crcgem import reference_flux  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("line")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--feas", type=float, default=1e-8, help="Gurobi FeasibilityTol for the re-solves")
ap.add_argument("--trace-max", type=float, default=1e-5, help="trace band: 0 < max|v_ref| over disabled reactions <= this")
ap.add_argument("--n-control", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--src-root", default="result/moma_20261027")
ap.add_argument("--out-root", default="result/moma_20261030_tolpatch")
ap.add_argument("--models-root", default="result/tinit_localT2_20261027/models")
ap.add_argument("--caps-dir", default="data/medium/uptake_caps/line_requirement_caps")
ap.add_argument("--assembly")
ap.add_argument("--no-ref-check", action="store_true")
ap.add_argument("--genes", default="", help="comma-separated genes to re-solve instead of the trace band; no control set")
ap.add_argument("--set", action="append", default=[], metavar="PARAM=VALUE",
                help="measurement only: additional Gurobi parameter(s) for the re-solves (e.g. Presolve=-1); recorded in meta.json and headers")
args = ap.parse_args()
EXTRA = {}
for kv in args.set:
    k, v = kv.split("=", 1); EXTRA[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
ln = args.line
SRC = ROOT / args.src_root / ln
OUT = ROOT / args.out_root / ln
assert SRC.resolve() != OUT.resolve(), "output root must differ from the source root"
assert not (OUT / "summary.tsv").exists(), f"{OUT} already patched (no overwrite)"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
LETHAL = 0.05
PARAMS = ["FeasibilityTol", "OptimalityTol", "IntFeasTol", "Presolve", "Method", "Threads", "Seed", "BarConvTol", "BarQCPConvTol",
          "NumericFocus", "DualReductions", "Crossover", "ScaleFlag", "Aggregate"]


def params(m):
    P = m.solver.problem
    return {k: getattr(P.Params, k) for k in PARAMS}


# ---------------- shipped run ----------------
meta0 = json.load(open(SRC / "meta.json"))
assert meta0.get("status") == "done", f"{ln}: production status {meta0.get('status')!r}"
assert not meta0.get("gapfill_added"), f"{ln}: production used growth gap-fill (gapfill_added); not handled"
mu = float(meta0["reference_growth"])
ref = pd.read_parquet(SRC / "reference.parquet")["reference"]
sm0 = pd.read_csv(SRC / "summary.tsv", sep="\t", index_col=0)
fx0 = pd.read_parquet(SRC / "flux_matrix.parquet")

# ---------------- model build, verbatim logic of 58_moma_line.py:34-61 ----------------
lm = pd.read_csv(ROOT / "data/medium/line_medium_map.tsv", sep="\t").set_index("ModelID")
assembly = args.assembly or lm.loc[ln, "assembly_id"]
assert assembly == meta0["assembly"], (assembly, meta0["assembly"])
lb = pd.read_csv(ROOT / "data/medium/uptake_caps/exchange_lower_bounds_by_assembly.tsv", sep="\t", comment="#")
lb = lb[lb.assembly_id == assembly].set_index("exchange_rxn_id").lower_bound.to_dict()
m = cobra.io.read_sbml_model(str(ROOT / args.models_root / ln / f"{ln}.repaired.xml"))
m.solver = "gurobi"
m.solver.configuration.verbosity = 0
m.solver.problem.Params.Threads = args.threads
LRC = ROOT / args.caps_dir
try:
    ov = pd.read_csv(LRC / "lower_bound_overrides.tsv", sep="\t", comment="#")
except pd.errors.EmptyDataError:   # header-only file (58b, common-medium runs): no overrides
    ov = pd.DataFrame(columns=["line", "exchange_rxn_id", "lower_bound_new"])
ov = ov[ov.line == ln]
lb.update(dict(zip(ov.exchange_rxn_id, ov.lower_bound_new)))
# 23 of the 59 colorectal cell lines were screened before script 62 existed (meta without caps_raised / gapfill_script62); for every one of
# them the caps tables hold no row (checked 2026-10-30 13:45), so the rebuild with empty caps equals the production build
assert dict(zip(ov.exchange_rxn_id, ov.lower_bound_new)) == meta0.get("caps_raised", {}), "caps differ from production meta"
gf62 = json.load(open(LRC / "gapfill.json")).get(ln, [])
assert gf62 == meta0.get("gapfill_script62", []), "script-62 gap-fill differs from production meta"
if gf62:
    scaf62 = cobra.io.read_sbml_model(str(ROOT / "data/scaffold/Human-GEM.xml"))
    m.add_reactions([scaf62.reactions.get_by_id(x).copy() for x in gf62 if x not in m.reactions])
for r in m.exchanges:
    r.lower_bound = lb.get(r.id, 0.0)
bio = m.reactions.get_by_id("MAR13082")
m.objective = bio
wt = m.slim_optimize()
wt = 0.0 if wt is None or wt != wt else float(wt)
assert set(ref.index) == {r.id for r in m.reactions}, "reaction set differs from the shipped reference"
assert len(m.reactions) == meta0["n_reactions"], (len(m.reactions), meta0["n_reactions"])
prod_params = params(m)
print(f"{ln}: build {time.time()-t0:.0f}s; WT FBA mine {wt:.10g} shipped {meta0['wt_growth']:.10g}; production-path params {prod_params}", flush=True)
if meta0.get("reference_fallback"):
    # 58:97-99: after the Infeasible pFBA the biomass upper bound stays at wt*(1-1e-6) through the whole screen
    bio.upper_bound = meta0["wt_growth"] * (1 - 1e-6)
    print(f"{ln}: reference fallback in production -> biomass upper bound {bio.upper_bound:.10g} kept for the screen", flush=True)

# ---------------- reference measurements (decision input, not a patch) ----------------
refcheck = {}
if not args.no_ref_check:
    S = create_stoichiometric_matrix(m, array_type="DataFrame")
    v = ref.reindex(S.columns).to_numpy()
    resid = S.to_numpy() @ v
    lbv = np.array([r.lower_bound for r in m.reactions]); ubv = np.array([r.upper_bound for r in m.reactions])
    viol = np.maximum(np.maximum(lbv - v, v - ubv), 0)
    refcheck.update(shipped_ref_max_abs_Sv=float(np.abs(resid).max()), shipped_ref_n_metabolites_abs_Sv_gt_1e8=int((np.abs(resid) > 1e-8).sum()),
                    shipped_ref_n_metabolites_abs_Sv_gt_1e6=int((np.abs(resid) > 1e-6).sum()), shipped_ref_max_bound_violation=float(viol.max()),
                    shipped_ref_growth=float(ref["MAR13082"]), shipped_ref_n_abs_flux_le_1e5_gt_0=int(((np.abs(v) > 0) & (np.abs(v) <= 1e-5)).sum()))
    tr = time.time()
    with m:
        m.solver.problem.Params.FeasibilityTol = args.feas
        try:
            if meta0.get("reference_fallback"):
                m.solver.problem.reset()
            ref8 = reference_flux.compute(m, "pfba", diagnose=False)
            d = (ref8.fluxes.reindex(ref.index) - ref)
            band = (np.abs(ref) > 0) & (np.abs(ref) <= 1e-5)
            refcheck.update(pfba_feas_status="optimal", pfba_feas_growth=float(ref8.growth), pfba_feas_growth_minus_shipped=float(ref8.growth - mu),
                            pfba_feas_max_abs_dv=float(d.abs().max()), pfba_feas_n_rxn_abs_dv_gt_1e6=int((d.abs() > 1e-6).sum()),
                            pfba_feas_n_rxn_abs_dv_gt_1e3=int((d.abs() > 1e-3).sum()),
                            pfba_feas_sum_abs_flux_minus_shipped=float(ref8.fluxes.abs().sum() - ref.abs().sum()),
                            pfba_feas_band_max_abs_dv=float(d[band].abs().max()) if band.any() else 0.0,
                            pfba_feas_band_n_changed_gt_1e8=int((d[band].abs() > 1e-8).sum()), pfba_feas_seconds=round(time.time() - tr, 1))
        except Exception as e:  # noqa: BLE001
            refcheck.update(pfba_feas_status=f"failed: {type(e).__name__}: {e}"[:200], pfba_feas_seconds=round(time.time() - tr, 1))
    print(f"{ln}: reference check {refcheck}", flush=True)

# ---------------- knockout sets ----------------
genes = [g.id for g in m.genes if any(r.gpr.eval({g.id}) is False for r in g.reactions)]
assert len(genes) == meta0["n_screened"] and set(genes) == set(sm0.index), "screened gene set differs from production"
dis_of = {g: [r for r in m.genes.get_by_id(g).reactions if r.gpr.eval({g}) is False] for g in genes}
mx = {g: max(abs(float(ref[r.id])) for r in dis_of[g]) for g in genes}
trace = [g for g in genes if 0 < mx[g] <= args.trace_max]
if args.genes:
    trace = [g for g in args.genes.split(",") if g]
    assert set(trace) <= set(genes), "a --genes entry is not a screened gene"
    args.n_control = 0
rng = np.random.default_rng(args.seed)
tset = set(trace)
non = [g for g in genes if g not in tset and sm0.status[g] == "optimal"]
gr0 = sm0.growth_ratio
strata = {"lethal": [g for g in non if gr0[g] < LETHAL], "mid": [g for g in non if LETHAL <= gr0[g] <= 0.99], "one": [g for g in non if gr0[g] > 0.99]}
control = []
for s_, lst in strata.items():
    control += list(rng.choice(lst, size=min(args.n_control // 3, len(lst)), replace=False))
print(f"{ln}: screened {len(genes)}, trace {len(trace)} (shipped lethal {int((gr0.reindex(trace) < LETHAL).sum())}, non-optimal "
      f"{int((sm0.status.reindex(trace) != 'optimal').sum())}), control {len(control)} from {{{', '.join(f'{k}: {len(v)}' for k, v in strata.items())}}}", flush=True)

# ---------------- FBA knockout optimum (LP) of every re-solved gene: MOMA growth can never exceed it ----------------
# at the production-path parameters and at the patch tolerance; an FBA optimum of 0 with a MOMA growth ratio of 1 is a
# tolerance artefact (review_tolerance.md section 4). Consistency gate, written to patch_diff.tsv.
fba_ko = {}
P0 = m.solver.problem
with m:
    for tag, feas in (("prod", None), ("feas", args.feas)):
        if feas is not None:
            P0.Params.FeasibilityTol = feas
            for k, v_ in EXTRA.items():
                P0.setParam(k, v_)
        for gid in trace + list(control):
            dis = dis_of[gid]; saved = [(r, r.bounds) for r in dis]
            for r in dis:
                r.bounds = (0.0, 0.0)
            P0.reset()
            g_ = m.slim_optimize(error_value=np.nan)
            fba_ko[(tag, gid)] = float(g_) / mu if g_ == g_ else np.nan
            for r, b in saved:
                r.bounds = b
    P0.Params.FeasibilityTol = prod_params["FeasibilityTol"]
    for k in EXTRA:
        P0.setParam(k, prod_params[k] if k in prod_params else getattr(P0.Params, k))
print(f"{ln}: FBA knockout optima done {time.time()-t0:.0f}s", flush=True)

# ---------------- MOMA QP built once on the shipped reference (58:107-136) ----------------
rows, flux_rows, diff = {}, {}, []
with m:
    tb = time.time()
    add_moma(m, solution=Solution(objective_value=np.nan, status="optimal", fluxes=ref), linear=False)
    rids = [r.id for r in m.reactions if not r.id.startswith("moma_")]
    assert rids == list(fx0.columns), "flux matrix column order differs from the model"
    P = m.solver.problem
    P.Params.DualReductions = 0
    moma_params_prod = params(m)                 # what the production screen solved with (observed on this path)
    P.Params.FeasibilityTol = args.feas          # the ONLY change (plus --set, measurement runs only)
    for k, v_ in EXTRA.items():
        P.setParam(k, v_)
    patch_params = params(m)
    print(f"{ln}: MOMA built {time.time()-tb:.0f}s; screen params production-path {moma_params_prod}; patch {patch_params}", flush=True)
    for st, glist in (("trace", trace), ("control", control)):
        for i, gid in enumerate(glist):
            dis = dis_of[gid]
            saved = [(r, r.bounds) for r in dis]
            for r in dis:
                r.bounds = (0.0, 0.0)
            P.reset()
            ts = time.time()
            presolve_used = P.Params.Presolve
            try:
                status = m.solver.optimize()
                if status != "optimal":
                    # non-optimal status (infeasible / unbounded / numeric) under the patch condition: same one retry with Presolve 0
                    # (1-5 trace knockouts per cell line in the first 8 cell lines, 2026-10-30 14:05); recorded per gene (resolve_presolve)
                    st1 = status; P.Params.Presolve = 0; P.reset(); presolve_used = 0
                    try:
                        status = m.solver.optimize()
                        print(f"{ln}: {st} {gid} status {st1} under Presolve {patch_params['Presolve']}; retry Presolve 0 -> {status}", flush=True)
                    finally:
                        P.Params.Presolve = patch_params["Presolve"]
            except Exception as e:  # noqa: BLE001
                # ACH-000296 (2026-10-30 13:55, probe_296.py): GurobiError 100001 within 0.1 s under Presolve -1 for 44 of 51 trace QPs of
                # this one cell line (0 errors in the other cell lines); the same QP solves with Presolve 0 (8-18 s). Fallback = one retry
                # with Presolve 0 at the same FeasibilityTol, recorded per gene (column resolve_presolve); a second failure keeps the shipped row.
                err1 = f"error:{getattr(e, 'errno', '')}:{str(e)[:60]}"
                try:
                    P.Params.Presolve = 0; P.reset(); presolve_used = 0
                    status = m.solver.optimize()
                    print(f"{ln}: {st} {gid} solver error {err1} under Presolve {patch_params['Presolve']}; retry Presolve 0 -> {status}", flush=True)
                except Exception as e2:  # noqa: BLE001
                    status = f"{err1}|retry_presolve0:{getattr(e2, 'errno', '')}"
                    print(f"{ln}: {st} {gid} solver error twice {status}", flush=True)
                finally:
                    P.Params.Presolve = patch_params["Presolve"]
            sol = get_solution(m) if status == "optimal" else None
            if status == "optimal":
                gr = float(sol.fluxes["MAR13082"])
                row = dict(gene=gid, status=sol.status, growth=gr, growth_ratio=gr / mu, moma_objective=float(sol.objective_value), n_disabled=len(dis))
                fl = sol.fluxes[rids].astype("float32")
            else:
                row = dict(gene=gid, status=status, growth=np.nan, growth_ratio=np.nan, moma_objective=np.nan, n_disabled=len(dis))
                fl = None
            for r, b in saved:
                r.bounds = b
            old_opt = gid in fx0.index
            dv = float(np.abs(fl.to_numpy() - fx0.loc[gid].to_numpy()).max()) if (fl is not None and old_opt) else np.nan
            o_gr, n_gr = float(gr0[gid]), row["growth_ratio"]
            o_call = "lethal" if o_gr < LETHAL else ("nonlethal" if o_gr == o_gr else "nonoptimal")
            n_call = "lethal" if n_gr < LETHAL else ("nonlethal" if n_gr == n_gr else "nonoptimal")
            diff.append(dict(line=ln, set=st, gene=gid, n_disabled=len(dis), max_abs_ref_flux_disabled=mx[gid],
                             old_status=sm0.status[gid], new_status=row["status"], old_growth=float(sm0.growth[gid]), new_growth=row["growth"],
                             old_growth_ratio=o_gr, new_growth_ratio=n_gr, old_call=o_call, new_call=n_call, call_changed=o_call != n_call,
                             abs_d_growth_ratio=abs(n_gr - o_gr) if (n_gr == n_gr and o_gr == o_gr) else np.nan, max_abs_dv=dv,
                             old_moma_objective=float(sm0.moma_objective[gid]), new_moma_objective=row["moma_objective"],
                             fba_ko_growth_ratio_prod=fba_ko[("prod", gid)], fba_ko_growth_ratio_feas=fba_ko[("feas", gid)],
                             old_moma_exceeds_fba_feas=bool(o_gr == o_gr and o_gr > fba_ko[("feas", gid)] + 1e-3),
                             new_moma_exceeds_fba_feas=bool(n_gr == n_gr and n_gr > fba_ko[("feas", gid)] + 1e-3),
                             resolve_presolve=presolve_used, solve_s=round(time.time() - ts, 2)))
            if st == "trace":
                rows[gid] = row; flux_rows[gid] = fl
            if i % 20 == 0:
                print(f"{ln}: {st} {i}/{len(glist)} {time.time()-t0:.0f}s", flush=True)

# ---------------- patch ----------------
sm = sm0.copy()
sm["feasibility_tol"] = moma_params_prod["FeasibilityTol"]; sm["patched"] = False; sm["patch_status"] = "not_trace"
not_patched = {}
for gid, row in list(rows.items()):
    if row["status"] != "optimal":
        # re-solve not optimal (infeasible / unbounded / solver error): the shipped row is KEPT, flagged, and listed in meta.json
        not_patched[gid] = row["status"]; sm.loc[gid, "patch_status"] = f"kept_shipped_resolve_{row['status']}"[:120]
        rows.pop(gid); flux_rows.pop(gid, None); continue
    for k in ("status", "growth", "growth_ratio", "moma_objective", "n_disabled"):
        sm.loc[gid, k] = row[k]
    sm.loc[gid, "feasibility_tol"] = args.feas; sm.loc[gid, "patched"] = True; sm.loc[gid, "patch_status"] = "patched"
fx = fx0.drop(index=[g for g in rows if g in fx0.index])
new_fl = {g: f for g, f in flux_rows.items() if f is not None}
if new_fl:
    fx = pd.concat([fx, pd.DataFrame(new_fl).T.astype("float32")[fx0.columns]])
fx = fx.reindex([g for g in sm.index if g in set(fx.index)])      # production order (summary order), optimal rows only
assert (sm.status == "optimal").sum() == len(fx), ((sm.status == "optimal").sum(), len(fx))
dd = pd.DataFrame(diff)
tr_d = dd[dd.set == "trace"]; ct_d = dd[dd.set == "control"]
counts = dict(n_trace=len(trace), n_trace_call_changed=int(tr_d.call_changed.sum()),
              n_trace_lethal_to_nonlethal=int(((tr_d.old_call == "lethal") & (tr_d.new_call == "nonlethal")).sum()),
              n_trace_nonlethal_to_lethal=int(((tr_d.old_call == "nonlethal") & (tr_d.new_call == "lethal")).sum()),
              n_trace_nonoptimal_new=int((tr_d.new_status != "optimal").sum()), n_trace_nonoptimal_old=int((tr_d.old_status != "optimal").sum()),
              n_trace_lethal_old=int((tr_d.old_call == "lethal").sum()), n_trace_lethal_new=int((tr_d.new_call == "lethal").sum()),
              n_control=len(control), n_control_call_changed=int(ct_d.call_changed.sum()),
              control_max_abs_d_growth_ratio=float(ct_d.abs_d_growth_ratio.max()) if len(ct_d) else np.nan,
              control_median_abs_d_growth_ratio=float(ct_d.abs_d_growth_ratio.median()) if len(ct_d) else np.nan,
              control_max_abs_dv=float(ct_d.max_abs_dv.max()) if len(ct_d) else np.nan,
              control_n_abs_d_growth_ratio_gt_1e3=int((ct_d.abs_d_growth_ratio > 1e-3).sum()),
              n_trace_old_moma_exceeds_fba_feas=int(tr_d.old_moma_exceeds_fba_feas.sum()), n_trace_new_moma_exceeds_fba_feas=int(tr_d.new_moma_exceeds_fba_feas.sum()),
              n_trace_fba_feas_lethal=int((tr_d.fba_ko_growth_ratio_feas < LETHAL).sum()), n_trace_fba_prod_lethal=int((tr_d.fba_ko_growth_ratio_prod < LETHAL).sum()),
              n_control_old_moma_exceeds_fba_feas=int(ct_d.old_moma_exceeds_fba_feas.sum()), n_control_new_moma_exceeds_fba_feas=int(ct_d.new_moma_exceeds_fba_feas.sum()),
              n_trace_kept_shipped_resolve_nonoptimal=len(not_patched), trace_kept_shipped=not_patched,
              n_trace_resolved_with_presolve0=int((tr_d.resolve_presolve == 0).sum()) if "resolve_presolve" in tr_d else 0,
              n_control_resolved_with_presolve0=int((ct_d.resolve_presolve == 0).sum()) if "resolve_presolve" in ct_d else 0,
              fallback_policy="presolve0_retry_on_error_or_nonoptimal_then_keep_shipped",
              extra_params=EXTRA, trace_mean_solve_s=round(float(tr_d.solve_s.mean()), 2) if len(tr_d) else np.nan)
hdr = (f"# {ln}: solver-tolerance patch (script_python/104_moma_tolpatch_line.py, option A, user decision 2026-10-30 13:18 KST); source {SRC}; "
       f"model {ROOT / args.models_root / ln}/{ln}.repaired.xml; assembly {assembly}; caps {LRC}; reference = shipped reference.parquet (not re-solved)\n"
       f"# re-solved: trace knockouts (0 < max|v_ref| over GPR-disabled reactions <= {args.trace_max:g}) at FeasibilityTol {args.feas:g}; all other Gurobi parameters as on the production path: {patch_params}; "
       f"production-path screen parameters {moma_params_prod}\n"
       f"# tools: cobra {cobra.__version__}, gurobipy {'.'.join(map(str, __import__('gurobipy').gurobi.version()))}, pandas {pd.__version__}, numpy {np.__version__}; "
       f"threads {args.threads}; control set: {args.n_control} non-trace knockouts (seed {args.seed}, 1/3 shipped-lethal, 1/3 intermediate, 1/3 ratio > 0.99), re-solved at the same tolerance, NOT patched\n"
       f"# date {time.strftime('%Y-%m-%d %H:%M %Z')}; n screened {len(genes)}; counts {counts}; lethal call = growth_ratio < {LETHAL}\n")
with open(OUT / "summary.tsv", "w") as fh:
    fh.write(hdr); sm.to_csv(fh, sep="\t")
fx.astype("float32").to_parquet(OUT / "flux_matrix.parquet")
shutil.copyfile(SRC / "reference.parquet", OUT / "reference.parquet")
with open(OUT / "patch_diff.tsv", "w") as fh:
    fh.write(hdr + f"# rows: every re-solved knockout (set trace = patched, control = measured only); max_abs_dv = max |v_new - v_old| over all reactions (NaN when either solve is non-optimal)\n")
    dd.to_csv(fh, sep="\t", index=False)
if refcheck:
    with open(OUT / "reference_check.tsv", "w") as fh:
        fh.write(hdr + f"# reference measurements: shipped_ref_* = shipped reference.parquet against this model's S and bounds; pfba_feas_* = fresh pFBA (reference_flux.compute, diagnose False) at FeasibilityTol {args.feas:g} vs shipped\n")
        pd.DataFrame([dict(line=ln, **refcheck)]).to_csv(fh, sep="\t", index=False)
meta = dict(meta0)
meta.update(tolpatch=dict(script="script_python/104_moma_tolpatch_line.py", date=time.strftime("%Y-%m-%d %H:%M %Z"), source=str(SRC), feasibility_tol=args.feas,
                          trace_max=args.trace_max, threads=args.threads, seed=args.seed, n_control=args.n_control, wt_growth_rebuilt=wt,
                          production_path_params_before_moma=prod_params, production_path_params_screen=moma_params_prod, patch_params=patch_params,
                          trace_genes=trace, control_genes=list(map(str, control)), counts=counts, reference_check=refcheck,
                          versions=dict(cobra=cobra.__version__, gurobipy=".".join(map(str, __import__("gurobipy").gurobi.version())), pandas=pd.__version__, numpy=np.__version__),
                          seconds=round(time.time() - t0)),
            n_optimal=int((sm.status == "optimal").sum()), n_lethal_0p05=int((sm.growth_ratio < LETHAL).sum()))
json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
print(f"{ln}: done {time.time()-t0:.0f}s; {counts}; n_lethal {meta0['n_lethal_0p05']} -> {meta['n_lethal_0p05']}", flush=True)
