"""Patient POC quadratic MOMA re-solve with invariant gates G0-G2 (2026-11-04).
Task: coordinator brief 2026-11-04 (user rulings: all POC QPs at Gurobi FeasibilityTol 1e-9 on both sides, float64 storage).
Fixes vs 142 (verify_poc_fable_20261104.md R1, R2):
  (a) tumor reference = pFBA recomputed here at FeasibilityTol 1e-9, crcgem reference_flux.compute(model, 'pfba') as 58:96
      (diagnose=False: the face FVA of 58 only records degenerate reactions and does not enter the flux; 58's own retry uses the same).
      No fallback: if the pFBA is infeasible or not optimal, it is recorded in gates_ref and the tumor is stopped.
  (b) every knockout QP is solved with Gurobi Method 0 (primal simplex) and Method 1 (dual simplex); the solution with the lower
      objective among the optimal ones is kept (tie: Method 0).
Model build: verbatim from 142:53-77 (repaired tINIT model + assembly lower bounds + line_requirement_caps overrides and gap-fill;
WT growth > 1e-9 asserted). add_moma (cobra, quadratic) built once, reference swapped via moma_constraint RHS (142:143-157).
Knockout: GPR-disabled reactions to (0, 0) (142:80-96), gurobipy reset() before every solve, DualReductions 0.
Gates (all written to gates_ref_<tag>.tsv / summary_<tag>.tsv):
  G0 per reference: max bound violation <= 1e-9 and max|S v| <= 1e-9 on the full built model (incl. gap-fill reactions).
  G1 per reference: MOMA without knockout, Method 0 and Method 1; pass if the kept (lower, optimal) objective <= 1e-12
      (both method objectives recorded, plus G1_both_pass). A reference failing G0 or G1 gets no knockout solves (rows kept,
      status 'reference_gate_failed').
  G2 per knockout: status/objective of both methods, |obj0 - obj1|, kept method, max|S v_KO| and bound violation (knockout bounds)
      of the kept solution.
Outputs (--out/<line>/): summary_<tag>.tsv, gates_ref_<tag>.tsv, flux_<s>.parquet (kept solutions, float64, genes x reactions),
flux_alt_<s>.parquet (the other method's optimal solution), reference_pfba_feas<feas>.parquet (tumor; column 'reference'),
meta_<tag>.json; progress line per sample on stdout. --no-overwrite: new files get __YYYYMMDD_HHMMSS.
Usage: 171_moma_gated.py --line L --assembly A --models-root M --caps-dir C --genes G (--samples P | --tumor-pfba) [--sample-idx a-b]
       --out O [--feas 1e-9] [--threads 1] [--no-overwrite]   Env: crcgem.
"""
import argparse, json, platform, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, cobra, gurobipy, optlang
from cobra.flux_analysis.moma import add_moma
from cobra.core import Solution
from cobra.core.solution import get_solution
from cobra.util.array import create_stoichiometric_matrix

ROOT = Path("/data/workspace/cdgu/Winery/seeding/CRC_GEM")
sys.path.insert(0, str(ROOT / "module"))
from crcgem import reference_flux  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--line", required=True); ap.add_argument("--assembly")
ap.add_argument("--models-root", required=True); ap.add_argument("--caps-dir", required=True)
ap.add_argument("--samples"); ap.add_argument("--tumor-pfba", action="store_true")
ap.add_argument("--compare-ref", help="tumor: 58 reference.parquet, only for the max|v_new - v_58| record")
ap.add_argument("--genes", required=True)
ap.add_argument("--sample-idx", default=None)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--feas", type=float, default=1e-9)
ap.add_argument("--g0-tol", type=float, default=1e-9); ap.add_argument("--g1-tol", type=float, default=1e-12)
ap.add_argument("--out", required=True); ap.add_argument("--no-overwrite", action="store_true")
args = ap.parse_args()
assert bool(args.samples) != bool(args.tumor_pfba), "give exactly one of --samples / --tumor-pfba"
ab = lambda p: Path(p) if str(p).startswith("/") else ROOT / p
T0 = time.time()
METHODS = (0, 1)


def build_model():
    """142:53-77 verbatim."""
    lm = pd.read_csv(ROOT / "data/medium/line_medium_map.tsv", sep="\t").set_index("ModelID")
    assembly = args.assembly or lm.loc[args.line, "assembly_id"]
    lbd = pd.read_csv(ROOT / "data/medium/uptake_caps/exchange_lower_bounds_by_assembly.tsv", sep="\t", comment="#")
    lbd = lbd[lbd.assembly_id == assembly].set_index("exchange_rxn_id").lower_bound.to_dict()
    m = cobra.io.read_sbml_model(str(ab(args.models_root) / args.line / f"{args.line}.repaired.xml"))
    m.solver = "gurobi"; m.solver.configuration.verbosity = 0; m.solver.problem.Params.Threads = args.threads
    LRC = ab(args.caps_dir)
    try:
        ov = pd.read_csv(LRC / "lower_bound_overrides.tsv", sep="\t", comment="#")
    except pd.errors.EmptyDataError:
        ov = pd.DataFrame(columns=["line", "exchange_rxn_id", "lower_bound_new"])
    ov = ov[ov.line == args.line]
    lbd.update(dict(zip(ov.exchange_rxn_id, ov.lower_bound_new)))
    gf62 = json.load(open(LRC / "gapfill.json")).get(args.line, [])
    if gf62:
        scaf62 = cobra.io.read_sbml_model(str(ROOT / "data/scaffold/Human-GEM.xml"))
        m.add_reactions([scaf62.reactions.get_by_id(x).copy() for x in gf62 if x not in m.reactions])
    for r in m.exchanges:
        r.lower_bound = lbd.get(r.id, 0.0)
    m.objective = m.reactions.get_by_id("MAR13082")
    wt = m.slim_optimize(); wt = 0.0 if wt is None or wt != wt else float(wt)
    assert wt > 1e-9, "WT growth 0; the growth gap-fill branch of 58 is not reproduced (as 110)"
    return m, dict(assembly=assembly, wt_growth=wt, gapfill_script62=gf62, caps_raised=dict(zip(ov.exchange_rxn_id, ov.lower_bound_new)))


def solve_both(m, rids):
    """Method 0 and Method 1 on the current QP; returns per-method (status, objective, fluxes) and the kept method."""
    res = {}
    grb = m.solver.problem
    for meth in METHODS:
        grb.setParam("Method", meth); grb.reset()
        st = m.solver.optimize()
        if st == "optimal":
            sol = get_solution(m); res[meth] = (st, float(sol.objective_value), sol.fluxes[rids].to_numpy(np.float64))
        else:
            res[meth] = (st, np.nan, None)
    opt = [k for k in METHODS if res[k][0] == "optimal"]
    kept = min(opt, key=lambda k: (res[k][1], k)) if opt else None
    return res, kept


m, meta = build_model()
rids = [r.id for r in m.reactions]
genes = [l.strip() for l in open(ab(args.genes)) if l.strip() and not l.startswith("#")]
Smat = create_stoichiometric_matrix(m, array_type="lil").tocsr()
LB = np.array([r.lower_bound for r in m.reactions]); UB = np.array([r.upper_bound for r in m.reactions])
rpos = {r: i for i, r in enumerate(rids)}
out = ab(args.out) / args.line; out.mkdir(parents=True, exist_ok=True)
pfba_info = {}
if args.tumor_pfba:
    m.solver.configuration.tolerances.feasibility = args.feas
    m.solver.problem.Params.FeasibilityTol = args.feas
    # cold start: without reset() the FBA at 1e-9 reused the basis of build_model's 1e-7 solve, returned the 1e-7 optimum and the
    # pFBA (biomass >= growth*(1-1e-9)) was infeasible (dry 2026-11-04, TCGA-A6-2678-01A: warm 0.0338721499 vs cold 0.0336479214)
    m.solver.problem.reset()
    try:
        ref = reference_flux.compute(m, "pfba", diagnose=False)
        S = ref.fluxes[rids].astype("float64").to_frame().T.reset_index(drop=True)
        pfba_info = dict(pfba_status="optimal", pfba_growth=float(ref.growth))
        if args.compare_ref:
            r58 = pd.read_parquet(ab(args.compare_ref))["reference"][rids].to_numpy(np.float64)
            pfba_info["max_abs_diff_vs_58_reference"] = float(np.max(np.abs(S.loc[0].to_numpy() - r58)))
        fp = out / f"reference_pfba_feas{args.feas:g}.parquet"
        if fp.exists():
            assert args.no_overwrite, f"exists: {fp}"
            fp = fp.with_name(fp.stem + f"__{datetime.now():%Y%m%d_%H%M%S}.parquet")
        S.loc[0].rename("reference").to_frame().to_parquet(fp); pfba_info["reference_file"] = str(fp)
    except Exception as e:   # recorded, tumor stopped (no other method)
        pfba_info = dict(pfba_status=f"failed: {type(e).__name__}: {e}")
        S = None
else:
    S = pd.read_parquet(ab(args.samples))
    assert set(S.columns) == set(rids), f"reference columns != model reactions ({len(set(S.columns) ^ set(rids))} differ)"
    S = S[rids].astype("float64")
nS = 1 if S is None else len(S)
a, b = (map(int, args.sample_idx.split("-")) if args.sample_idx else (0, nS - 1))
idx = list(range(a, b + 1)); assert 0 <= a <= b < nS

tag = f"s{a}-{b}"
if args.no_overwrite and (out / f"summary_{tag}.tsv").exists():
    tag += "__" + datetime.now().strftime("%Y%m%d_%H%M%S")
HDR = [f"# source: script_python/171_moma_gated.py; model {args.models_root}/{args.line}/{args.line}.repaired.xml + assembly {meta['assembly']} "
       f"lower bounds + {args.caps_dir} (as 142/58; gap-fill {meta['gapfill_script62']}, caps raised {meta['caps_raised']}); reference "
       f"{args.samples + f' rows {a}-{b}' if args.samples else 'pFBA recomputed (crcgem reference_flux.compute pfba, diagnose=False, solver reset() before the FBA) at FeasibilityTol ' + format(args.feas, 'g')}",
       f"# params: quadratic MOMA (cobra add_moma linear=False) built once, reference swapped via moma_constraint RHS; gurobi Threads {args.threads}, "
       f"DualReductions 0, FeasibilityTol {args.feas:g}, Method 0 and Method 1 per QP, kept = lower objective among optimal (tie Method 0), "
       f"reset() before every solve; G0 tol {args.g0_tol:g} (bound violation, max|Sv|), G1 tol {args.g1_tol:g} (no-knockout objective, kept); "
       f"genes {args.genes} (n={len(genes)})",
       f"# versions: cobra {cobra.__version__}, optlang {optlang.__version__}, gurobi {'.'.join(map(str, gurobipy.gurobi.version()))}, "
       f"python {platform.python_version()}, host {platform.node()}",
       f"# date: {datetime.now().isoformat(timespec='seconds')} KST; n_references {len(idx)}; WT growth {meta['wt_growth']:.10g}; {json.dumps(pfba_info)}",
       "# exclusions: none; genes not in model -> status not_in_model; references failing G0/G1 -> status reference_gate_failed; "
       "non-optimal rows kept with NaN"]
fsum, fgate = out / f"summary_{tag}.tsv", out / f"gates_ref_{tag}.tsv"
SUMCOLS = ["sample_idx", "gene", "status", "moma_objective", "growth", "n_disabled", "kept_method", "status_m0", "objective_m0",
           "status_m1", "objective_m1", "abs_obj_diff", "max_abs_Sv_kept", "max_bound_violation_kept", "seconds"]
GATECOLS = ["sample_idx", "max_bound_violation", "max_abs_Sv", "G0_pass", "G1_status_m0", "G1_objective_m0", "G1_status_m1",
            "G1_objective_m1", "G1_kept_method", "G1_objective_kept", "G1_pass", "G1_both_pass", "knockouts_solved", "reference_biomass"]
fsum.write_text("\n".join(HDR) + "\n" + "\t".join(SUMCOLS) + "\n")
fgate.write_text("\n".join(HDR) + "\n" + "\t".join(GATECOLS) + "\n")
gin = [g for g in genes if g in m.genes]; missing = [g for g in genes if g not in m.genes]
fmt = lambda x: "nan" if x is None or (isinstance(x, float) and x != x) else (f"{x:.17g}" if isinstance(x, float) else str(x))


def append(f, cols, row):
    with open(f, "a") as fh:
        fh.write("\t".join(fmt(row.get(c)) for c in cols) + "\n")


if S is None:   # pFBA failed: one gate row, gene rows with the marker
    append(fgate, GATECOLS, dict(sample_idx=0, G0_pass=False, G1_pass=False, knockouts_solved=False))
    for g in genes:
        append(fsum, SUMCOLS, dict(sample_idx=0, gene=g, status="reference_pfba_failed", n_disabled=0))
    print("pFBA FAILED", pfba_info, flush=True)
else:
    with m:
        add_moma(m, solution=Solution(objective_value=np.nan, status="optimal", fluxes=S.loc[idx[0]]), linear=False)
        grb = m.solver.problem
        grb.Params.DualReductions = 0; grb.Params.FeasibilityTol = args.feas
        cons = [m.constraints[f"moma_constraint_{r}"] for r in rids]
        gcons = [grb.getConstrByName(c.name) for c in cons]
        assert all(gc.Sense == "=" for gc in gcons)
        for s in idx:
            ts = time.time()
            v = S.loc[s].to_numpy()
            viol = float(max(np.max(LB - v), np.max(v - UB), 0.0)); sv = float(np.max(np.abs(Smat @ v)))
            for c, val in zip(cons, v):
                c._lb = float(val); c._ub = float(val)
            grb.setAttr("RHS", gcons, v.tolist()); grb.update()
            g0 = viol <= args.g0_tol and sv <= args.g0_tol
            r1, k1 = solve_both(m, rids)
            o1 = r1[k1][1] if k1 is not None else np.nan
            g1 = k1 is not None and o1 <= args.g1_tol
            g1both = all(r1[k][0] == "optimal" and r1[k][1] <= args.g1_tol for k in METHODS)
            run = g0 and g1
            append(fgate, GATECOLS, dict(sample_idx=s, max_bound_violation=viol, max_abs_Sv=sv, G0_pass=g0,
                                         G1_status_m0=r1[0][0], G1_objective_m0=r1[0][1], G1_status_m1=r1[1][0], G1_objective_m1=r1[1][1],
                                         G1_kept_method=k1, G1_objective_kept=o1, G1_pass=g1, G1_both_pass=g1both, knockouts_solved=run,
                                         reference_biomass=float(S.loc[s, "MAR13082"])))
            fl, fla = {}, {}
            for gid in genes:
                if gid in missing:
                    append(fsum, SUMCOLS, dict(sample_idx=s, gene=gid, status="not_in_model", n_disabled=0)); continue
                dis = [r for r in m.genes.get_by_id(gid).reactions if r.gpr.eval({gid}) is False]
                if not dis:
                    append(fsum, SUMCOLS, dict(sample_idx=s, gene=gid, status="no_reaction_disabled", n_disabled=0)); continue
                if not run:
                    append(fsum, SUMCOLS, dict(sample_idx=s, gene=gid, status="reference_gate_failed", n_disabled=len(dis))); continue
                t = time.time()
                saved = [(r, r.bounds) for r in dis]
                for r in dis:
                    r.bounds = (0.0, 0.0)
                res, k = solve_both(m, rids)
                for r, bb in saved:
                    r.bounds = bb
                row = dict(sample_idx=s, gene=gid, n_disabled=len(dis), status_m0=res[0][0], objective_m0=res[0][1],
                           status_m1=res[1][0], objective_m1=res[1][1], abs_obj_diff=abs(res[0][1] - res[1][1]), kept_method=k)
                if k is None:
                    row["status"] = f"m0:{res[0][0]}|m1:{res[1][0]}"
                else:
                    vk = res[k][2]; lbk, ubk = LB.copy(), UB.copy()
                    for r in dis:
                        lbk[rpos[r.id]] = 0.0; ubk[rpos[r.id]] = 0.0
                    row.update(status="optimal", moma_objective=res[k][1], growth=float(vk[rpos["MAR13082"]]),
                               max_abs_Sv_kept=float(np.max(np.abs(Smat @ vk))),
                               max_bound_violation_kept=float(max(np.max(lbk - vk), np.max(vk - ubk), 0.0)))
                    fl[gid] = vk
                    o = 1 - k
                    if res[o][0] == "optimal":
                        fla[gid] = res[o][2]
                row["seconds"] = round(time.time() - t, 2)
                append(fsum, SUMCOLS, row)
            for d, nm in ((fl, "flux"), (fla, "flux_alt")):
                fp = out / f"{nm}_{s}.parquet"
                if fp.exists():
                    assert args.no_overwrite, f"exists: {fp}"
                    fp = out / f"{nm}_{s}__{datetime.now():%Y%m%d_%H%M%S}.parquet"
                pd.DataFrame(d, index=rids).T.astype("float64").to_parquet(fp)
            print(f"[{datetime.now():%F %T}] [{time.time()-T0:.0f}s] {args.line} sample {s}: G0 {g0} (viol {viol:.2e}, Sv {sv:.2e}) "
                  f"G1 {g1} (obj {o1:.3e}) knockouts {len(fl)}/{len(gin)} kept, {time.time()-ts:.0f}s", flush=True)

summ = pd.read_csv(fsum, sep="\t", comment="#")
assert len(summ) == len(idx) * len(genes), "row count != references x genes"
meta.update(line=args.line, args=vars(args), n_reactions=len(rids), genes_not_in_model=missing, sample_idx=[a, b], pfba=pfba_info,
            n_rows=len(summ), n_optimal=int((summ.status == "optimal").sum()), seconds=round(time.time() - T0))
json.dump(meta, open(out / f"meta_{tag}.json", "w"), indent=1, default=str)
print("DONE", out, tag, json.dumps({k: meta[k] for k in ("n_rows", "n_optimal", "seconds")}), flush=True)
