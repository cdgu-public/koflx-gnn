"""Quadratic MOMA from each OptGP sample (adjacent-normal reference of the patient POC), 2026-11-03.
Spec: result/external_plan_20261103/design_poc_addendum_20261103.md section 4 ("MOMA from each sample, NEW script_python/142").
Model and bounds: the lines of script_python/110_optgp_timing.py (verbatim from 58: repaired tINIT model, medium assembly lower
bounds, per-line requirement caps of 62 and their gap-fill; the growth gap-fill branch of 58 is not reproduced, stop if WT growth 0).
add_moma (cobra, quadratic) is built ONCE on the first reference. cobra stores the reference only in the bounds of the constraints
moma_constraint_<rxn> (lb = ub = reference flux, cobra flux_analysis/moma.py:153-157). For each further reference the bounds of every
such constraint are set to v_s[rxn]: lb and ub together, written as one RHS update of the equality rows in Gurobi, with the optlang
_lb/_ub mirrored, so no transient lb > ub occurs. Knockout loop as 58:111-141 (GPR-disabled reactions to (0,0), gurobipy reset()
before every solve, DualReductions 0, solve through the solver interface).
Checks (design section 10 P2, implementation only):
  --check-equivalence  for the dry genes and every selected sample, moma_objective of the reused QP vs a fresh model + add_moma per
                       sample; assert relative difference <= 1e-6 (written to equivalence.tsv).
  bound violation      per sample, max violation of the model bounds by the stored reference, and max|S v| (in the summary file).
  --reference-parquet  use a single reference (58 reference.parquet; sample_idx 0) instead of --samples; with --compare-58 <summary.tsv>
                       the recomputed rows are compared with the 58 output (compare58.tsv; assert rel. diff of moma_objective <= 1e-6
                       for optimal rows and identical status).
Outputs (--out/<line>/, # header per rule 26): summary_s<a>-<b>.tsv (sample_idx, gene, status, moma_objective, growth, n_disabled;
appended per sample), refcheck_s<a>-<b>.tsv, flux_<sample_idx>.parquet (genes x model reactions, --flux-dtype, default float64 since 2026-11-03), meta_s<a>-<b>.json.
Per-worker file names (not one summary.tsv) because W workers write into one folder. --no-overwrite: existing files of the same range
get a __YYYYMMDD_HHMMSS suffix on the new files instead of being replaced.
Usage: see design section 9 (f). Env: crcgem.
"""
import argparse, json, os, platform, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, cobra, gurobipy, optlang
from cobra.flux_analysis.moma import add_moma
from cobra.core import Solution
from cobra.core.solution import get_solution
from cobra.util.array import create_stoichiometric_matrix

ROOT = Path("/data/workspace/cdgu/Winery/seeding/CRC_GEM")
ap = argparse.ArgumentParser()
ap.add_argument("--line", required=True); ap.add_argument("--assembly")
ap.add_argument("--models-root", required=True); ap.add_argument("--caps-dir", required=True)
ap.add_argument("--samples"); ap.add_argument("--reference-parquet"); ap.add_argument("--compare-58")
ap.add_argument("--genes", required=True, help="text file, one ENSG per line, '#' lines skipped")
ap.add_argument("--sample-idx", default=None, help="a-b inclusive (default: all)")
ap.add_argument("--dry-genes", type=int, default=0, help="use only the first N genes (dry test)")
ap.add_argument("--check-equivalence", action="store_true")
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--flux-dtype", choices=["float32", "float64"], default="float64",
                help="storage dtype of flux_<s>.parquet (2026-11-03: float32 lost 50-60%% of sum dv^2 at small objectives, verify_poc_distance)")
ap.add_argument("--feas", type=float, default=None, help="Gurobi FeasibilityTol for the MOMA QP (default None = cobra model.tolerance 1e-7, unchanged; user 2026-11-03 re-solve at 1e-9)")
ap.add_argument("--set", action="append", default=[], metavar="PARAM=VALUE", help="extra Gurobi parameter, measurement runs only (2026-11-04 dv noise at 1e-9)")
ap.add_argument("--out", required=True); ap.add_argument("--no-overwrite", action="store_true")
args = ap.parse_args()
assert bool(args.samples) != bool(args.reference_parquet), "give exactly one of --samples / --reference-parquet"
ab = lambda p: Path(p) if str(p).startswith("/") else ROOT / p
T0 = time.time()


def build_model():
    """Lines of 110:83-106 (verbatim from 58), with 58's empty-override guard."""
    lm = pd.read_csv(ROOT / "data/medium/line_medium_map.tsv", sep="\t").set_index("ModelID")
    assembly = args.assembly or lm.loc[args.line, "assembly_id"]
    lbd = pd.read_csv(ROOT / "data/medium/uptake_caps/exchange_lower_bounds_by_assembly.tsv", sep="\t", comment="#")
    lbd = lbd[lbd.assembly_id == assembly].set_index("exchange_rxn_id").lower_bound.to_dict()
    m = cobra.io.read_sbml_model(str(ab(args.models_root) / args.line / f"{args.line}.repaired.xml"))
    m.solver = "gurobi"; m.solver.configuration.verbosity = 0; m.solver.problem.Params.Threads = args.threads
    LRC = ab(args.caps_dir)
    try:
        ov = pd.read_csv(LRC / "lower_bound_overrides.tsv", sep="\t", comment="#")
    except pd.errors.EmptyDataError:   # as 58:47
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


def ko_solve(m, gid, rids):
    """58:122-141 for one gene on a model that already carries add_moma."""
    dis = [r for r in m.genes.get_by_id(gid).reactions if r.gpr.eval({gid}) is False]
    if not dis:
        return dict(gene=gid, status="no_reaction_disabled", moma_objective=np.nan, growth=np.nan, n_disabled=0), None
    saved = [(r, r.bounds) for r in dis]
    for r in dis:
        r.bounds = (0.0, 0.0)
    m.solver.problem.reset()
    status = m.solver.optimize()
    row = dict(gene=gid, status=status, moma_objective=np.nan, growth=np.nan, n_disabled=len(dis)); fl = None
    if status == "optimal":
        sol = get_solution(m)
        row.update(moma_objective=float(sol.objective_value), growth=float(sol.fluxes["MAR13082"])); fl = sol.fluxes[rids]
    for r, b in saved:
        r.bounds = b
    return row, fl


def with_moma(m, ref):
    add_moma(m, solution=Solution(objective_value=np.nan, status="optimal", fluxes=ref), linear=False)
    m.solver.problem.Params.DualReductions = 0
    if args.feas is not None:
        m.solver.problem.Params.FeasibilityTol = args.feas
    for kv in args.set:
        k, val = kv.split("="); m.solver.problem.setParam(k, float(val) if "." in val or "e" in val else int(val))


m, meta = build_model()
rids = [r.id for r in m.reactions]
genes = [l.strip() for l in open(ab(args.genes)) if l.strip() and not l.startswith("#")]
if args.dry_genes:
    genes = genes[: args.dry_genes]
if args.samples:
    S = pd.read_parquet(ab(args.samples))
else:
    S = pd.read_parquet(ab(args.reference_parquet))["reference"].to_frame().T.reset_index(drop=True)
assert set(S.columns) == set(rids), f"reference columns != model reactions ({len(set(S.columns) ^ set(rids))} differ)"
S = S[rids].astype("float64")
a, b = (map(int, args.sample_idx.split("-")) if args.sample_idx else (0, len(S) - 1))
idx = list(range(a, b + 1)); assert 0 <= a <= b < len(S)

out = ab(args.out) / args.line; out.mkdir(parents=True, exist_ok=True)
tag = f"s{a}-{b}"
if args.no_overwrite and (out / f"summary_{tag}.tsv").exists():
    tag += "__" + datetime.now().strftime("%Y%m%d_%H%M%S")
HDR = [f"# source: script_python/142_moma_from_samples.py; model {args.models_root}/{args.line}/{args.line}.repaired.xml + assembly "
       f"{meta['assembly']} lower bounds + {args.caps_dir} (as 58/110); reference {args.samples or args.reference_parquet} rows {a}-{b}",
       f"# params: quadratic MOMA (cobra add_moma linear=False) built once, reference swapped via moma_constraint RHS; gurobi threads "
       f"{args.threads}, DualReductions 0, FeasibilityTol {args.feas if args.feas is not None else 'cobra default'}, extra {args.set}, reset() before every solve; genes {args.genes} (n={len(genes)}, dry_genes={args.dry_genes})",
       f"# versions: cobra {cobra.__version__}, optlang {optlang.__version__}, gurobi {'.'.join(map(str, gurobipy.gurobi.version()))}, "
       f"python {platform.python_version()}, host {platform.node()}",
       f"# date: {datetime.now().isoformat(timespec='seconds')} KST; n_samples {len(idx)}; status != optimal rows kept with NaN"]
fsum, fref = out / f"summary_{tag}.tsv", out / f"refcheck_{tag}.tsv"
for f, cols in ((fsum, "sample_idx\tgene\tstatus\tmoma_objective\tgrowth\tn_disabled\tseconds"),
                (fref, "sample_idx\tmax_bound_violation\tmax_abs_Sv\treference_biomass")):
    f.write_text("\n".join(HDR) + "\n" + cols + "\n")

Smat = create_stoichiometric_matrix(m, array_type="lil").tocsr()
LB = np.array([r.lower_bound for r in m.reactions]); UB = np.array([r.upper_bound for r in m.reactions])
gin = [g for g in genes if g in m.genes]
missing = [g for g in genes if g not in m.genes]

with m:
    with_moma(m, S.loc[idx[0]])
    cons = [m.constraints[f"moma_constraint_{r}"] for r in rids]
    grb = m.solver.problem
    gcons = [grb.getConstrByName(c.name) for c in cons]
    assert all(gc.Sense == "=" for gc in gcons)
    qp_times = []
    for s in idx:
        v = S.loc[s].to_numpy()
        viol = float(max(np.max(LB - v), np.max(v - UB), 0.0))
        with open(fref, "a") as f:
            f.write(f"{s}\t{viol:.3e}\t{float(np.max(np.abs(Smat @ v))):.3e}\t{float(S.loc[s, 'MAR13082']):.6g}\n")
        for c, val in zip(cons, v):   # optlang mirror (lb = ub, no transient lb > ub)
            c._lb = float(val); c._ub = float(val)
        grb.setAttr("RHS", gcons, v.tolist()); grb.update()
        rows, fl = [], {}
        for gid in gin:
            t = time.time(); row, f_ = ko_solve(m, gid, rids); row["seconds"] = round(time.time() - t, 2); qp_times.append(row["seconds"])
            rows.append(row)
            if f_ is not None:
                fl[gid] = f_
        rows += [dict(gene=g, status="not_in_model", moma_objective=np.nan, growth=np.nan, n_disabled=0, seconds=0) for g in missing]
        with open(fsum, "a") as f:
            pd.DataFrame(rows).assign(sample_idx=s)[["sample_idx", "gene", "status", "moma_objective", "growth", "n_disabled",
                                                    "seconds"]].to_csv(f, sep="\t", header=False, index=False)
        fp = out / f"flux_{s}.parquet"
        if args.no_overwrite and fp.exists():
            fp = out / f"flux_{s}__{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        pd.DataFrame(fl).T.astype(args.flux_dtype).to_parquet(fp)
        print(f"[{time.time()-T0:.0f}s] sample {s}: {len(gin)} genes, median QP {np.median(qp_times[-len(gin):]) if gin else 0:.2f}s", flush=True)

summ = pd.read_csv(fsum, sep="\t", comment="#")
meta.update(line=args.line, args=vars(args), n_reactions=len(rids), genes_not_in_model=missing, sample_idx=[a, b],
            qp_seconds_median=float(np.median(qp_times)) if qp_times else None, qp_seconds_max=float(np.max(qp_times)) if qp_times else None,
            n_rows=len(summ), n_optimal=int((summ.status == "optimal").sum()), seconds=round(time.time() - T0))
assert len(summ) == len(idx) * len(genes), "row count != samples x genes"

if args.check_equivalence:   # fresh model + add_moma per sample (P2)
    eq = []
    for s in idx:
        m2, _ = build_model(); r2 = [r.id for r in m2.reactions]
        with m2:
            with_moma(m2, S.loc[s])
            for gid in gin:
                row, _ = ko_solve(m2, gid, r2)
                ref_row = summ[(summ.sample_idx == s) & (summ.gene == gid)].iloc[0]
                x, y = float(ref_row.moma_objective), row["moma_objective"]
                rel = abs(x - y) / max(abs(x), abs(y), 1e-300)
                eq.append(dict(sample_idx=s, gene=gid, status_reused=ref_row.status, status_fresh=row["status"],
                               moma_objective_reused=x, moma_objective_fresh=y, rel_diff=rel))
    eq = pd.DataFrame(eq)
    with open(out / f"equivalence_{tag}.tsv", "w") as f:
        f.write("\n".join(HDR + ["# P2 equivalence: reused QP (RHS swap) vs fresh add_moma per sample; pass if rel_diff <= 1e-6"]) + "\n")
        eq.to_csv(f, sep="\t", index=False)
    meta["equivalence_max_rel_diff"] = float(eq.rel_diff.max())
    print(eq.to_string(), flush=True)
    assert (eq.status_reused == eq.status_fresh).all() and (eq.rel_diff <= 1e-6).all(), "P2 equivalence FAILED"

if args.compare_58:   # P2: 142-style recomputation vs the 58 output on the same model and reference
    o = pd.read_csv(ab(args.compare_58), sep="\t", comment="#").set_index("gene")
    c = summ.set_index("gene").join(o[["status", "moma_objective"]], rsuffix="_58", how="left")
    c["rel_diff"] = (c.moma_objective - c.moma_objective_58).abs() / np.maximum(np.maximum(c.moma_objective.abs(), c.moma_objective_58.abs()), 1e-300)
    with open(out / f"compare58_{tag}.tsv", "w") as f:
        f.write("\n".join(HDR + [f"# P2: vs {args.compare_58}; pass if status equal and rel_diff <= 1e-6 for optimal rows"]) + "\n")
        c.to_csv(f, sep="\t")
    opt = c.status == "optimal"
    meta["compare58_max_rel_diff"] = float(c.rel_diff[opt].max()) if opt.any() else None
    print(c.to_string(), flush=True)
    assert (c.status == c.status_58).all() and (c.rel_diff[opt] <= 1e-6).all(), "P2 compare-58 FAILED"

json.dump(meta, open(out / f"meta_{tag}.json", "w"), indent=1, default=str)
print("DONE", out, tag, json.dumps({k: meta[k] for k in ("qp_seconds_median", "qp_seconds_max", "n_rows", "n_optimal", "seconds")}), flush=True)
