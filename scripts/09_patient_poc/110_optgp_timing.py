"""Timing probe for OptGP flux sampling on one cell line model (user ruling 2026-10-30: n = 100 samples; chains,
thinning and other sampling conditions from the literature; "time it").

Model build is a verbatim copy of script_python/58_moma_line.py (repaired tINIT model, medium assembly lower bounds,
per-cell-line requirement caps and their script-62 gap-fill, biomass MAR13082 as FBA objective; the growth gap-fill
branch of 58 is not reproduced here, the probe stops if WT growth is 0). No growth constraint is added during sampling
(the question under test).

Measured (each step appended to timing.tsv immediately, with the 1-min load average before/after):
  (a) OptGPSampler constructor on the WT model, split into problem build (constraint matrices + nullspace) and
      generate_fva_warmup (2 LPs per non-fixed reaction, objective Zero, cobra 0.29.1 hr_sampler.py:283-330)
  (b) sample(n) at each --thinning value, --processes chains (cobra OptGP: one chain per process, n/processes samples
      per chain, rows stacked chain by chain, optgp.py:162-170); validate(); biomass flux of warmup points and samples
      relative to the WT FBA optimum (does sampling drive toward optimal growth? measured, not assumed)
  (c) one knockout per --ko-gene (bounds of the GPR-disabled reactions set to 0 as in 58): fraction of WT warmup points
      and WT samples that violate the knockout bounds (is the WT warmup reusable?), then warmup regeneration on the
      knockout model and sample(n) on it
  (d) convergence on the n-sample set: Gelman-Rubin potential scale reduction factor R-hat per reaction across the
      --processes chains (Gelman & Rubin 1992, Stat Sci 7:457; Brooks & Gelman 1998 corrected form), and the
      replicate agreement of the mean flux between two seeds (relative difference), both in convergence_*.tsv
Usage: python 110_optgp_timing.py --line ACH-000009 --processes 4 --thinning 100 --n 100 --seeds 0 1 \
          --ko-genes ENSG00000073578 ENSG00000001084 --threads 2 --out result/optgp_timing_20261030
Env: crcgem (cobra 0.29.1, gurobi). Never overwrites: an existing out dir gets a __YYYYMMDD_HHMMSS suffix.
"""
import argparse, json, os, sys, time, platform
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, cobra, gurobipy, optlang
from cobra.sampling import OptGPSampler

ROOT = Path("/data/workspace/cdgu/Winery/seeding/CRC_GEM")
ap = argparse.ArgumentParser()
ap.add_argument("--line", default="ACH-000009")
ap.add_argument("--processes", type=int, default=4)
ap.add_argument("--thinning", type=int, nargs="+", default=[100])
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
ap.add_argument("--ko-genes", nargs="*", default=["ENSG00000073578", "ENSG00000001084"])
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--out", default="result/optgp_timing_20261030")
ap.add_argument("--models-root", default="result/tinit_localT2_20261027/models")
ap.add_argument("--caps-dir", default="data/medium/uptake_caps/line_requirement_caps")
ap.add_argument("--sample-dtype", choices=["float32", "float64"], default="float32",
                help="storage dtype of the samples parquet (2026-11-03: float32 rounding breaks S v = 0 at |v| ~ 1000; float64 for MOMA references)")
ap.add_argument("--assembly")  # patient POC 2026-11-03 (design_poc_addendum_20261103.md section 4): as 58:27
args = ap.parse_args()

out = ROOT / args.out
if out.exists():
    out = Path(str(out) + "__" + datetime.now().strftime("%Y%m%d_%H%M%S"))
out.mkdir(parents=True)
T0 = time.time()
HDR = [f"# source: script_python/110_optgp_timing.py, model {args.models_root}/{args.line}/{args.line}.repaired.xml + "
       f"medium assembly lower bounds + {args.caps_dir} (as script 58)",
       f"# params: processes={args.processes} thinning={args.thinning} n={args.n} seeds={args.seeds} ko_genes={args.ko_genes} "
       f"gurobi_threads={args.threads} growth_constraint=none",
       f"# versions: cobra {cobra.__version__}, optlang {optlang.__version__}, gurobi {'.'.join(map(str, gurobipy.gurobi.version()))}, "
       f"numpy {np.__version__}, python {platform.python_version()}, host {platform.node()} ({os.cpu_count()} cores)",
       f"# date: {datetime.now().isoformat(timespec='seconds')} KST",
       "# load: 1-min load average (os.getloadavg) recorded before/after each step; other jobs (58b MOMA screens) share the host"]


def load1():
    return round(os.getloadavg()[0], 2)


def write_tsv(df, path, extra=()):
    with open(path, "w") as f:
        f.write("\n".join(HDR + list(extra)) + "\n")
        df.to_csv(f, sep="\t")


tfile = out / "timing.tsv"
with open(tfile, "w") as f:
    f.write("\n".join(HDR) + "\n")
    f.write("step\tmodel_variant\tthinning\tn\tseed\tprocesses\tseconds\tload1_before\tload1_after\tnote\n")


def rec(step, variant, thin, n, seed, seconds, lb, note=""):
    with open(tfile, "a") as f:
        f.write(f"{step}\t{variant}\t{thin}\t{n}\t{seed}\t{args.processes}\t{seconds:.1f}\t{lb}\t{load1()}\t{note}\n")
    print(f"[{time.time()-T0:7.0f}s] {step} {variant} thin={thin} n={n} seed={seed}: {seconds:.1f}s {note}", flush=True)


# ---------------- model build (verbatim from 58_moma_line.py) ----------------
lb = load1(); t = time.time()
lm = pd.read_csv(ROOT / "data/medium/line_medium_map.tsv", sep="\t").set_index("ModelID")
assembly = args.assembly or lm.loc[args.line, "assembly_id"]  # as 58:35
lbd = pd.read_csv(ROOT / "data/medium/uptake_caps/exchange_lower_bounds_by_assembly.tsv", sep="\t", comment="#")
lbd = lbd[lbd.assembly_id == assembly].set_index("exchange_rxn_id").lower_bound.to_dict()
m = cobra.io.read_sbml_model(str(ROOT / args.models_root / args.line / f"{args.line}.repaired.xml"))
m.solver = "gurobi"
m.solver.configuration.verbosity = 0
m.solver.problem.Params.Threads = args.threads
LRC = ROOT / args.caps_dir
ov = pd.read_csv(LRC / "lower_bound_overrides.tsv", sep="\t", comment="#")
ov = ov[ov.line == args.line]
lbd.update(dict(zip(ov.exchange_rxn_id, ov.lower_bound_new)))
gf62 = json.load(open(LRC / "gapfill.json")).get(args.line, [])
if gf62:
    scaf62 = cobra.io.read_sbml_model(str(ROOT / "data/scaffold/Human-GEM.xml"))
    m.add_reactions([scaf62.reactions.get_by_id(x).copy() for x in gf62 if x not in m.reactions])
for r in m.exchanges:
    r.lower_bound = lbd.get(r.id, 0.0)
bio = m.reactions.get_by_id("MAR13082")
m.objective = bio
wt = m.slim_optimize()
wt = 0.0 if wt is None or wt != wt else float(wt)
assert wt > 1e-9, "WT growth 0; the growth gap-fill branch of 58 is not reproduced in this probe"
meta = dict(line=args.line, assembly=assembly, n_reactions=len(m.reactions), n_metabolites=len(m.metabolites),
            n_genes=len(m.genes), n_variables=len(m.variables), wt_growth=wt, caps_raised=dict(zip(ov.exchange_rxn_id, ov.lower_bound_new)),
            gapfill_script62=gf62, model_tolerance=m.tolerance, args=vars(args), versions=HDR[2][2:])
rec("build_model", "WT", "", "", "", time.time() - t, lb, f"reactions={len(m.reactions)} variables={len(m.variables)} wt={wt:.6g}")


# ---------------- instrumented sampler ----------------
class TimedOptGP(OptGPSampler):
    def generate_fva_warmup(self):
        # HRSampler.__init__ copies the model (hr_sampler.py:198); the copy is a fresh gurobi model whose Threads
        # parameter is the gurobi default (all cores). Measured 2026-10-30 17:07: the first launch ran the warmup at
        # ~960% CPU and pushed the host load from 11 to 44. Set the thread cap on the copy before the LPs.
        self.model.solver.problem.Params.Threads = args.threads
        self._t_build = time.time() - self._t0
        t = time.time()
        super().generate_fva_warmup()
        self._t_warmup = time.time() - t


def make_sampler(model, thin, seed, variant):
    lb = load1()
    TimedOptGP._t0 = time.time()
    s = TimedOptGP(model, processes=args.processes, thinning=thin, seed=seed)
    rec("sampler_problem_build", variant, thin, "", seed, s._t_build, lb,
        f"homogeneous={s.problem.homogeneous} nullspace={s.problem.nullspace.shape}")
    rec("warmup_generate_fva", variant, thin, "", seed, s._t_warmup, lb, f"n_warmup={s.n_warmup}")
    return s


def bio_stats(fluxes_bio, ref):
    q = np.quantile(fluxes_bio, [0, 0.05, 0.5, 0.95, 1])
    return dict(n=len(fluxes_bio), min=q[0], q05=q[1], median=q[2], q95=q[3], max=q[4], mean=float(np.mean(fluxes_bio)),
                frac_ge_0p9ref=float(np.mean(fluxes_bio >= 0.9 * ref)), frac_le_0p1ref=float(np.mean(fluxes_bio <= 0.1 * ref)),
                ref=ref)


def rhat(chains):
    """Gelman-Rubin PSRF per column. chains: (m chains, n draws, k reactions)."""
    mch, n, _ = chains.shape
    cm = chains.mean(axis=1)                       # (m,k)
    W = chains.var(axis=1, ddof=1).mean(axis=0)    # within
    B = n * cm.var(axis=0, ddof=1)                 # between
    var_hat = (n - 1) / n * W + B / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(var_hat / W)


def sample_and_eval(s, model, variant, thin, seed, ref_growth):
    lb = load1(); t = time.time()
    smp = s.sample(args.n)
    dt = time.time() - t
    valid = s.validate(smp)
    nv = int((valid == "v").sum())
    rec("sample", variant, thin, len(smp), seed, dt, lb, f"valid={nv}/{len(smp)} retries={s.retries}")
    bs = bio_stats(smp["MAR13082"].values, ref_growth)
    smp.astype(args.sample_dtype).to_parquet(out / f"samples_{variant}_thin{thin}_seed{seed}.parquet")
    # R-hat across the chains (rows stacked chain by chain, optgp.py:168-170)
    n_per = len(smp) // args.processes
    ch = smp.values.reshape(args.processes, n_per, -1)
    rh = rhat(ch)
    sd = smp.values.std(axis=0)
    moving = sd > 1e-9
    conv = pd.DataFrame(dict(reaction=smp.columns, mean=smp.values.mean(0), sd=sd, rhat=rh)).set_index("reaction")
    rh_m = rh[moving]
    summ = dict(n_reactions=len(rh), n_moving=int(moving.sum()), rhat_median=float(np.nanmedian(rh_m)),
                rhat_q90=float(np.nanquantile(rh_m, 0.9)), rhat_max=float(np.nanmax(rh_m)),
                frac_rhat_gt_1p1=float(np.nanmean(rh_m > 1.1)), frac_rhat_gt_1p2=float(np.nanmean(rh_m > 1.2)),
                chains=args.processes, draws_per_chain=n_per)
    write_tsv(conv, out / f"convergence_{variant}_thin{thin}_seed{seed}.tsv",
              [f"# rhat: Gelman-Rubin PSRF over {args.processes} chains x {n_per} draws; sd=0 reactions have rhat NaN",
               "# summary: " + json.dumps(summ)])
    return smp, dict(seconds_sample=dt, valid=nv, biomass=bs, rhat=summ)


def replicate_diff(a, b, variant, thin):
    ma, mb = a.values.mean(0), b.values.mean(0)
    sd = np.sqrt((a.values.var(0, ddof=1) + b.values.var(0, ddof=1)) / 2)
    scale = np.maximum(np.abs(ma), np.abs(mb))
    moving = sd > 1e-9
    rel = np.abs(ma - mb) / np.where(scale > 1e-9, scale, np.nan)
    z = np.abs(ma - mb) / np.where(sd > 1e-9, sd, np.nan)
    df = pd.DataFrame(dict(reaction=a.columns, mean_seed_a=ma, mean_seed_b=mb, abs_diff=np.abs(ma - mb), rel_diff=rel,
                           diff_over_pooled_sd=z)).set_index("reaction")
    summ = dict(n_moving=int(moving.sum()), rel_diff_median=float(np.nanmedian(rel[moving])),
                rel_diff_q90=float(np.nanquantile(rel[moving], 0.9)), frac_rel_diff_gt_0p1=float(np.nanmean(rel[moving] > 0.1)),
                diff_over_sd_median=float(np.nanmedian(z[moving])), diff_over_sd_q90=float(np.nanquantile(z[moving], 0.9)),
                corr_means=float(np.corrcoef(ma, mb)[0, 1]))
    write_tsv(df, out / f"replicate_means_{variant}_thin{thin}.tsv", ["# summary: " + json.dumps(summ)])
    return summ


results = dict(meta=meta, WT={}, KO={})
# ---------------- (a),(b),(d): WT ----------------
wt_samplers = {}
for thin in args.thinning:
    results["WT"][thin] = {}
    smps = {}
    for seed in args.seeds:
        s = make_sampler(m, thin, seed, "WT")
        if seed == args.seeds[0]:
            wb = s.warmup[:, s.fwd_idx] - s.warmup[:, s.rev_idx]
            bi = [r.id for r in m.reactions].index("MAR13082")
            results["WT"][thin]["warmup_biomass"] = bio_stats(wb[:, bi], wt)
            results["WT"][thin]["n_warmup"] = int(s.n_warmup)
            wt_samplers[thin] = s
        smps[seed], results["WT"][thin][f"seed{seed}"] = sample_and_eval(s, m, "WT", thin, seed, wt)
        json.dump(results, open(out / "results.json", "w"), indent=1, default=float)
    if len(args.seeds) >= 2:
        results["WT"][thin]["replicate"] = replicate_diff(smps[args.seeds[0]], smps[args.seeds[1]], "WT", thin)
    json.dump(results, open(out / "results.json", "w"), indent=1, default=float)

# ---------------- (c): knockouts ----------------
thin = args.thinning[0]
s_wt = wt_samplers[thin]
wt_smp = pd.read_parquet(out / f"samples_WT_thin{thin}_seed{args.seeds[0]}.parquet")
rids = [r.id for r in m.reactions]
for gid in args.ko_genes:
    g = m.genes.get_by_id(gid)
    dis = [r for r in g.reactions if r.gpr.eval({gid}) is False]
    idx = [rids.index(r.id) for r in dis]
    wb = s_wt.warmup[:, s_wt.fwd_idx] - s_wt.warmup[:, s_wt.rev_idx]
    tol = m.tolerance
    warm_viol = float(np.mean((np.abs(wb[:, idx]) > tol).any(axis=1)))
    smp_viol = float(np.mean((np.abs(wt_smp.values[:, idx]) > tol).any(axis=1)))
    ko = dict(gene=gid, disabled=[r.id for r in dis], n_disabled=len(dis),
              frac_wt_warmup_violating_ko_bounds=warm_viol, frac_wt_samples_violating_ko_bounds=smp_viol,
              wt_sample_mean_flux_disabled={r.id: float(wt_smp[r.id].mean()) for r in dis})
    saved = [(r, r.bounds) for r in dis]
    for r in dis:
        r.bounds = (0.0, 0.0)
    gko = m.slim_optimize(); gko = 0.0 if gko is None or gko != gko else float(gko)
    ko["ko_fba_growth"] = gko; ko["ko_fba_growth_ratio"] = gko / wt
    # validate WT samples against the knockout polytope (cobra validate uses the sampler's own bounds -> rebuild)
    s_ko = make_sampler(m, thin, args.seeds[0], f"KO_{gid}")
    v = s_ko.validate(wt_smp)
    ko["wt_samples_valid_in_ko_polytope"] = int((v == "v").sum())
    _, ko["sampling"] = sample_and_eval(s_ko, m, f"KO_{gid}", thin, args.seeds[0], wt)
    for r, b in saved:
        r.bounds = b
    results["KO"][gid] = ko
    json.dump(results, open(out / "results.json", "w"), indent=1, default=float)
    del s_ko

results["total_seconds"] = round(time.time() - T0)
json.dump(results, open(out / "results.json", "w"), indent=1, default=float)
print("DONE", out, flush=True)
