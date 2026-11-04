"""Medium sensitivity of the step-1 cancer-lethal gene list.

For each (DepMap CRC line, medium) build an E-Flux model on Human-GEM and run a
GPR-aware single-gene deletion screen; lethal = KO growth < 5% of that model's
own WT FBA optimum. Three media: as-shipped (open), essential-task IN set
unbounded, essential-task IN set capped at 1 mmol/gDW/h.

Ported 2026-10-26 from the 2026-10-19 screen script `lethal_media2.py` (verbatim
copy in result/ko_screen_20261026/recovered_1019_scripts/). The LP logic is
unchanged. Three changes only: (1) the HiGHS bindings path now points to
bin/pylibs_highs1.15.1, pinned to the same versions; (2) --permute-seed, a
gene-label permutation control; (3) --solver, for solver cross-checks
(set before the model is read).
--permute-seed: line expression values are permuted among the scaffold genes in
the panel, which preserves the distribution of metabolic-gene expression while
removing the gene-to-value assignment.
"""
import sys, os, json, time, argparse, warnings
sys.path.insert(0, "/data/workspace/cdgu/Winery/seeding/CRC_GEM/bin/pylibs_highs1.15.1")
sys.path.insert(0, "/data/workspace/cdgu/Winery/seeding/CRC_GEM/module")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, cobra
from crcgem.expression import reaction_scores, eflux_bounds

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
HG = f"{ROOT}/data/scaffold/Human-GEM.xml"
TASKS = f"{ROOT}/data/scaffold/metabolicTasks_Essential.txt"
EXPR = f"{ROOT}/data/depmap/crc_trainset/crc_expression.ENSG.parquet"


def task_in_mets(path):
    mets = set()
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7 or p[0].startswith("#"):
                continue
            for tok in p[4].split(";"):
                tok = tok.strip()
                if tok.endswith("[e]"):
                    mets.add(tok[:-3])
    return mets


def allowed_exchanges(m):
    names = task_in_mets(TASKS)
    e_mets = {met.name: met for met in m.metabolites if met.compartment == "e"}
    ex = set(m.exchanges)
    allow = set()
    for nm in sorted(names):
        met = e_mets.get(nm)
        if met is None:
            continue
        exs = [r for r in met.reactions if r in ex]
        if exs:
            allow.add(exs[0].id)
    return allow


def ham_exchanges(m):
    names = []; started = False
    for line in open(TASKS):
        p = line.rstrip("\n").split("\t")
        if len(p) < 6: continue
        if p[1] == "GR": started = True
        elif started and p[1] in ("BS","ER","IC","SU","GR"): break
        if started and p[4].strip().endswith("[e]"): names.append(p[4].strip()[:-3])
    ex = set(m.exchanges); e_mets = {mt.name: mt for mt in m.metabolites if mt.compartment == "e"}
    out = set()
    for nm in names:
        mt = e_mets.get(nm)
        if mt is None: continue
        exs = [r for r in mt.reactions if r in ex]
        if exs: out.add(exs[0].id)
    return out


def set_medium2(m, mode, allow, ham):
    if mode in ("ham","ham_cap1"):
        cap = 1000.0 if mode == "ham" else 1.0
        for r in m.exchanges:
            r.lower_bound = -cap if r.id in ham else 0.0
        return
    return set_medium(m, mode, allow)


def set_medium(m, mode, allow):
    if mode == "open":
        return
    cap = 1000.0 if mode == "taskIN" else 1.0
    for r in m.exchanges:
        r.lower_bound = -cap if r.id in allow else 0.0


def ko_reaction_map(m, genes):
    """gene id -> list of reaction ids that lose function when it is deleted."""
    out = {}
    for gid in genes:
        try:
            g = m.genes.get_by_id(gid)
        except KeyError:
            continue
        hit = []
        for r in g.reactions:
            try:
                functional = r.gpr.eval({gid})
            except Exception:
                functional = None
            if functional is False:
                hit.append(r.id)
        out[gid] = hit
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", required=True)
    ap.add_argument("--medium", required=True, choices=["open","taskIN","taskIN_cap1","ham","ham_cap1"])
    ap.add_argument("--max-genes", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eflux-scale", type=float, default=1000.0)
    ap.add_argument("--no-eflux", action="store_true")
    ap.add_argument("--permute-seed", type=int, default=None)
    ap.add_argument("--solver", default="hybrid")
    a = ap.parse_args()

    t0 = time.time()
    # Env default solver is Gurobi since 2026-10-26; cloning a Gurobi problem into
    # 'hybrid' fails, so the solver is fixed before the model is read.
    cobra.Configuration().solver = a.solver
    m = cobra.io.read_sbml_model(HG)
    m.solver = a.solver
    allow = allowed_exchanges(m)
    expr = pd.read_parquet(EXPR)
    s = expr[a.line]
    universe = sorted(set(g.id for g in m.genes) & set(expr.index))
    komap = ko_reaction_map(m, universe)
    if a.permute_seed is not None:
        rng = np.random.default_rng(a.permute_seed)
        u = s.loc[universe]
        s = s.copy()
        s.loc[universe] = rng.permutation(u.to_numpy())
    n_struct = sum(1 for v in komap.values() if v)

    set_medium2(m, a.medium, allow, ham_exchanges(m))
    if a.no_eflux:
        bnds = {}
    else:
        scores = reaction_scores(m, s)
        bnds = eflux_bounds(m, scores, scale=a.eflux_scale, default=None)
        for rid, (lo, hi) in bnds.items():
            m.reactions.get_by_id(rid).bounds = (lo, hi)
    wt = m.slim_optimize()
    rec = {"line": a.line, "medium": a.medium, "eflux_scale": (None if a.no_eflux else a.eflux_scale), "permute_seed": a.permute_seed, "solver": a.solver, "wt": None if wt is None else float(wt),
           "n_universe": len(universe), "n_structural_candidates": n_struct,
           "n_rxn_eflux_bounded": len(bnds), "load_s": time.time() - t0}

    cand = [g for g, v in komap.items() if v]
    if a.max_genes:
        cand = cand[:a.max_genes]
    res = {}
    t1 = time.time()
    for gid in cand:
        with m:
            for rid in komap[gid]:
                m.reactions.get_by_id(rid).bounds = (0.0, 0.0)
            v = m.slim_optimize()
        res[gid] = 0.0 if (v is None or not np.isfinite(v)) else float(v)
    rec["screen_s"] = time.time() - t1
    rec["n_lp"] = len(cand)
    denom = wt if (wt and wt > 0) else np.nan
    ratios = {g: (v / denom if np.isfinite(denom) else np.nan) for g, v in res.items()}
    lethal = sorted(g for g, r in ratios.items() if np.isfinite(r) and r < 0.05)
    rec["n_lethal"] = len(lethal)
    rec["lethal"] = lethal
    rec["ratios"] = ratios
    with open(a.out, "w") as f:
        json.dump(rec, f)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("lethal", "ratios")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
