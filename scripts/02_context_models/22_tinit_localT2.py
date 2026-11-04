#!/usr/bin/env python
"""tINIT extraction with the local threshold of Richelle et al. 2019 (localT2, 25th/75th percentile)
[ref: doi: 10.1371/journal.pcbi.1007185] (user ruling 2026-10-26). Copy of script 20; changes:
gene-level threshold theta_g = 25th global percentile if the gene's mean over the lines is below it,
75th if above it, otherwise the gene's mean; gene score = log2(TPM+1) - theta_g; reaction score by
GPR (AND=min, OR=max); optional --essential-reactions file (task protection);
allow_excretion=True, no_reverse_loops=False (with allow_excretion=False the full Human-GEM MILP is
infeasible at once, probe 2026-10-27); kept reactions read as indices (TROPPO return value; script 20
read them as a boolean mask, which is wrong). Exchanges are left
as in the scaffold (all open; no medium constraint at extraction, user 2026-10-27).

Original docstring of script 20 follows.
STANDALONE tINIT extraction for the server (extraction env A).

Self-contained: imports ONLY cobra + troppo + numpy/pandas (NOT the crcgem
package, which targets modern cobra). Runs on the extraction env
(cobra 0.22.1 / optlang 1.5.2 / troppo; see environment_extraction.yml) with a
commercial MILP solver (Gurobi or CPLEX — the tolerances.optimality API that
GLPK lacks is native there, so no patch is needed).

For each cell line (column of the ENSG expression matrix) it builds a
context-specific model with tINIT and writes an SBML + kept-reaction list.

    python script_python/20_tinit_extract_standalone.py \
        --expression data/depmap/crc_trainset/crc_expression.ENSG.parquet \
        --scaffold   data/scaffold/Human-GEM.xml \
        --solver     gurobi \
        --threshold-percentile 25 \
        --out        result/crc_context_models

Output per line: <ModelID>.xml, <ModelID>.kept_reactions.txt; plus
extraction_summary.tsv and consensus_reactions.txt (reactions kept in >=2 models
— the fixed reaction set the downstream GNN/MOMA use).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("tinit")

_TOK = re.compile(r"\(|\)|\bor\b|\band\b|[^\s()]+", re.IGNORECASE)


def gpr_score(rule: str, vals: dict) -> float:
    """Evaluate a GPR to a scalar: AND=min (scarcest subunit), OR=max (isozymes).
    Genes absent from `vals` contribute NaN; an all-NaN term stays NaN."""
    rule = (rule or "").strip()
    if not rule:
        return float("nan")
    toks = _TOK.findall(rule)
    p = [0]

    def por():
        v = [pand()]
        while p[0] < len(toks) and toks[p[0]].lower() == "or":
            p[0] += 1; v.append(pand())
        v = [x for x in v if x == x]
        return max(v) if v else float("nan")

    def pand():
        v = [atom()]
        while p[0] < len(toks) and toks[p[0]].lower() == "and":
            p[0] += 1; v.append(atom())
        v = [x for x in v if x == x]
        return min(v) if v else float("nan")

    def atom():
        t = toks[p[0]]
        if t == "(":
            p[0] += 1; val = por()
            if p[0] < len(toks) and toks[p[0]] == ")":
                p[0] += 1
            return val
        p[0] += 1
        return float(vals.get(t, float("nan")))

    return por()


def reaction_scores(model, gene_score: pd.Series) -> list:
    """Signed reaction scores for tINIT from gene scores (expression - local threshold):
    >0 = evidence present, <0 = absent. NaN reactions (no GPR / no data) -> 0 (neutral)."""
    vals = gene_score.to_dict()
    raw = np.array([gpr_score(r.gene_reaction_rule, vals) for r in model.reactions])
    raw[np.isnan(raw)] = 0.0
    return raw.tolist()


def local_thresholds(expr: pd.DataFrame, lo_pct: float, hi_pct: float) -> pd.Series:
    """localT2 (Richelle et al. 2019): global lower/upper percentiles over all genes x lines;
    gene threshold = clip(gene mean over lines, lower, upper)."""
    v = expr.to_numpy().ravel(); v = v[~np.isnan(v)]
    lo, hi = np.percentile(v, lo_pct), np.percentile(v, hi_pct)
    return expr.mean(axis=1).clip(lower=lo, upper=hi), float(lo), float(hi)


def extract_one(scaffold_path, gene_score, solver, extra_essential=()):
    import cobra
    from troppo.methods_wrappers import ReconstructionWrapper
    from troppo.methods.reconstruction.tINIT import tINITProperties

    model = cobra.io.read_sbml_model(scaffold_path)
    model.solver = solver
    scores = reaction_scores(model, gene_score)
    ids = [r.id for r in model.reactions]
    biomass = [i for i, r in enumerate(model.reactions) if r.objective_coefficient != 0]
    biomass += [ids.index(x) for x in extra_essential if x in ids and ids.index(x) not in biomass]

    rw = ReconstructionWrapper(model, ttg_ratio=1e4)
    props = tINITProperties(reactions_scores=scores, present_metabolites=[],
                            essential_reactions=biomass, production_weight=0.0,
                            allow_excretion=True, no_reverse_loops=False)
    keep = rw.run(props)   # TROPPO returns the INDICES of kept reactions (not a boolean mask)
    all_ids = [r.id for r in model.reactions]
    kept = [all_ids[i] for i in sorted(set(int(i) for i in keep))]

    ctx = model.copy()
    ctx.remove_reactions([r for r in ctx.reactions if r.id not in set(kept)],
                         remove_orphans=True)
    return ctx, kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--expression", required=True, help="ENSG x cell-lines parquet")
    ap.add_argument("--scaffold", required=True)
    ap.add_argument("--solver", default="gurobi", choices=["gurobi", "cplex"])
    ap.add_argument("--lower-percentile", type=float, default=25.0)
    ap.add_argument("--upper-percentile", type=float, default=75.0)
    ap.add_argument("--essential-reactions", help="file of reaction ids to protect (task protection)")
    ap.add_argument("--consensus-min-models", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lines", help="comma-separated subset of ModelIDs (default: all)")
    ap.add_argument("--threshold-expression", help="matrix the localT2 thresholds are computed from (default: "
                    "--expression); out-of-bag lines use the 59 CRC lines (script 76, 2026-10-28)")
    args = ap.parse_args()

    import cobra

    expr = pd.read_parquet(args.expression)          # ENSG x cell lines (log2 TPM+1)
    lines = ([c.strip() for c in args.lines.split(",")] if args.lines
             else list(expr.columns))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # thresholds always from ALL lines of the matrix, so a subset run gives the same models
    texpr = pd.read_parquet(args.threshold_expression) if args.threshold_expression else expr
    theta, lo, hi = local_thresholds(texpr, args.lower_percentile, args.upper_percentile)
    theta = theta.reindex(expr.index)
    extra = ([x.strip() for x in Path(args.essential_reactions).read_text().split() if x.strip()]
             if args.essential_reactions else [])
    log.info("extracting %d lines (solver=%s); localT2 p%.0f=%.4g p%.0f=%.4g; %d extra essential",
             len(lines), args.solver, args.lower_percentile, lo, args.upper_percentile, hi, len(extra))
    threshold = {"lower": lo, "upper": hi}

    rows, kept_counter = [], Counter()
    for i, line in enumerate(lines):
        t = time.time()
        try:
            ctx, kept = extract_one(args.scaffold, expr[line] - theta, args.solver, extra)
            cobra.io.write_sbml_model(ctx, str(out / f"{line}.xml"))
            (out / f"{line}.kept_reactions.txt").write_text("\n".join(kept))
            kept_counter.update(kept)
            rows.append({"model_id": line, "n_kept": len(kept),
                         "n_genes": len(ctx.genes), "seconds": round(time.time()-t, 1)})
            log.info("[%d/%d] %s -> %d reactions (%.0fs)", i+1, len(lines), line,
                     len(kept), time.time()-t)
        except Exception as exc:
            rows.append({"model_id": line, "n_kept": -1, "error": f"{type(exc).__name__}: {exc}"})
            log.error("[%d/%d] %s FAILED: %s", i+1, len(lines), line, exc)

    pd.DataFrame(rows).to_csv(out / "extraction_summary.tsv", sep="\t", index=False)
    consensus = sorted(r for r, n in kept_counter.items() if n >= args.consensus_min_models)
    (out / "consensus_reactions.txt").write_text("\n".join(consensus))
    (out / "manifest.json").write_text(json.dumps({
        "scaffold": args.scaffold, "solver": args.solver,
        "threshold": "localT2", "lower_percentile": args.lower_percentile,
        "upper_percentile": args.upper_percentile, "threshold_values": threshold,
        "essential_reactions_file": args.essential_reactions, "n_extra_essential": len(extra),
        "n_lines": len(lines), "n_succeeded": int(sum(r["n_kept"] >= 0 for r in rows)),
        "consensus_min_models": args.consensus_min_models,
        "n_consensus_reactions": len(consensus),
    }, indent=2))
    log.info("done: %d consensus reactions (kept in >=%d models)",
             len(consensus), args.consensus_min_models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
