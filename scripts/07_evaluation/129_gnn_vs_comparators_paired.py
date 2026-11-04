"""Paired GNN - comparator AUROC differences on the same test folds (2026-11-02; user request: GNN vs HGB on rel>=0.2 with interval).

Per (seed, fold): AUROC_gnn - AUROC_<comparator> on the same test fold and stratum (85 columns AUROC_moma, AUROC_hgb_prob), averaged
over seeds per fold, 95% t interval over the fold means (df = n_folds - 1), as 118. Comparators: MOMA, HGB (no medium, fluxonly).
Usage: 129_gnn_vs_comparators_paired.py <per_fold_seed.tsv> <graph> <variant> <out.tsv> [--hgb-label "<text>"]
--hgb-label (added 2026-11-02): names the HGB variant of the source eval in the header (default "no medium (AUROC_hgb_prob, fluxonly)"; use
"with medium (AUROC_hgb_prob, flat_v2_dsv2)" for eval_grid_withmedHGB per_fold_seed.tsv). Arithmetic unchanged.
"""
import sys, datetime
import numpy as np, pandas as pd
from scipy import stats

src, graph, variant, out = sys.argv[1:5]
hgb_label = sys.argv[sys.argv.index("--hgb-label") + 1] if "--hgb-label" in sys.argv else "no medium (AUROC_hgb_prob, fluxonly)"
d = pd.read_csv(src, sep="\t", comment="#")
d["variant"] = d.variant.fillna("")
d = d[(d.graph == graph) & (d.split == "gene") & (d.variant == variant)]
rows = []
for st, g in d.groupby("stratum"):
    for comp in ("AUROC_moma", "AUROC_hgb_prob"):
        fd = (g.AUROC_gnn - g[comp]).groupby(g.fold).mean().to_numpy(); n = len(fd)
        ci = stats.t.interval(0.95, n - 1, loc=fd.mean(), scale=fd.std(ddof=1) / np.sqrt(n)) if n > 1 else (np.nan, np.nan)
        rows.append(dict(stratum=st, comparator=comp.replace("AUROC_", ""), n_pairs=len(g), n_folds=n, n_test_per_seed=int(g["n"].sum() / g.seed.nunique()),
                         AUROC_gnn=g.AUROC_gnn.mean(), AUROC_comp=g[comp].mean(), diff=fd.mean(), ci_lo=ci[0], ci_hi=ci[1],
                         folds_gnn_higher=int((fd > 0).sum()), p_t=stats.ttest_1samp(fd, 0).pvalue if n > 1 else np.nan))
r = pd.DataFrame(rows).round(4)
with open(out, "w") as f:
    f.write(f"# source: {src}\n# filter: graph {graph}, gene-disjoint split, variant '{variant}', all seeds\n"
            "# diff = AUROC_gnn - AUROC_comparator per (seed, fold), averaged over seeds per fold; 95% t over fold means (df = n_folds - 1)\n"
            f"# comparators: MOMA (AUROC_moma), HGB {hgb_label}; intervals descriptive (folds share training data)\n"
            f"# script: script_python/129_gnn_vs_comparators_paired.py; date {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
    r.to_csv(f, sep="\t", index=False)
print(r.to_string(index=False))
