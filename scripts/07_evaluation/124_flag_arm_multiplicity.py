"""Multiplicity adjustment of the flag-arm differences (2026-11-01; user request: FDR / other corrections besides Bonferroni).

D1 = disflag - baseline, D2 = wtonly - disflag per (seed, fold), averaged over seeds per fold (as 118); p = two-sided one-sample
t-test of the fold means against 0 (df = n_folds - 1, the same t used for the 118 intervals). Family = 14 tests (7 strata x D1, D2),
as in verify_flag_arm_20261101.md. Adjustments: Bonferroni, Holm, Benjamini-Hochberg (BH), Benjamini-Yekutieli (BY).
Usage: 124_flag_arm_multiplicity.py <per_fold_seed.tsv> <out.tsv>
"""
import sys, datetime
import numpy as np, pandas as pd
from scipy import stats

def adjust(p, method):   # statsmodels is not in crcgem_gnn; standard step-down / step-up formulas
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); ps = p[o]; k = np.arange(1, m + 1)
    if method == "bonferroni": a = np.minimum(ps * m, 1)
    elif method == "holm": a = np.minimum(np.maximum.accumulate(ps * (m - k + 1)), 1)
    else:
        c = 1.0 if method == "fdr_bh" else np.sum(1.0 / k)
        a = np.minimum(np.minimum.accumulate((ps * m * c / k)[::-1])[::-1], 1)
    r = np.empty(m); r[o] = a; return r

src, out = sys.argv[1:3]
d = pd.read_csv(src, sep="\t")
d = d[(d.graph == "A") & (d.split == "gene")]
V = dict(base="nomed_fixedcos30_tolpatch", flag="nomed_disflag_fixedcos30_tolpatch", wt="nomed_wtonly_fixedcos30_tolpatch")
rows = []
for st, g in d.groupby("stratum"):
    a = {k: g[g.variant == v].set_index(["seed", "fold"]).AUROC_gnn for k, v in V.items()}
    for name, x, y in (("D1 = disflag - baseline", "flag", "base"), ("D2 = wtonly - disflag", "wt", "flag")):
        keys = sorted(set(a[x].index) & set(a[y].index))
        fd = (a[x].loc[keys] - a[y].loc[keys]).groupby(level=1).mean().to_numpy()
        t, p = stats.ttest_1samp(fd, 0.0)
        rows.append(dict(stratum=st, contrast=name, n_folds=len(fd), mean_diff=fd.mean(), t=t, p_raw=p))
r = pd.DataFrame(rows)
for m, col in (("bonferroni", "p_bonferroni"), ("holm", "p_holm"), ("fdr_bh", "q_BH"), ("fdr_by", "q_BY")):
    r[col] = adjust(r.p_raw, m)
r = r.round(4)
with open(out, "w") as f:
    f.write(f"# source: {src}\n# filter: graph A, gene-disjoint split, seeds 0-2; diff per (seed, fold) averaged over seeds per fold (as 118)\n"
            "# p_raw: two-sided one-sample t on fold means (df = n_folds - 1); family = all rows (7 strata x 2 contrasts = 14)\n"
            "# adjustments: bonferroni / holm / BH / BY (step formulas in adjust(), statsmodels not installed); folds share training genes so tests are dependent\n"
            f"# script: script_python/124_flag_arm_multiplicity.py; date {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
    r.to_csv(f, sep="\t", index=False)
print(r.to_string(index=False))
