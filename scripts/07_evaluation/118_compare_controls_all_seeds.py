"""Seed-matched comparison of control runs vs the no-medium A' baseline over seeds 0-2 (2026-10-31; tolpatch controls).

Same comparison as 98_compare_seed0_controls.py, extended to all seeds: per stratum and fold, the variant - baseline AUROC
difference is taken on the same (seed, fold), averaged over seeds within each fold, and a fold t-interval (df = n_folds - 1)
is computed on the 5 fold means (folds are the independent test sets; seeds share folds). Per-seed fold means are also listed.
Usage: 118_compare_controls_all_seeds.py <per_fold_seed.tsv> <baseline variant> <out.tsv> [--graph A|B]
--graph (added 2026-11-04 for the graph B medium-input / stopping-rule runs, seed 0): default A, unchanged arithmetic.
"""
import sys, datetime
import numpy as np, pandas as pd
from scipy import stats

src, base, out = sys.argv[1:4]
G = sys.argv[sys.argv.index("--graph") + 1] if "--graph" in sys.argv else "A"
d = pd.read_csv(src, sep="\t")
d["variant"] = d.variant.fillna("")   # 2026-11-02: the with-medium variant has an empty label (85), read as NaN
d = d[(d.graph == G) & (d.split == "gene")]
rows = []
for st, g in d.groupby("stratum"):
    b = g[g.variant == base].set_index(["seed", "fold"])
    for v, x in g.groupby("variant"):
        x = x.set_index(["seed", "fold"])
        keys = sorted(set(x.index) & set(b.index))
        pair = pd.DataFrame({"diff": x.loc[keys, "AUROC_gnn"] - b.loc[keys, "AUROC_gnn"]}, index=pd.MultiIndex.from_tuples(keys))
        fd = pair.groupby(level=1)["diff"].mean().to_numpy(); n = len(fd)
        ci = stats.t.interval(0.95, n - 1, loc=fd.mean(), scale=fd.std(ddof=1) / np.sqrt(n)) if n > 1 else (np.nan, np.nan)
        xs = x.loc[keys]
        rows.append(dict(stratum=st, variant=v, n_pairs=len(keys), n_seeds=len({k[0] for k in keys}), n_folds=n,
                         n_test_per_seed=int(xs["n"].sum() / len({k[0] for k in keys})), AUROC_gnn=xs.AUROC_gnn.mean(),
                         AUROC_base=b.loc[keys, "AUROC_gnn"].mean(), AUROC_moma=xs.AUROC_moma.mean(), AUROC_hgb=xs.AUROC_hgb_prob.mean(),
                         diff_vs_base=fd.mean(), ci_lo=ci[0], ci_hi=ci[1], folds_better=int((fd > 0).sum()),
                         pairs_better=int((pair["diff"] > 0).sum()),
                         per_seed_diff=";".join(f"{v:.3f}" for v in pair.groupby(level=0)["diff"].mean())))
r = pd.DataFrame(rows).round(3)
with open(out, "w") as f:
    f.write(f"# source: {src}\n# filter: 85 column graph == {G} (MLP variants mlp81/mlphgb carry this tag but have no graph, 93_*), gene-disjoint split, seeds and folds present in both (pairs present in both variant and baseline)\n"
            f"# baseline variant: '{base}'; diff = variant - baseline on the same (seed, fold), averaged over seeds per fold;\n"
            "# 95% interval = t over the fold means (df = n_folds - 1); pairs_better counts (seed, fold) pairs with diff > 0\n"
            "# AUROC means over (seed, fold); MOMA / HGB = comparators on the same test folds (85 columns AUROC_moma, AUROC_hgb_prob; HGB no medium)\n"
            f"# script: script_python/118_compare_controls_all_seeds.py; date {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
    r.to_csv(f, sep="\t", index=False)
print(r.drop(columns=["per_seed_diff"]).to_string(index=False))
