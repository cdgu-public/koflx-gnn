"""Comparison table of the protocol v3 tolpatch grid (2026-10-31): every model type side by side per graph, split, stratum.
Reads summary.tsv of the two 85 evaluations of the same grid runs, one with the with-medium HGB comparator (flat root
flat_v2_dsv2) and one with the no-medium HGB comparator (flat_v2_expr_dsv2/fluxonly). GNN, MOMA, per-gene base rate
(85 'prior') and line-propensity prior (85 'lineprior') are taken from the no-medium file and asserted equal in both.
Values are fold means over 5 folds x 3 seeds as written by 85 (gnn_mean, comp_mean). No selection, no re-computation.
Usage: 114_grid_comparison_table.py <eval_withmedHGB dir> <eval_nomedHGB dir> <out.tsv>
"""
import sys, datetime
import pandas as pd

wm, nm, out = sys.argv[1:4]
A = pd.read_csv(f"{wm}/summary.tsv", sep="\t"); B = pd.read_csv(f"{nm}/summary.tsv", sep="\t")
key = ["graph", "split", "variant", "stratum", "metric"]
rows = []
for k, g in B.groupby(key):
    a = A.set_index(key + ["comparator"]); r = dict(zip(key, k)); r["n"] = int(g.n.iloc[0]); r["n_folds"] = int(g.n_folds.iloc[0])
    r["n_seeds"] = int(g.n_seeds.iloc[0]); r["GNN"] = g.gnn_mean.iloc[0]
    for _, x in g.iterrows():
        c = x.comparator.split("_")[-1] if not x.comparator.endswith("hgb_prob") else "hgb_prob"
        name = {"moma": "MOMA", "prior": "per-gene base rate", "lineprior": "line-propensity prior", "random": "random",
                "hgb_prob": "HGB no medium"}[c]
        r[name] = x.comp_mean
        ka = k + (x.comparator,)
        if ka in a.index:
            y = a.loc[ka]
            assert abs(y.gnn_mean - x.gnn_mean) < 1e-9, ka
            if c == "hgb_prob":
                r["HGB with medium"] = y.comp_mean
            else:
                assert abs(y.comp_mean - x.comp_mean) < 1e-9, ka
    rows.append(r)
T = pd.DataFrame(rows)
T["metric"] = T.metric.str.replace("_gnn$", "", regex=True)
cols = key + ["n", "n_folds", "n_seeds", "GNN", "MOMA", "HGB no medium", "HGB with medium", "per-gene base rate", "line-propensity prior", "random"]
T = T[[c for c in cols if c in T.columns]].sort_values(["metric", "split", "stratum", "graph"]).round(3)
with open(out, "w") as f:
    f.write(f"# sources: {wm}/summary.tsv (HGB with medium), {nm}/summary.tsv (HGB no medium, all other columns)\n"
            "# runs: result/gnn_20261030_tolpatch/runs_v2_sel_grid (protocol v3 fixed-epoch, no-medium GNN, dataset_v2 md5 c7a4c4849bdfe4b696aaf517d7afacbb)\n"
            "# values: fold mean over 5 test folds x 3 seeds (script 85 gnn_mean / comp_mean); GNN, MOMA and priors asserted equal across both files\n"
            "# per-gene base rate = 85 'prior'; line-propensity prior = 85 'lineprior'; empty = comparator not computed by 85 for that metric\n"
            f"# script: script_python/114_grid_comparison_table.py; date {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
    T.to_csv(f, sep="\t", index=False)
print(T[T.metric.isin(["AUROC", "AUPRC", "within_gene_AUROC"])].to_string(index=False))
