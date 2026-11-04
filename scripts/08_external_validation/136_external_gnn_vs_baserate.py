"""External validation set: how much of the GNN ranking is the colorectal per-gene base rate (2026-11-03; user "spearman은 그냥 돌리면").
Moved into the project from the verifier's scratch script (verify_external_20261101.md:110-112, gnn_vs_baserate.py); arithmetic unchanged.
Per graph (A', B) and labelled cell line: GNN = mean probability over the 15 cell-line-disjoint models (pred_v3/<graph>_line_*/fold*/
pred_external.tsv); base rate = mean dependency of the gene over the 59 colorectal cell lines (dataset_v2 samples.tsv; absent genes get
the global mean; same definition as 85b:49); HGB = hgb_nomed_prob averaged over the 5 cell-line folds (flat_dsv2/line); MOMA = 1 - growth
ratio clipped to [0, 1]. Spearman between scores; AUROC of GNN and base rate within base rate < 0.5 and >= 0.5.
Usage: 136_external_gnn_vs_baserate.py <out.tsv> [<external result root>]   (2026-11-03: optional root, default
result/external_20261030_tolpatch; used for set 2 result/external_gi_normal_20261103, design 6(g). Arithmetic unchanged; backup
_backup/20261103_external/136_external_gnn_vs_baserate.py.bak)
"""
import sys, glob, datetime
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

R = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"; E = sys.argv[2] if len(sys.argv) > 2 else f"{R}/result/external_20261030_tolpatch"; out = sys.argv[1]
assert not __import__("os").path.exists(out), f"{out} exists (no overwrite)"
tr = pd.read_csv(f"{R}/result/gnn_20261030_tolpatch/dataset_v2/samples.tsv", sep="\t", comment="#")
BR = tr.groupby("symbol").dependency.mean()
fl = {k: pd.read_csv(f"{E}/flat_dsv2/line/fold{k}.tsv", sep="\t", comment="#").set_index(["line", "gene"]) for k in range(5)}
def auc(d, s):
    y = d.label == 1
    return roc_auc_score(y, d[s]) if 0 < y.sum() < len(d) else np.nan
rows = []
for g in ("A", "B"):
    fs = sorted(glob.glob(f"{E}/pred_v3/{g}_line_*/fold*/pred_external.tsv"))
    ds = [pd.read_csv(f, sep="\t", comment="#") for f in fs]; ref = ds[0].copy()
    ref["gnn"] = np.mean([d.gnn_prob.to_numpy() for d in ds], 0); ref["br"] = ref.symbol.map(BR).fillna(tr.dependency.mean())
    idx = list(zip(ref.line, ref.gene)); ref["hgb0"] = np.mean([fl[k].hgb_nomed_prob.reindex(idx).to_numpy() for k in range(5)], 0)
    ref["moma"] = 1 - ref.growth_ratio.clip(0, 1)
    for ln, x in ref[ref.has_label].groupby("line"):
        lo, hi = x[x.br < 0.5], x[x.br >= 0.5]
        rows.append(dict(graph=g, line=ln, n_models=len(fs), n=len(x), rho_gnn_baserate=spearmanr(x.gnn, x.br)[0],
                         rho_hgb_baserate=spearmanr(x.hgb0, x.br)[0], rho_gnn_moma=spearmanr(x.gnn, x.moma)[0], rho_gnn_hgb=spearmanr(x.gnn, x.hgb0)[0],
                         n_lo=len(lo), pos_lo=int((lo.label == 1).sum()), AUROC_gnn_lo=auc(lo, "gnn"), AUROC_baserate_lo=auc(lo, "br"),
                         n_hi=len(hi), pos_hi=int((hi.label == 1).sum()), AUROC_gnn_hi=auc(hi, "gnn"), AUROC_baserate_hi=auc(hi, "br")))
r = pd.DataFrame(rows).round(3)
with open(out, "w") as f:
    f.write(f"# source: {E}/pred_v3 (cell-line-disjoint models, no medium, protocol v3), {E}/flat_dsv2/line, base rate from dataset_v2 samples.tsv\n"
            "# GNN = mean probability over all models of the graph; lo/hi = per-gene base rate < 0.5 / >= 0.5; label = DepMap dependency >= 0.5;\n"
            f"# cell lines without a DepMap label excluded (as 85b): {sorted(ref.line[~ref.has_label].unique())}; Spearman = rank correlation per cell line\n"
            f"# script: script_python/136_external_gnn_vs_baserate.py; date {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
    r.to_csv(f, sep="\t", index=False)
print(r.to_string(index=False))
