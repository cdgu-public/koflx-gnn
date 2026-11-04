"""Read-only (2026-11-04): GNN ensemble probability of the POC drug-target genes per patient model (pFBA references, MOMA),
both encodings, with the within-model rank among all genes scored in that model (percentile of genes with lower probability)
and the MOMA growth ratio r. Input: result/patient_poc_20261103/gnn_ensemble.tsv (script 146; cell-line-disjoint models, 15 per encoding)."""
import pandas as pd
P = "/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/patient_poc_20261103/gnn_ensemble.tsv"
g = pd.read_csv(P, sep="\t", comment="#")
cols = {"gnn_mean_A_line_nomed_fixedcos30": "MFG", "gnn_mean_B_line_nomed_meanmax_fixedcos30": "BMRG"}
for c, n in cols.items():
    g[n + "_pct"] = g.groupby("line")[c].rank(pct=True) * 100
T = ["DHODH", "FASN", "GART", "HMGCR", "RRM1", "RRM2", "TYMS"]
out = g[g.symbol.isin(T)][["line", "symbol", "growth_ratio", *cols, "MFG_pct", "BMRG_pct"]].rename(columns=cols)
pd.set_option("display.width", 200); pd.set_option("display.float_format", "{:.3f}".format)
print(out.sort_values(["line", "symbol"]).to_string(index=False))
print("\nper model: n genes, share with prob > 0.5, share with r < 0.05, MFG prob median of r<0.05 vs r>=0.05 genes")
for l, d in g.groupby("line"):
    leth = d.growth_ratio < 0.05
    print(l, len(d), f"{(d[list(cols)[0]] > 0.5).mean():.2f}", f"{leth.mean():.2f}",
          f"{d.loc[leth, list(cols)[0]].median():.3f}", f"{d.loc[~leth, list(cols)[0]].median():.3f}")
