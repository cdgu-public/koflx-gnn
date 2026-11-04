"""Read-only summary of the patient POC by readout (2026-11-04), three blocks:
(A) GNN: within-model percentile of each drug-target gene among ALL genes scored in that model (gnn_ensemble.tsv, script 146,
    pFBA references, cell-line-disjoint ensembles; percentile = rank(pct=True)*100, higher = more essential-like), plus the same
    percentile restricted to genes with an essential-like DepMap profile in the 59 colorectal lines is NOT computed (no label on TCGA).
(B) DepMap reference for the same genes in the 59 colorectal cell lines (dataset_v2/samples.tsv): median dependency probability and
    share of cell lines with dependency >= 0.5, and the per-gene percentile of that share among all genes of the sample set.
(C) KEGG pathway-map distance d_sym (141/174, route kegg_reaction_or_ec, maps map01100-map01299 excluded): per drug target and side,
    number of maps with a percentile and the median over those maps of the within-model percentile among control genes.
Outputs printed; run with > result/patient_poc_20261103/203_target_summary.log."""
import pandas as pd
R = "/data/workspace/cdgu/Winery/seeding/CRC_GEM/"
T = ["DHODH", "FASN", "GART", "HMGCR", "RRM1", "RRM2", "TYMS"]
pd.set_option("display.width", 220); pd.set_option("display.float_format", "{:.2f}".format)
g = pd.read_csv(R + "result/patient_poc_20261103/gnn_ensemble.tsv", sep="\t", comment="#")
A, B = "gnn_mean_A_line_nomed_fixedcos30", "gnn_mean_B_line_nomed_meanmax_fixedcos30"
for c, n in ((A, "MFG"), (B, "BMRG")):
    g[n] = g.groupby("line")[c].rank(pct=True) * 100
g["tissue"] = g.line.str[-3:].map({"01A": "tumor", "11A": "normal"})
print("(A) GNN percentile among all genes scored in the model (pFBA reference)")
print(g[g.symbol.isin(T)].pivot_table(index="symbol", columns="tissue", values=["MFG", "BMRG"], aggfunc="median").to_string())
print("n genes per model:", g.groupby("line").size().to_dict())
s = pd.read_csv(R + "result/gnn_20261030_tolpatch/dataset_v2/samples.tsv", sep="\t", comment="#")
s = s[s.has_label & s.in_gnn]
d = s.groupby("symbol").agg(dep_median=("dependency", "median"), essential_share=("label", "mean"), n_lines=("line", "nunique"))
d["essential_share_pct_among_genes"] = d.essential_share.rank(pct=True) * 100
print("\n(B) DepMap, 59 colorectal cell lines (labeled knockouts in the GNN sample set)")
print(d.loc[[t for t in T if t in d.index]].to_string()); print("genes in sample set:", len(d))
p = pd.concat([pd.read_csv(R + f"result/patient_poc_20261103/distance/{f}", sep="\t", comment="#") for f in
               ("poc_percentile_gated_20261104.tsv",)])
q = pd.read_csv(R + "result/patient_poc_20261103/distance/poc_percentile_tumor_optgp_gated_20261104.tsv", sep="\t", comment="#")
q = q[q.side == "normal"].assign(side="tumor_OptGP")
p["side"] = p.side.map({"tumor": "tumor_pFBA", "normal": "normal_OptGP"})
k = pd.concat([p, q]); k = k[(k.readout == "d_sym") & (k.route == "kegg_reaction_or_ec") & k.percentile.notna()]
print("\n(C) KEGG map distance d_sym: median over maps of the percentile among controls [n maps], median over patients")
c = k.groupby(["symbol", "side", "model"]).agg(pm=("percentile", "median"), nm=("kegg_map", "nunique")).reset_index()
print(c.groupby(["symbol", "side"]).agg(pct_median=("pm", "median"), n_maps=("nm", "median")).unstack("side").to_string())
top = k[k.side == "tumor_pFBA"].groupby(["symbol", "kegg_map_name"]).percentile.median().reset_index()
print("\n(C2) maps where the drug target is at percentile 100 in all tumor pFBA models, per target (first 6)")
for t, x in top[top.percentile >= 100].groupby("symbol"):
    print(t, "|", "; ".join(x.kegg_map_name.head(6)), f"| n={len(x)}")
# (D) same gene pool: DepMap essential share percentile among the genes scored in each patient model (consistency with block A)
print("\n(D) DepMap essential-share percentile among the genes scored in each patient model (same pool as A), median over models by tissue")
rows = []
for l, x in g.groupby("line"):
    pool = d.loc[d.index.intersection(x.symbol)].essential_share.rank(pct=True) * 100
    for t in T:
        if t in pool.index and t in set(x.symbol):
            rows.append(dict(line=l, tissue=x.tissue.iloc[0], symbol=t, depmap_pct=pool[t], n_pool=len(pool)))
r = pd.DataFrame(rows); print(r.pivot_table(index="symbol", columns="tissue", values="depmap_pct", aggfunc="median").to_string())
print("pool sizes:", r.groupby("line").n_pool.first().to_dict())
