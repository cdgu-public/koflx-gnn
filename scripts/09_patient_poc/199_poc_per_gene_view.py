"""Read-only per-gene view of the patient POC (2026-11-04): d_all median per (model, reference) for drug targets and controls,
and the GNN ensemble probability (gnn_ensemble.tsv, script 146, pFBA-reference fluxomes of all six models) for the same genes.
Inputs: distance/poc_percentile_gated_20261104.tsv, distance/poc_percentile_tumor_optgp_gated_20261104.tsv, gnn_ensemble.tsv."""
import pandas as pd
P = "/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/patient_poc_20261103/"
a = pd.read_csv(P + "distance/poc_percentile_gated_20261104.tsv", sep="\t", comment="#")
b = pd.read_csv(P + "distance/poc_percentile_tumor_optgp_gated_20261104.tsv", sep="\t", comment="#")
a["ref"] = a.side.map({"tumor": "pFBA", "normal": "OptGP"}); b = b[b.side == "normal"].copy(); b["ref"] = "OptGP"  # keep only tumor-model OptGP rows (file repeats tumor pFBA rows)
d = pd.concat([a, b]); d = d[d.readout == "d_all"]
d["tissue"] = d.model.str[-3:]; d["patient"] = d.model.str[:12]
w = d.pivot_table(index=["patient", "gene"], columns=["tissue", "ref"], values="value_target_median", aggfunc="first")
w.columns = [f"{t}_{r}" for t, r in w.columns]
g = pd.read_csv(P + "gnn_ensemble.tsv", sep="\t", comment="#")
g["tissue"] = g.line.str[-3:]; g["patient"] = g.line.str[:12]
gg = g.pivot_table(index=["patient", "gene"], columns="tissue", values=["gnn_mean_A_line_nomed_fixedcos30", "growth_ratio"], aggfunc="first")
gg.columns = [("gnnMFG_" if c[0].startswith("gnn") else "r_") + c[1] for c in gg.columns]
sym = d.drop_duplicates("gene").set_index("gene").symbol
out = w.join(gg, how="left").reset_index(); out.insert(2, "symbol", out.gene.map(sym))
pd.set_option("display.width", 250); pd.set_option("display.float_format", "{:.3g}".format)
print(out.drop(columns="gene").sort_values(["patient", "symbol"]).to_string(index=False))
print("GNN MFG mean prob over all genes in gnn_ensemble, by sample:")
print(g.groupby("line").gnn_mean_A_line_nomed_fixedcos30.describe()[["count", "mean", "50%"]].to_string())
