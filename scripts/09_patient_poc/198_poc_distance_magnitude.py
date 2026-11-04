"""Distance magnitude (d_all, median over references per gene) by side and reference method for the patient POC.
Inputs: result/patient_poc_20261103/distance/poc_percentile_gated_20261104.tsv (tumor pFBA, normal OptGP) and
poc_percentile_tumor_optgp_gated_20261104.tsv (tumor model, OptGP references; side label 'normal' in that file). Read-only."""
import pandas as pd
D = "/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/patient_poc_20261103/distance/"
a = pd.read_csv(D + "poc_percentile_gated_20261104.tsv", sep="\t", comment="#")
b = pd.read_csv(D + "poc_percentile_tumor_optgp_gated_20261104.tsv", sep="\t", comment="#")
b = b[b.side == "normal"].assign(side="tumor_OptGP")  # that file also repeats the tumor pFBA rows (side tumor); fixed 2026-11-04 23:0x
a["side"] = a.side.map({"tumor": "tumor_pFBA", "normal": "normal_OptGP"})
d = pd.concat([a, b]); d = d[d.readout == "d_all"]
print(d.groupby("side").value_target_median.describe().to_string())
print(d.groupby(["side", "model"]).value_target_median.agg(["min", "median", "max"]).to_string())
print(d.groupby("side").n_references_with_tie.agg(["mean", "max"]).to_string())
