"""d_all percentile of each drug-target gene among the 10 control genes, per patient: tumor model with its pFBA reference,
tumor model with 100 OptGP references (175 chain), normal model with 100 OptGP references. Read-only join of two 174 outputs.
Usage: 191_poc_reference_comparison.py <poc_percentile_gated.tsv> <poc_percentile_tumor_optgp_gated.tsv> <out.tsv>"""
import sys, pandas as pd

a, b, out = sys.argv[1:4]
rd = lambda p: pd.read_csv(p, sep="\t", comment="#")
A, B = rd(a), rd(b)
A, B = A[A.readout == "d_all"], B[B.readout == "d_all"]
A["patient"] = A.model.str[:12]; B["patient"] = B.model.str[:12]
fmt = lambda r: "NA" if pd.isna(r.percentile) else f"{r.percentile:g} [{r.percentile_min:g}, {r.percentile_max:g}] n={int(r.n_references_with_percentile)}"
cols = {"tumor_pFBA": A[A.side == "tumor"], "tumor_OptGP": B[B.side == "normal"], "normal_OptGP": A[A.side == "normal"]}
assert all(B[B.side == "normal"].model.str.endswith("-01A")) and all(A[A.side == "normal"].model.str.endswith("-11A"))
assert (A[A.side == "tumor"].set_index(["model", "gene"]).percentile.fillna(-1).sort_index()
        == B[B.side == "tumor"].set_index(["model", "gene"]).percentile.fillna(-1).sort_index()).all(), "tumor pFBA rows differ between files"
rows = []
for k, d in cols.items():
    for _, r in d.iterrows():
        rows.append({"patient": r.patient, "symbol": r.symbol, "column": k, "value": fmt(r)})
t = pd.DataFrame(rows).pivot_table(index=["patient", "symbol"], columns="column", values="value", aggfunc="first")[list(cols)]
assert len(t) == 24, len(t)
with open(out, "w") as f:
    f.write(f"# source: {a} (tumor pFBA, normal OptGP); {b} (tumor model, OptGP references, side label 'normal' in that file)\n"
            "# readout d_all; percentile among 10 control genes, median [min, max] over references, n = references with a percentile; NA = no percentile\n"
            "# script_python/191_poc_reference_comparison.py, 2026-11-04; n rows 24 (3 patients x 8 drug-target genes)\n")
    t.to_csv(f, sep="\t")
print(t.to_string())
