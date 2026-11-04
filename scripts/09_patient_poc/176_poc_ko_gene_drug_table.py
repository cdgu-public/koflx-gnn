"""Knockout gene list of the patient POC with the drugs that led to each target (table for the manuscript).

Rows come unchanged from the drug table; a drug is listed under every POC gene it targets.
Usage: 176_poc_ko_gene_drug_table.py OUT.tsv
"""
import sys
from datetime import datetime
import pandas as pd

DRUGS = "result/external_plan_20261103/drug_targets_metabolic_20261103.tsv"
GENES = ["TYMS", "GART", "RRM1", "RRM2", "DHODH", "HMGCR", "FASN"]   # GLS removed by user ruling 2026-11-04 12:3x
COLS = ["drug", "status_class", "indication_status", "mechanism", "reference"]

d = pd.read_csv(DRUGS, sep="\t", comment="#")
targets = d.target_genes.fillna("").str.replace(",", ";").str.replace(" ", "").str.split(";")
rows = []
for g in GENES:
    hit = d[targets.apply(lambda t: g in t)]
    ensg = sorted(set(e for s, t in zip(hit.target_ensg.fillna(""), targets[hit.index])
                      for sym, e in zip(t, str(s).replace(",", ";").replace(" ", "").split(";")) if sym == g))
    if hit.empty:
        rows.append(dict(gene=g, ensembl="", **{c: "" for c in COLS}))
    for _, r in hit.iterrows():
        rows.append(dict(gene=g, ensembl=";".join(ensg), **{c: r[c] for c in COLS}))
out = pd.DataFrame(rows)
with open(sys.argv[1], "w") as fh:
    fh.write(f"# source: {DRUGS} (rows unchanged); script_python/176_poc_ko_gene_drug_table.py; {datetime.now():%F %T} KST\n")
    fh.write(f"# knockout genes (n={len(GENES)}): {', '.join(GENES)}; GLS removed (user 2026-11-04: knockout leaves the fluxome unchanged)\n")
    fh.write(f"# n rows {len(out)}; genes without a drug row keep one empty row\n")
    out.to_csv(fh, sep="\t", index=False)
print(out[["gene", "drug", "status_class", "reference"]].to_string())
