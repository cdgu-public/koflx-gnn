"""Mean-probability GNN ensemble on the patient POC models (2026-11-03; user scope update 2026-11-03: GNN predictions on all 6
patient models, prediction only). Ensemble rule as 85b:99-105: per model family (A_line_nomed_fixedcos30 = graph A, B_line_nomed_meanmax_
fixedcos30 = graph B; 3 seeds x 5 folds = 15 models each), mean gnn_prob over all models on the (line, gene) index; every model must
cover the same samples (asserted). No labels exist for patients (label -1), so no metric is computed.
Input: <pred-root>/<run>/fold<k>/pred_external.tsv (script 99). Output: --out TSV with # header (runs, n models, n rows, date).
Env: crcgem. Refuses to overwrite.
"""
import argparse, glob, os
from datetime import datetime
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--pred-root", required=True); ap.add_argument("--out", required=True)
args = ap.parse_args()
assert not os.path.exists(args.out), f"{args.out} exists (no overwrite)"
fam = {"A_line_nomed_fixedcos30": [], "B_line_nomed_meanmax_fixedcos30": []}
for f in sorted(glob.glob(f"{args.pred_root}/*/fold*/pred_external.tsv")):
    run = f.split("/")[-3]
    k = [x for x in fam if run.startswith(x + "_s") and run.endswith("_tolpatch")]
    assert len(k) == 1, run
    fam[k[0]].append((f, pd.read_csv(f, sep="\t", comment="#")))
assert all(len(v) == 15 for v in fam.values()), {k: len(v) for k, v in fam.items()}
base = fam["A_line_nomed_fixedcos30"][0][1].set_index(["line", "gene"]).sort_index()
idx = base.index
out = base[["symbol", "growth_ratio", "label", "has_label"]].copy()
for k, lst in fam.items():
    P = np.stack([d.set_index(["line", "gene"]).gnn_prob.reindex(idx).to_numpy() for _, d in lst])
    assert np.isfinite(P).all() and all(len(d) == len(idx) for _, d in lst), f"{k}: models do not cover the same samples"
    out[f"gnn_mean_{k}"] = P.mean(0); out[f"gnn_min_{k}"] = P.min(0); out[f"gnn_max_{k}"] = P.max(0)
with open(args.out, "w") as fh:
    fh.write(f"# source: script_python/146_patient_gnn_ensemble.py; predictions {args.pred_root}/<run>/fold<k>/pred_external.tsv (script 99)\n")
    fh.write(f"# ensemble: mean gnn_prob over 15 models per family (3 seeds x 5 folds), rule of 85b:99-105; min/max over the same models\n")
    fh.write(f"# labels: none (TCGA tissue, no DepMap row; label -1); no metric computed; date {datetime.now().isoformat(timespec='seconds')} KST; "
             f"n rows {len(out)}, samples {sorted(set(idx.get_level_values(0)))}\n")
    out.to_csv(fh, sep="\t")
print(out.groupby(level=0).size().to_string()); print("written", args.out)
