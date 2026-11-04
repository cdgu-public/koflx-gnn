"""Re-lay the per-sample MOMA output of script 142 into the input layout of script 141 (2026-11-03, patient POC normals).
No value is changed: rows and fluxes are copied; only file layout and the reference id label change.
In  (--in-root/<line>/, written by 142): summary_s<a>-<b>.tsv (sample_idx, gene, status, moma_objective, growth, n_disabled, seconds),
    flux_<sample_idx>.parquet (optimal genes x model reactions, float32), meta_s<a>-<b>.json (args.samples = OptGP sample parquet).
Out (--out-root/<line>/, read by 141): summary.tsv (reference_id, gene, status, moma_objective, growth, n_disabled),
    flux_matrix.parquet (MultiIndex (reference_id, gene)), reference.parquet (index reactions, one column per reference_id).
reference_id = 'optgp_s<sample_idx>'. Files with a '__' (no-overwrite) suffix are refused, so each sample has one source.
Checks: one sample source each; rows = samples x genes; flux rows = optimal rows; flux columns = sample columns; refuses to overwrite.
Usage: python 158_layout_142_for_141.py --in-root result/patient_poc_20261103/moma_normal --out-root result/patient_poc_20261103/moma_normal_141
"""
import argparse, glob, json, os, sys
from datetime import datetime
import pandas as pd

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
ap = argparse.ArgumentParser()
ap.add_argument("--in-root", required=True)
ap.add_argument("--out-root", required=True)
a = ap.parse_args()
ab = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
for d in sorted(glob.glob(f"{ab(a.in_root)}/*/")):
    line = os.path.basename(d.rstrip("/")); out = f"{ab(a.out_root)}/{line}"
    if os.path.exists(f"{out}/summary.tsv"):
        sys.exit(f"exists, not overwritten: {out}")
    assert not glob.glob(f"{d}/*__*"), f"{line}: no-overwrite duplicates present, resolve first"
    summ = pd.concat([pd.read_csv(f, sep="\t", comment="#") for f in sorted(glob.glob(f"{d}/summary_s*.tsv"))])
    metas = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/meta_s*.json"))]
    src = {m["args"]["samples"] for m in metas}; assert len(src) == 1, src
    n_genes = summ.gene.nunique(); idx = sorted(summ.sample_idx.unique())
    assert len(summ) == len(idx) * n_genes and not summ.duplicated(["sample_idx", "gene"]).any(), f"{line}: rows != samples x genes"
    S = pd.read_parquet(ab(src.pop()))
    fl = []
    for s in idx:
        f = pd.read_parquet(f"{d}/flux_{s}.parquet")
        opt = summ[(summ.sample_idx == s) & (summ.status == "optimal")].gene
        assert set(f.index) == set(opt), f"{line} sample {s}: flux rows != optimal genes"
        f.index = pd.MultiIndex.from_arrays([[f"optgp_s{s}"] * len(f), f.index], names=["reference_id", "gene"]); fl.append(f)
    flux = pd.concat(fl); assert list(flux.columns) == list(S.columns) or set(flux.columns) == set(S.columns)
    ref = S.loc[idx, list(flux.columns)].T; ref.columns = [f"optgp_s{s}" for s in idx]
    summ = summ.assign(reference_id=[f"optgp_s{s}" for s in summ.sample_idx])
    os.makedirs(out)
    with open(f"{out}/summary.tsv", "w") as fh:
        fh.write(f"# source: {d} (script 142 summary_s*.tsv, flux_*.parquet), reference = OptGP samples {src or ''}{metas[0]['args']['samples']}\n"
                 f"# re-laid by script_python/158_layout_142_for_141.py {datetime.now().isoformat(timespec='seconds')} KST; values unchanged; "
                 f"n_samples {len(idx)}, n_genes {n_genes}, n_optimal {int((summ.status == 'optimal').sum())}; non-optimal rows kept (NaN)\n")
        summ[["reference_id", "gene", "status", "moma_objective", "growth", "n_disabled"]].to_csv(fh, sep="\t", index=False)
    flux.to_parquet(f"{out}/flux_matrix.parquet"); ref.to_parquet(f"{out}/reference.parquet")
    print(f"{line}: samples {len(idx)} genes {n_genes} rows {len(summ)} optimal {len(flux)} reactions {flux.shape[1]}")
