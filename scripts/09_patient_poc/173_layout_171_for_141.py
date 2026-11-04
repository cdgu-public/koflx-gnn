"""Re-lay script 171 output into the input layout of script 141 (as 158 for normals, 167 for tumors), 2026-11-04. Values unchanged.
Only finished ranges (meta_s*.json present) are used; the samples included are written in the header.
normal: summary.tsv (reference_id = optgp_s<i>, gene, status, moma_objective, growth, n_disabled, kept_method, abs_obj_diff),
        flux_matrix.parquet (MultiIndex (reference_id, gene), kept solutions), reference.parquet (one column per reference_id).
tumor:  summary.tsv indexed by gene, flux_matrix.parquet indexed by gene, reference.parquet = the 171 pFBA reference (column 'reference').
Refuses to overwrite. Usage: 173_layout_171_for_141.py <171 side root (…/tumor or …/normal)> <out root> {tumor|normal}
"""
import glob, json, os, re, sys
from datetime import datetime
import numpy as np, pandas as pd

src, dst, side = sys.argv[1:4]
KEEP = ["status", "moma_objective", "growth", "n_disabled", "kept_method", "abs_obj_diff"]
for d in sorted(glob.glob(f"{src}/*/")):
    line = os.path.basename(d.rstrip("/")); out = f"{dst}/{line}"
    if os.path.exists(f"{out}/summary.tsv"):
        sys.exit(f"exists, not overwritten: {out}")
    tags = [re.sub(r"^meta_(.*)\.json$", r"\1", os.path.basename(f)) for f in sorted(glob.glob(f"{d}/meta_s*.json"))]
    assert not any("__" in t for t in tags), f"{line}: no-overwrite duplicates"
    if not tags:
        print(f"{line}: no finished range, skipped"); continue
    metas = [json.load(open(f"{d}/meta_{t}.json")) for t in tags]
    summ = pd.concat([pd.read_csv(f"{d}/summary_{t}.tsv", sep="\t", comment="#") for t in tags])
    idx = sorted(summ.sample_idx.unique()); ng = summ.gene.nunique()
    assert len(summ) == len(idx) * ng and not summ.duplicated(["sample_idx", "gene"]).any()
    fl = []
    for s in idx:
        f = pd.read_parquet(f"{d}/flux_{s}.parquet")
        assert set(f.index) == set(summ[(summ.sample_idx == s) & (summ.status == "optimal")].gene), f"{line} s{s}: flux rows != optimal"
        if side == "normal":
            f.index = pd.MultiIndex.from_arrays([[f"optgp_s{s}"] * len(f), f.index], names=["reference_id", "gene"])
        fl.append(f)
    flux = pd.concat(fl); assert (flux.dtypes == np.float64).all()
    if side == "normal":
        srcs = {m["args"]["samples"] for m in metas}; assert len(srcs) == 1, srcs; sp = srcs.pop()
        S = pd.read_parquet(sp if sp.startswith("/") else f"/data/workspace/cdgu/Winery/seeding/CRC_GEM/{sp}")
        ref = S.loc[idx, list(flux.columns)].T.astype("float64"); ref.columns = [f"optgp_s{s}" for s in idx]
        summ = summ.assign(reference_id=[f"optgp_s{s}" for s in summ.sample_idx])[["reference_id", "gene"] + KEEP]
        refdesc = f"OptGP samples {sp}"
    else:
        assert idx == [0] and metas[0]["pfba"].get("pfba_status") == "optimal", f"{line}: no pFBA reference"
        rf = metas[0]["pfba"]["reference_file"]; ref = pd.read_parquet(rf); assert list(ref.columns) == ["reference"]
        summ = summ[["gene"] + KEEP].set_index("gene"); refdesc = f"pFBA {rf}"
    os.makedirs(out)
    with open(f"{out}/summary.tsv", "w") as fh:
        fh.write(f"# source: {d} (script 171 summary_*.tsv, flux_*.parquet kept solutions), reference = {refdesc}; ranges {tags}\n"
                 f"# re-laid by script_python/173_layout_171_for_141.py {datetime.now().isoformat(timespec='seconds')} KST; values unchanged; "
                 f"n_references {len(idx)} (sample_idx {idx[0]}-{idx[-1]}), n_genes {ng}, n_optimal {len(flux)}; non-optimal rows kept (NaN)\n")
        summ.to_csv(fh, sep="\t", index=(side == "tumor"))
    flux.to_parquet(f"{out}/flux_matrix.parquet"); ref.to_parquet(f"{out}/reference.parquet")
    print(f"{line}: references {len(idx)} genes {ng} rows {len(summ)} optimal {len(flux)} reactions {flux.shape[1]}")
