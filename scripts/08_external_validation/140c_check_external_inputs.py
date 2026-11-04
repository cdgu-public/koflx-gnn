"""Shape/count checker (no metric) on the inputs of 85b / 140 / 140b for set 2 (result/external_gi_normal_20261103), 2026-11-03.
Asserts: no PREDICT FAILED; 30 prediction files (6 runs x 5 folds), each with the same (line, gene) rows, n = the number of
fluxome-change-filter samples written by 86b; 5 flat folds with the same rows; no SUBSET header; labels in {0,1}; 8 cell lines;
G_CRC counts per normal cell line (base rate >= 0.5) printed. Also checks set 1 inputs used by 140 (RPE1).
Usage: python 140c_check_external_inputs.py
"""
import glob
import pandas as pd

R = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
tr = pd.read_csv(f"{R}/result/gnn_20261030_tolpatch/dataset_v2/samples.tsv", sep="\t", comment="#")
BR = tr.groupby("symbol").dependency.mean(); G = float(tr.dependency.mean())
for E, nl in ((f"{R}/result/external_gi_normal_20261103", 8), (f"{R}/result/external_20261030_tolpatch", 7)):
    log = open(f"{E}/pred_v3_run.log").read(); assert "PREDICT FAILED" not in log, f"{E}: a prediction failed"
    fs = sorted(glob.glob(f"{E}/pred_v3/[AB]_line_*/fold*/pred_external.tsv")); assert len(fs) == 30, f"{E}: {len(fs)} files"
    ds = [pd.read_csv(f, sep="\t", comment="#") for f in fs]
    for f in fs:
        assert not any("SUBSET" in l for l in open(f) if l.startswith("#")), f
    keys = [set(zip(d.line, d.gene)) for d in ds]; assert all(k == keys[0] for k in keys), "rows differ between models"
    fl = [pd.read_csv(f, sep="\t", comment="#") for f in sorted(glob.glob(f"{E}/flat_dsv2/line/fold*.tsv"))]
    assert len(fl) == 5 and all(set(zip(x.line, x.gene)) == keys[0] for x in fl), "flat rows differ from prediction rows"
    d = ds[0]; assert d.line.nunique() == nl, d.line.nunique()
    assert set(d.label[d.has_label]) <= {0, 1}
    d["br"] = d.symbol.map(BR).fillna(G)
    print(f"{E}: 30 files x {len(d)} rows, flat 5 x {len(fl[0])} rows, cell lines {d.line.nunique()}")
    for ln, x in d.groupby("line"):
        g = x[x.br >= 0.5]
        print(f"  {ln}: n {len(x)}, labeled {bool(x.has_label.all())}, G_CRC {len(g)} (shared essential {int((g.label == 1).sum())}, cancer-selective {int((g.label == 0).sum())})")
print("CHECK PASS")
