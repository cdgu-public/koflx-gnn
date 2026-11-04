"""Measure what the disabled-reaction flag marks in the baseline A' node set (2026-11-01; design of the baseline-plus-flag arm,
result/gnn_20261030_tolpatch/design_flag_arm_20261101.md). Read-only on the dataset; no training.

For every GNN sample (knockouts that change the fluxome, the filter of 81b / 90) on dataset_v2 with the growth mask applied:
  n_dis        disabled scaffold reactions (scaffold_meta gene_rxns, mask removed)
  n_dis_base   of those, in the baseline node set (v_WT != 0 or v_KO != 0)          = the nodes the new arm can flag
  n_dis_wtonly of those, in the WT-only node set (v_WT != 0 or disabled) = n_dis    = the nodes the WT-only arm flags
  n_dis_ko0    of n_dis_base, with v_KO == 0 (the flag coincides with a zero KO flux)
  n_ko0_notdis non-disabled baseline nodes with v_WT != 0 and v_KO == 0 (reactions silenced downstream; the implicit zero is not the flag)
Output: one TSV with per-sample rows (# header with condition) and a summary on stdout.
Usage: python 120_measure_disabled_flag_coverage.py [--dataset D] [--out TSV]
Env: crcgem_gnn.
"""
import argparse, json, os, datetime
import numpy as np, pandas as pd

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=f"{ROOT}/result/gnn_20261030_tolpatch/dataset_v2")
ap.add_argument("--mask-file", default=f"{ROOT}/result/gnn_20261028/growth_mask_v2_20261029.json")
ap.add_argument("--out", default=f"{ROOT}/result/gnn_20261030_tolpatch/disabled_flag_coverage_20261101.tsv")
args = ap.parse_args()
D = args.dataset.rstrip("/")
if os.path.exists(args.out):
    raise SystemExit(f"REFUSED: output exists, never overwritten: {args.out}")
meta = json.load(open(f"{D}/scaffold_meta.json")); NR = len(meta["reactions"])
MASK = np.array(sorted(set(json.load(open(args.mask_file))["mask_index"])), int)
KEEP = np.ones(NR, bool); KEEP[MASK] = False
gene_rxns = meta["gene_rxns"]
rows = []
for f in sorted(os.listdir(f"{D}/lines")):
    ln = f[:-4]; z = np.load(f"{D}/lines/{ln}.npz")
    th = 1e-6 / float(z["mu_ref"])
    vwt = np.where(np.abs(z["vwt"]) > th, z["vwt"], 0).astype(np.float32)
    V = np.where(np.abs(z["V"]) > th, z["V"], 0).astype(np.float32)
    for i, g in enumerate(z["genes"]):
        vk = V[i]
        if np.abs(vk - vwt).max() <= th:
            continue   # not a GNN sample (fluxome-change filter of 81b / 90)
        dis = np.zeros(NR, bool); dis[np.asarray(gene_rxns.get(g, []), int)] = True; dis &= KEEP
        base = ((vwt != 0) | (vk != 0)) & KEEP
        rows.append(dict(line=ln, gene=g, n_dis=int(dis.sum()), n_dis_base=int((dis & base).sum()),
                         n_dis_wtonly=int((dis & KEEP).sum()), n_dis_ko0=int((dis & base & (vk == 0)).sum()),
                         n_dis_wt_active=int((dis & (vwt != 0)).sum()), n_ko0_notdis=int((base & ~dis & (vwt != 0) & (vk == 0)).sum()),
                         n_nodes_base=int(base.sum())))
d = pd.DataFrame(rows)
with open(args.out, "w") as fh:
    fh.write(f"# source: {D} (lines/*.npz, scaffold_meta.json gene_rxns); mask {args.mask_file}; flux threshold 1e-6 / mu_ref as 81b/90;\n"
             f"# sample filter: max|v_KO - v_WT| > threshold (GNN sample set); n = {len(d)}; script script_python/120_measure_disabled_flag_coverage.py;"
             f" date {datetime.datetime.now():%Y-%m-%d %H:%M}; no exclusions beyond the filter\n")
    d.to_csv(fh, sep="\t", index=False)
tot = d[["n_dis", "n_dis_base", "n_dis_ko0", "n_dis_wt_active", "n_ko0_notdis"]].sum()
print(f"samples {len(d)}; disabled reactions total {tot.n_dis}, in baseline node set {tot.n_dis_base} ({tot.n_dis_base / tot.n_dis:.4f}),"
      f" WT-active {tot.n_dis_wt_active} ({tot.n_dis_wt_active / tot.n_dis:.4f}); of baseline-node disabled, v_KO == 0: {tot.n_dis_ko0} ({tot.n_dis_ko0 / max(tot.n_dis_base, 1):.4f})")
print(f"samples with every disabled reaction inside the baseline node set: {(d.n_dis == d.n_dis_base).mean():.4f};"
      f" samples with >= 1 flagged baseline node: {(d.n_dis_base > 0).mean():.4f}; median n_dis {int(d.n_dis.median())}, median n_dis_base {int(d.n_dis_base.median())}")
print(f"non-disabled baseline nodes with v_WT != 0 and v_KO == 0 per sample: median {d.n_ko0_notdis.median():.0f}, mean {d.n_ko0_notdis.mean():.1f}, "
      f"share of samples with >= 1 such node {(d.n_ko0_notdis > 0).mean():.3f}; median baseline nodes {int(d.n_nodes_base.median())}")
print(f"written {args.out}")
