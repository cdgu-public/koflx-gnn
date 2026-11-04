"""Normal (non-cancerous) cell lines: readout of design result/external_plan_20261103/design_external_gi_normal_20261103.md
section 2.2 / 2.4 (falsifiers N1, N2), step 6(h). Pre-specified there; this script adds no threshold.

G_CRC = knockout samples of a normal cell line that passed the fluxome-change filter (= rows of the script 99 prediction files)
and whose gene has colorectal base rate >= 0.5 (base rate = mean DepMap dependency of the gene over the colorectal dataset
samples.tsv, absent genes get the global mean; same definition as 85b:49). Within G_CRC: positive = essential in the normal
cell line (DepMap CRISPRGeneDependency >= 0.5, shared essential), negative = non-essential (cancer-selective).
Scores (as 85b): GNN = mean probability over all models of the graph (A' headline, B reported; 15 cell-line-disjoint tolpatch
models each); base rate; MOMA = 1 - growth ratio clipped to [0, 1]; HGB without medium = hgb_nomed_prob averaged over the 5
cell-line folds of 86b. Primary metric AUROC on G_CRC; paired bootstrap of GNN - comparator (2,000 resamples of the G_CRC samples
of the cell line, numpy default_rng(0) restarted per (graph, cell line), single-class resamples skipped, percentile 2.5/97.5).
Secondary: specificity on the cancer-selective genes = fraction called non-essential (GNN probability < 0.5; MOMA growth ratio
>= 0.05; base rate calls all of G_CRC essential by construction, specificity 0). Scored only if n >= 20 and >= 5 per class.
Cell lines: DepMap OncotreePrimaryDisease == "Non-Cancerous" with a DepMap label, in each source set. Set 1
(result/external_20261030_tolpatch, RPE1 lines) is labeled "post hoc on previously seen predictions" (design 2.3).
Usage: python 140_eval_normal_gcrc.py --out <dir>   (refuses to overwrite). Env: crcgem_gnn.
"""
import argparse, glob, hashlib, os, re, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc_

R = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
SOURCES = [  # (set label, result root, line map, timing label)
    ("set1_external_20261030", f"{R}/result/external_20261030_tolpatch", f"{R}/data/depmap/oob_20261028/oob_lines.tsv",
     "post hoc on previously seen predictions"),
    ("set2_external_gi_normal_20261103", f"{R}/result/external_gi_normal_20261103",
     f"{R}/result/external_gi_normal_20261103/dataset_inputs/lines_merged.tsv", "pre-specified (design 2026-11-03 section 2.2)")]
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--train-samples", default=f"{R}/result/gnn_20261030_tolpatch/dataset_v2/samples.tsv")
ap.add_argument("--nb", type=int, default=2000)
ap.add_argument("--exclude", nargs="*", default=[], help="ModelIDs removed from the normal set (2026-11-04 user ruling: Hs 729 ACH-000133 = cancerous)")
args = ap.parse_args()
assert not os.path.exists(args.out), f"{args.out} exists (no overwrite)"; os.makedirs(args.out)
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
tr = pd.read_csv(args.train_samples, sep="\t", comment="#")
BR = tr.groupby("symbol").dependency.mean(); BR_GLOBAL = float(tr.dependency.mean())
RUNS = {"A": [f"A_line_nomed_fixedcos30_s{s}_tolpatch" for s in range(3)], "B": [f"B_line_nomed_meanmax_fixedcos30_s{s}_tolpatch" for s in range(3)]}
COMP = ["base_rate", "moma", "hgb_nomed_prob"]

rows, pairs, log = [], [], []
def say(s): print(s, flush=True); log.append(s)
for set_label, E, lm_p, timing in SOURCES:
    lm = pd.read_csv(lm_p, sep="\t", comment="#")
    normal = sorted(set(lm.ModelID[lm.OncotreePrimaryDisease == "Non-Cancerous"]) - set(args.exclude))
    fl = {int(re.search(r"fold(\d+)", f).group(1)): pd.read_csv(f, sep="\t", comment="#").set_index(["line", "gene"])
          for f in sorted(glob.glob(f"{E}/flat_dsv2/line/fold*.tsv"))}
    assert sorted(fl) == [0, 1, 2, 3, 4], f"{E}: flat folds {sorted(fl)}"
    say(f"{set_label}: normal cell lines in map {normal} (excluded by ruling: {args.exclude}); flat folds {sorted(fl)}")
    for graph, runs in RUNS.items():
        fs = sorted(f for r in runs for f in glob.glob(f"{E}/pred_v3/{r}/fold*/pred_external.tsv"))
        assert len(fs) == 15, f"{E} graph {graph}: {len(fs)} prediction files, expected 15"
        ds = [pd.read_csv(f, sep="\t", comment="#").set_index(["line", "gene"]).sort_index() for f in fs]
        assert all(d.index.equals(ds[0].index) for d in ds), "models do not cover the same samples"
        d = ds[0].reset_index(); d["gnn"] = np.mean([x.gnn_prob.to_numpy() for x in ds], 0)
        d["base_rate_missing"] = d.symbol.map(BR).isna(); d["base_rate"] = d.symbol.map(BR).fillna(BR_GLOBAL)
        d["moma"] = 1 - d.growth_ratio.clip(0, 1)
        idx = list(zip(d.line, d.gene))
        d["hgb_nomed_prob"] = np.mean([fl[k].hgb_nomed_prob.reindex(idx).to_numpy() for k in range(5)], 0)
        assert np.isfinite(d.hgb_nomed_prob).all(), "HGB comparator misses samples"
        for ln in normal:
            x0 = d[d.line == ln]
            if len(x0) == 0 or not x0.has_label.all():
                say(f"  {graph} {ln}: no DepMap label or no predicted samples (n {len(x0)}), not scored"); continue
            x = x0[x0.base_rate >= 0.5].reset_index(drop=True); y = (x.label == 1).to_numpy()
            npos, nneg = int(y.sum()), int((~y).sum())
            rec = dict(source_set=set_label, timing=timing, graph=graph, n_models=len(fs), line=ln,
                       cell_line_name=lm.set_index("ModelID").CellLineName.get(ln, ""), n_predicted=len(x0), n_GCRC=len(x),
                       n_shared_essential=npos, n_cancer_selective=nneg, n_base_rate_missing_in_GCRC=int(x.base_rate_missing.sum()),
                       scored=bool(len(x) >= 20 and npos >= 5 and nneg >= 5))
            neg = ~y
            rec["spec_gnn"] = float((x.gnn[neg] < 0.5).mean()) if nneg else np.nan
            rec["spec_moma"] = float((x.growth_ratio[neg] >= 0.05).mean()) if nneg else np.nan
            rec["spec_base_rate"] = 0.0 if nneg else np.nan
            if rec["scored"]:
                cols = ["gnn"] + COMP
                for c in cols:
                    rec[f"AUROC_{c}"] = auc_(y, x[c].to_numpy(float))
                rng = np.random.default_rng(0); B = {c: [] for c in cols}; Bd = {c: [] for c in COMP}
                for _ in range(args.nb):
                    bi = rng.integers(0, len(x), len(x)); yb = y[bi]
                    if not (0 < yb.sum() < len(yb)):
                        continue
                    av = {c: auc_(yb, x[c].to_numpy(float)[bi]) for c in cols}
                    for c in cols: B[c].append(av[c])
                    for c in COMP: Bd[c].append(av["gnn"] - av[c])
                for c in cols:
                    rec[f"AUROC_{c}_lo"], rec[f"AUROC_{c}_hi"] = np.percentile(B[c], [2.5, 97.5])
                rec["n_boot"] = len(B["gnn"])
                for c in COMP:
                    lo, hi = np.percentile(Bd[c], [2.5, 97.5])
                    pairs.append(dict(source_set=set_label, timing=timing, graph=graph, line=ln, n_GCRC=len(x), comparator=c,
                                      AUROC_gnn=rec["AUROC_gnn"], AUROC_comp=rec[f"AUROC_{c}"], diff=rec["AUROC_gnn"] - rec[f"AUROC_{c}"],
                                      boot_lo=lo, boot_hi=hi, excludes_0=bool(lo > 0 or hi < 0), n_boot=len(Bd[c])))
            rows.append(rec)

pl, pp = pd.DataFrame(rows), pd.DataFrame(pairs)
# falsifiers (design 2.4), per source set and graph, over scored cell lines
fz = []
for (s, g), x in pl[pl.scored].groupby(["source_set", "graph"]):
    n = len(x); cover = int(((x.AUROC_gnn_lo <= 0.5) & (x.AUROC_gnn_hi >= 0.5)).sum())
    q = pp[(pp.source_set == s) & (pp.graph == g) & (pp.comparator == "base_rate")]; le0 = int((q["diff"] <= 0).sum())
    fz.append(dict(source_set=s, graph=g, n_scored_lines=n, N1_interval_covers_0p5=cover, N1_falsified=bool(cover > n / 2),
                   N2_gnn_minus_base_rate_le0=le0, N2_falsified=bool(le0 > n / 2)))
fz = pd.DataFrame(fz)
COND = (f"# script_python/140_eval_normal_gcrc.py; {time.ctime()}; design result/external_plan_20261103/design_external_gi_normal_20261103.md 2.2/2.4\n"
        f"# sources: " + "; ".join(f"{a} = {b} (line map {c}; {t})" for a, b, c, t in SOURCES) + "\n"
        f"# base rate from {args.train_samples} (md5 {md5(args.train_samples)}), absent gene -> global mean {BR_GLOBAL:.6f}; G_CRC = base rate >= 0.5 among predicted (fluxome-change filter) samples\n"
        f"# label DepMap dependency >= 0.5 (positive = shared essential, negative = cancer-selective); GNN = mean probability of 15 models per graph "
        f"(A: {RUNS['A']}; B: {RUNS['B']}, folds 0-4); MOMA = 1 - growth ratio; HGB = hgb_nomed_prob mean over 5 folds (flat_dsv2/line)\n"
        f"# scored if n >= 20 and >= 5 per class; bootstrap nb {args.nb}, default_rng(0) per (graph, cell line), percentile 2.5/97.5; "
        f"specificity: GNN < 0.5, MOMA growth ratio >= 0.05, base rate 0 by construction\n"
        f"# N1 falsified if GNN AUROC interval covers 0.5 in the majority of scored cell lines; N2 falsified if GNN - base rate <= 0 in the majority; "
        f"per source set (set 1 RPE1 = post hoc). n rows per_line {len(pl)}, paired {len(pp)}\n")
for name, df in (("per_line", pl), ("paired", pp), ("falsifiers", fz)):
    with open(f"{args.out}/{name}.tsv", "w") as fh:
        fh.write(COND); df.to_csv(fh, sep="\t", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
say(pl.round(3).to_string(index=False)); say(pp.round(3).to_string(index=False)); say(fz.to_string(index=False))
open(f"{args.out}/summary.txt", "w").write(COND + "\n".join(log) + "\n")
