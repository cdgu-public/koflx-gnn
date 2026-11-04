"""Evaluation of GNN v2 (script 81; also v1 of script 73 for the dry run) against MOMA and the priors, per fold,
on test strata (scheme_v2.md section 2, 2026-10-29).

Input: <run_root>/<graph>_<split>[_presence][_nomed]_s<seed>/fold<k>/pred_test.tsv (script 81) or
<run_root>/<graph>_<split>[_presence]/fold<k>/pred_test.tsv (script 73, seed 0 implied). Folders whose name
contains "precheck" or starts with "_" or is eval*/logs are skipped. Partial results are handled; what is
missing (folds 0-4 per config and seed) is printed and written to missing.txt.
Per sample: label = DepMap dependency >= 0.5; MOMA score = 1 - growth ratio (clipped), MOMA call ratio < 0.05.
Strata (gene split): all; exact-dup-free (dup_in_train False); rel_nn_dist >= 0.2 (test knockouts that are not train-test near duplicates; pre-specified); rel_nn_dist >= 0.5;
MOMA-lethal (G < 0.05); G > 0.99; rel >= 0.2 and MOMA-lethal. rel_nn_dist = distance of the test fluxome to the
nearest training fluxome of the same line / the sample's own |v_KO - v_WT|_inf, column written by script 81; for
v1 files it is taken from --rel-file (fold, line, gene, rel_nn_dist). Strata (line split): all; MOMA-lethal;
G > 0.99.
Comparators on the same samples: MOMA; per-gene prior (line split: mean dependency of the gene over the non-test
lines, NaN filled with the non-test global mean, as script 74); line-propensity prior (gene split: mean label of
the non-test GNN samples of the same line; the GNN sample set = union of all pred_test files found); flat
baseline (HGB on masked delta fluxome + medium, script 86, column hgb_prob) and growth probe (ridge-recovered
growth ratio, script 86, score = 1 - ridge_growth) when <flat_root>/<split>/fold<k>.tsv exists.
Metrics per fold and stratum: AUROC, AUPRC, sensitivity and MCC at the fold's test-ROC point with the MOMA
specificity (oracle point, labeled so), P@5 / P@10 per line (gene split; lines with >= k samples in the stratum;
ties by expected precision), within-gene AUROC (line split, genes mixed in the stratum). No cut-off 0.5 metrics.
Seeds: GNN metrics averaged over the available seeds per fold first (per_fold_seed.tsv keeps each seed), then
the paired fold differences GNN - comparator give mean, fold t-interval (t_{0.975, n-1} sd / sqrt(n)) and the
number of folds with GNN better. Secondary interval: line bootstrap within folds (--nb, default 200).
Usage: python 85_eval_gnn_v2.py <run_root> [--flat-root ...] [--rel-file ...] [--nb 200]
Env: crcgem_gnn. Output: <run_root>/eval_v2/{per_fold_seed.tsv, per_fold.tsv, summary.tsv, probe.tsv,
summary.txt, missing.txt}
"""
import argparse, glob, json, os, re, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
from scipy import stats
from sklearn.metrics import roc_auc_score as auc_, average_precision_score as ap_

ap = argparse.ArgumentParser()
ap.add_argument("run_root")
ap.add_argument("--flat-root", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/flat_v2")
ap.add_argument("--rel-file", default="", help="fold, line, gene, rel_nn_dist for pred files without the column (v1 dry run)")
ap.add_argument("--nb", type=int, default=200, help="line bootstrap resamples within folds (secondary interval)")
ap.add_argument("--out", default="", help="output folder (default <run_root>/eval_v2)")
ap.add_argument("--samples", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/dataset/samples.tsv",
                help="samples.tsv of the dataset the runs were trained on (MOMA comparator and strata); default = the 2026-10-28 dataset, unchanged behavior (2026-10-30 tolpatch: result/gnn_20261030_tolpatch/dataset/samples.tsv)")
args = ap.parse_args()
ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
G = f"{ROOT}/result/gnn_20261028"; RUN = args.run_root.rstrip("/"); OUT = args.out or f"{RUN}/eval_v2"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0); t0 = time.time()
logf = open(f"{OUT}/summary.txt", "w")
def say(*a):
    s = " ".join(map(str, a)); print(s, flush=True); logf.write(s + "\n")

say(f"samples (MOMA comparator, strata): {args.samples}")
smp = pd.read_csv(args.samples, sep="\t", comment="#")
smp["moma"] = 1 - smp.growth_ratio.clip(0, 1); smp["moma_call"] = smp.growth_ratio < 0.05
DEP = smp.pivot_table(index="line", columns="symbol", values="dependency")
REL = None
if args.rel_file:
    REL = pd.read_csv(args.rel_file, sep="\t")[["fold", "line", "gene", "rel_nn_dist"]].drop_duplicates(["fold", "line", "gene"])

# ---------------- discover runs ----------------
runs = {}   # (graph, split, variant) -> {seed: {fold: path}}
for rd in sorted(glob.glob(f"{RUN}/*")):
    name = os.path.basename(rd)
    if not os.path.isdir(rd) or name.startswith("_") or name.startswith("eval") or name == "logs" or "precheck" in name:
        continue
    parts = name.split("_")
    if len(parts) < 2 or parts[0] not in ("A", "B") or parts[1] not in ("gene", "line"):
        continue
    seed = 0; variant = []
    for p in parts[2:]:
        if re.fullmatch(r"s\d+", p):
            seed = int(p[1:])
        else:
            variant.append(p)
    key = (parts[0], parts[1], "_".join(variant))
    for fd in sorted(glob.glob(f"{rd}/fold*")):
        if os.path.exists(f"{fd}/pred_test.tsv"):
            runs.setdefault(key, {}).setdefault(seed, {})[int(fd.split("fold")[-1])] = f"{fd}/pred_test.tsv"
missing = []
for key, sd in sorted(runs.items()):
    for seed, fd in sorted(sd.items()):
        miss = [k for k in range(5) if k not in fd]
        say(f"config {key} seed {seed}: folds {sorted(fd)}" + (f"  MISSING folds {miss}" if miss else ""))
        if miss:
            missing.append(f"{key} seed {seed} missing folds {miss}")
if not runs:
    say("no pred_test.tsv found under", RUN); raise SystemExit(1)
open(f"{OUT}/missing.txt", "w").write("\n".join(missing) + ("\n" if missing else "none\n"))

# GNN sample set (union of all test files) for the line-propensity prior
allp = pd.concat([pd.read_csv(p, sep="\t", usecols=["line", "gene", "label"]) for sd in runs.values() for fd in sd.values() for p in fd.values()]).drop_duplicates(["line", "gene"])
say(f"GNN sample set from pred files: {len(allp)} (line, gene) pairs")


def load_fold(path, split, fold):
    d = pd.read_csv(path, sep="\t").merge(smp[["line", "gene", "moma", "moma_call", "growth_ratio"]], on=["line", "gene"], how="left")
    assert d.moma.notna().all(), f"{path}: samples not in samples.tsv"
    sj = json.load(open(os.path.join(os.path.dirname(path), "split.json")))
    if "rel_nn_dist" not in d and REL is not None:
        d = d.merge(REL[REL.fold == fold].drop(columns="fold"), on=["line", "gene"], how="left")
    if "rel_nn_dist" not in d:
        d["rel_nn_dist"] = np.nan
    test_pairs = set(zip(d.line, d.gene))
    if split == "line":
        trl = [l for l in DEP.index if l not in set(sj["test_lines"])]
        pr = DEP.loc[trl].mean(); d["prior"] = d.symbol.map(pr).fillna(float(DEP.loc[trl].stack().mean()))
    else:
        nt = allp[[(l, g) not in test_pairs for l, g in zip(allp.line, allp.gene)]]
        d["lineprior"] = d.line.map(nt.groupby("line").label.mean())
    fp = f"{args.flat_root}/{split}/fold{fold}.tsv"
    if os.path.exists(fp):
        f = pd.read_csv(fp, sep="\t")[["line", "gene", "hgb_prob", "ridge_growth"]]
        d = d.merge(f, on=["line", "gene"], how="left"); d["probe"] = 1 - d.ridge_growth.clip(0, 1)
    return d


def strata(d, split):
    g = d.growth_ratio.to_numpy(); s = {"all": np.ones(len(d), bool), "MOMA-lethal (G<0.05)": g < 0.05, "G>0.99": g > 0.99}
    if split == "gene":
        r = d.rel_nn_dist.to_numpy(dtype=float)
        s.update({"exact-dup-free": ~d.dup_in_train.to_numpy(bool), "rel>=0.2": r >= 0.2, "rel>=0.5": r >= 0.5,
                  "rel>=0.2 & MOMA-lethal": (r >= 0.2) & (g < 0.05)})
    return s


def mcc(y, c):
    tp, fp, fn, tn = (c & y).sum(), (c & ~y).sum(), (~c & y).sum(), (~c & ~y).sum()
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); return (tp * tn - fp * fn) / den if den else np.nan


def prec_at_k(y, s, k):
    o = np.argsort(-s, kind="stable"); s_, y_ = s[o], y[o]; thr = s_[k - 1]
    above = s_ > thr; tie = s_ == thr
    return (y_[above].sum() + (k - above.sum()) * y_[tie].mean()) / k


COMP = ["moma", "prior", "lineprior", "hgb_prob", "probe"]
KS = (5, 10)


def metrics(d, split):
    y = d.label.to_numpy() == 1
    if len(y) < 20 or y.all() or not y.any():
        return None
    r = dict(n=len(d), n_lines=d.line.nunique(), pos_rate=float(y.mean()))
    cm = d.moma_call.to_numpy(bool); sp_m = ((~cm) & (~y)).sum() / (~y).sum(); r["spec_moma_call"] = sp_m
    r["sens_moma_call"] = (cm & y).sum() / y.sum(); r["MCC_moma_call"] = mcc(y, cm)
    for c in ["gnn_prob"] + COMP:
        if c not in d or d[c].isna().any():
            continue
        s = d[c].to_numpy(dtype=float); tag = "gnn" if c == "gnn_prob" else c
        r[f"AUROC_{tag}"] = auc_(y, s); r[f"AUPRC_{tag}"] = ap_(y, s)
        if tag != "moma":
            cs = s > np.quantile(s[~y], sp_m)   # oracle operating point at MOMA's specificity on these samples
            r[f"sens_at_moma_spec_{tag}"] = (cs & y).sum() / y.sum(); r[f"MCC_at_moma_spec_{tag}"] = mcc(y, cs)
        if split == "gene":
            for k in KS:
                v = [prec_at_k(x.label.to_numpy() == 1, x[c].to_numpy(dtype=float), k) for _, x in d.groupby("line") if len(x) >= k]
                r[f"P@{k}_{tag}"] = np.mean(v) if v else np.nan
                if tag == "gnn":
                    r[f"P@{k}_random"] = np.mean([x.label.mean() for _, x in d.groupby("line") if len(x) >= k]) if v else np.nan
                    r[f"n_lines@{k}"] = len(v)
        if split == "line":
            a = [auc_(x.label == 1, x[c]) for _, x in d.groupby("symbol") if 0 < (x.label == 1).sum() < len(x) and x[c].nunique() > 1]
            r[f"within_gene_AUROC_{tag}"] = np.mean(a) if a else np.nan
            if tag == "gnn":
                r["within_gene_n_genes"] = len(a)
    return r


# ---------------- per fold and seed ----------------
rows_fs = []; data = {}   # (key, seed, fold) -> df
for key, sd in sorted(runs.items()):
    graph, split, variant = key
    for seed, fd in sorted(sd.items()):
        for fold, path in sorted(fd.items()):
            d = load_fold(path, split, fold); data[(key, seed, fold)] = d
            for sname, sm in strata(d, split).items():
                m = metrics(d[sm], split)
                if m is None:
                    continue
                rows_fs.append(dict(graph=graph, split=split, variant=variant, seed=seed, fold=fold, stratum=sname, **m))
pfs = pd.DataFrame(rows_fs)
pfs.to_csv(f"{OUT}/per_fold_seed.tsv", sep="\t", index=False)
# seeds averaged per fold
gcols = ["graph", "split", "variant", "fold", "stratum"]
num = [c for c in pfs.columns if c not in gcols + ["seed"]]
pf = pfs.groupby(gcols, as_index=False).agg({**{c: "mean" for c in num}, "seed": "count"}).rename(columns={"seed": "n_seeds"})
pf.to_csv(f"{OUT}/per_fold.tsv", sep="\t", index=False)

# ---------------- summary: paired fold differences, t-interval, line bootstrap ----------------
PAIRS = [(f"{m}_gnn", f"{m}_{c}") for m in ("AUROC", "AUPRC", "within_gene_AUROC", "MCC_at_moma_spec", "sens_at_moma_spec") for c in COMP] + \
        [(f"P@{k}_gnn", f"P@{k}_{c}") for k in KS for c in COMP + ["random"]]
PAIRS = [(a, b) for a, b in PAIRS if not (b.endswith("_moma") and a.startswith(("MCC_at", "sens_at")))]
rows_s = []
for (graph, split, variant), grp in pf.groupby(["graph", "split", "variant"]):
    for sname, g in grp.groupby("stratum"):
        g = g.set_index("fold")
        # line bootstrap within folds (secondary), on the seed-mean predictions? bootstrap uses seed 0 predictions of each fold
        bs = []
        if args.nb > 0:
            per_fold_lines = {}
            for fold in g.index:
                seeds = sorted(s for (k, s, f) in data if k == (graph, split, variant) and f == fold)
                d = data[((graph, split, variant), seeds[0], fold)]
                if len(seeds) > 1:   # mean prediction over seeds
                    d = d.copy(); d["gnn_prob"] = np.mean([data[((graph, split, variant), s, fold)].set_index(["line", "gene"]).gnn_prob.reindex(list(zip(d.line, d.gene))).to_numpy() for s in seeds], 0)
                dd = d[strata(d, split)[sname]]
                per_fold_lines[fold] = {l: x for l, x in dd.groupby("line")}
            for _ in range(args.nb):
                ms = []
                for fold, byl in per_fold_lines.items():
                    ls = list(byl); x = pd.concat([byl[l].assign(line=f"{l}#{i}") for i, l in enumerate(rng.choice(ls, len(ls)))])
                    mm = metrics(x, split)
                    if mm is not None:
                        ms.append(mm)
                bs.append({f"{a}-{b}": np.nanmean([m[a] - m[b] for m in ms if a in m and b in m]) for a, b in PAIRS
                           if any(a in m and b in m for m in ms)})
        for a, b in PAIRS:
            if a not in g or b not in g or g[a].isna().all() or g[b].isna().all():
                continue
            dv = (g[a] - g[b]).dropna(); n = len(dv)
            if n == 0:
                continue
            tl = th = np.nan
            if n >= 2 and dv.std(ddof=1) > 0:
                h = stats.t.ppf(0.975, n - 1) * dv.std(ddof=1) / np.sqrt(n); tl, th = dv.mean() - h, dv.mean() + h
            bv = np.array([x[f"{a}-{b}"] for x in bs if f"{a}-{b}" in x])
            rows_s.append(dict(graph=graph, split=split, variant=variant, stratum=sname, n_folds=n, n_seeds=int(g.n_seeds.min()),
                               n=int(g.n.sum()), metric=a, comparator=b, gnn_mean=g[a].mean(), comp_mean=g[b].mean(),
                               diff_mean=dv.mean(), t_lo=tl, t_hi=th,
                               boot_lo=np.nanpercentile(bv, 2.5) if len(bv) else np.nan, boot_hi=np.nanpercentile(bv, 97.5) if len(bv) else np.nan,
                               folds_gnn_better=int((dv > 0).sum()), per_fold_diff=";".join(f"{x:.3f}" for x in dv)))
su = pd.DataFrame(rows_s); su.to_csv(f"{OUT}/summary.tsv", sep="\t", index=False)

# growth probe R2 per fold (script 86 meta), when present
probe = []
for mp in sorted(glob.glob(f"{args.flat_root}/*/fold*_meta.json")):
    j = json.load(open(mp)); probe.append({k: j[k] for k in j if not isinstance(j[k], (list, dict))})
if probe:
    pd.DataFrame(probe).to_csv(f"{OUT}/probe.tsv", sep="\t", index=False)

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 2000)
say(f"\n# per fold (seeds averaged), key columns")
cols = ["graph", "split", "variant", "fold", "stratum", "n_seeds", "n", "pos_rate"] + [c for c in ["AUROC_gnn", "AUROC_moma", "AUROC_prior", "AUROC_lineprior", "AUROC_hgb_prob", "AUROC_probe", "AUPRC_gnn", "AUPRC_moma", "within_gene_AUROC_gnn", "within_gene_AUROC_moma", "P@5_gnn", "P@5_moma", "P@5_random"] if c in pf]
say(pf[cols].round(3).to_string(index=False))
say(f"\n# summary: paired fold differences GNN - comparator (fold t-interval; line bootstrap {args.nb} secondary)")
if len(su):
    say(su[su.metric.isin(["AUROC_gnn", "AUPRC_gnn", "within_gene_AUROC_gnn", "P@5_gnn"])].round(3).to_string(index=False))
if probe:
    say("\n# growth probe (script 86)"); say(pd.DataFrame(probe).round(3).to_string(index=False))
say(f"\nmissing: {missing if missing else 'none'}")
say(f"done {time.time()-t0:.0f}s -> {OUT}")
