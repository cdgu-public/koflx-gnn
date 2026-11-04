"""Evaluation on the external validation set of non-colorectal cell lines (term pending ratification), exactly as
pre-specified in result/gnn_20261028/discussion_20261030/external_eval_plan.md (sections 4-8), 2026-10-30. Script 85
is not modified (running runners call it); the metric helpers follow its rules (set scored only with >= 20 samples and
both classes; AUROC / AUPRC; no cut-off metrics).

Input: one or more prediction roots written by 99_gnn_predict_external.py (each <root>/fold<k>/pred_external.tsv; the
root's basename is parsed as <graph>_<split>[_tokens]_s<seed>, seeds of the same (graph, tokens) are pooled into one
model family); the comparator file folder of 86b (<flat>/line/fold<k>.tsv); the colorectal dataset samples.tsv for the
per-gene base rate (mean dependency of the gene over the 59 colorectal cell lines; genes absent get the global mean).
Per cell line with a DepMap label, three sets: all; MOMA-lethal (growth ratio < 0.05); genes discordant between the
external cell line and the colorectal per-gene base rate ((base rate >= 0.5) != label). Scores: GNN mean-probability
ensemble over all models of the family (per-model distribution kept), per-gene base rate, MOMA (1 - growth ratio),
gradient-boosting baseline hgb_nomed_prob (paired comparator) and hgb_prob (with medium), growth probe (1 - mean
ridge_growth). Intervals: within-cell-line sample bootstrap (--nb, default 2000, seed 0, percentile 2.5/97.5) for the
ensemble AUROC, each comparator and the paired difference. Sign counts over the labeled cell lines. Falsifier F1/F2 of
the plan printed. ACH-002466 (no label) listed with its n and excluded. Not computed: relative-distance strata,
within-gene AUROC, t interval over cell lines, pooled AUROC, top-k, operating points.
Usage: python 85b_eval_external.py --pred-roots <root> [<root> ...] [--flat result/external_20261030/flat] [--nb 2000]
       [--out result/external_20261030/eval_<YYYYMMDD_HHMMSS>]   (never overwrites)
Env: crcgem_gnn. Output: per_model.tsv, per_line.tsv, paired.tsv, summary.txt (all with # condition lines).
"""
import argparse, glob, hashlib, json, os, re, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as auc_, average_precision_score as ap_

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
ap = argparse.ArgumentParser()
ap.add_argument("--pred-roots", nargs="+", required=True)
ap.add_argument("--flat", default=f"{ROOT}/result/external_20261030/flat")
ap.add_argument("--train-samples", default=f"{ROOT}/result/gnn_20261028/dataset/samples.tsv")
ap.add_argument("--external-samples", default=f"{ROOT}/result/external_20261030/dataset/samples.tsv")
ap.add_argument("--nb", type=int, default=2000)
ap.add_argument("--out", default="")
args = ap.parse_args()
OUT = args.out or f"{ROOT}/result/external_20261030/eval_{time.strftime('%Y%m%d_%H%M%S')}"
assert not os.path.exists(OUT), f"{OUT} exists (no overwrite)"; os.makedirs(OUT)
rng = np.random.default_rng(0); t0 = time.time()
logf = open(f"{OUT}/summary.txt", "w")
def say(*a):
    s = " ".join(map(str, a)); print(s, flush=True); logf.write(s + "\n")
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
COND = (f"# script_python/85b_eval_external.py; {time.ctime()}; pred roots {args.pred_roots}; flat {args.flat}; base rate from {args.train_samples} "
        f"(md5 {md5(args.train_samples)}); external samples {args.external_samples} (md5 {md5(args.external_samples)}); bootstrap nb {args.nb} seed 0; "
        f"label dependency >= 0.5; MOMA-lethal growth ratio < 0.05; discordant = (base rate >= 0.5) != label; set scored only if n >= 20 with both classes")
say(COND)

# ---------------- comparators ----------------
tr = pd.read_csv(args.train_samples, sep="\t", comment="#")
BR = tr.groupby("symbol").dependency.mean(); BR_GLOBAL = float(tr.dependency.mean())
ext = pd.read_csv(args.external_samples, sep="\t", comment="#")
no_label_lines = sorted(ext.line[~ext.has_label].unique())
flat = {}
for fp in sorted(glob.glob(f"{args.flat}/line/fold*.tsv")):
    flat[int(re.search(r"fold(\d+)", fp).group(1))] = pd.read_csv(fp, sep="\t", comment="#").set_index(["line", "gene"])
say(f"comparator folds found: {sorted(flat)}; cell lines without label (excluded from metrics): {no_label_lines}")

# ---------------- predictions, grouped into model families ----------------
fam = {}   # (graph, tokens) -> list of (seed, fold, df)
for root in args.pred_roots:
    name = os.path.basename(root.rstrip("/")); parts = name.split("_")
    assert parts[0] in ("A", "B"), f"cannot parse graph from {name}"
    seed = next((int(p[1:]) for p in parts if re.fullmatch(r"s\d+", p)), 0)
    tokens = "_".join(p for p in parts[1:] if not re.fullmatch(r"s\d+", p))
    for fp in sorted(glob.glob(f"{root}/fold*/pred_external.tsv")):
        fold = int(re.search(r"fold(\d+)", fp).group(1)); d = pd.read_csv(fp, sep="\t", comment="#")
        hdr = [l.strip() for l in open(fp) if l.startswith("#")]
        assert not any("SUBSET" in h for h in hdr), f"{fp} is a smoke-test subset file"
        fam.setdefault((parts[0], tokens), []).append((seed, fold, d, fp))
for key, lst in sorted(fam.items()):
    say(f"family graph {key[0]} tokens '{key[1]}': {len(lst)} models (seed, fold) = {sorted((s, f) for s, f, _, _ in lst)}")
assert fam, "no pred_external.tsv found"


def attach(d):
    d = d.copy(); d["base_rate"] = d.symbol.map(BR); d["n_base_rate_missing"] = int(d.base_rate.isna().sum()); d["base_rate"] = d.base_rate.fillna(BR_GLOBAL)
    d["moma"] = 1 - d.growth_ratio.clip(0, 1); d["moma_lethal"] = d.growth_ratio < 0.05
    if flat:
        key = list(zip(d.line, d.gene))
        for c in ("hgb_prob", "hgb_nomed_prob", "ridge_growth"):
            d[c] = np.mean([flat[k][c].reindex(key).to_numpy() for k in sorted(flat)], 0)
        d["probe"] = 1 - d.ridge_growth.clip(0, 1)
    return d


SCORES = ["gnn", "base_rate", "moma", "hgb_nomed_prob", "hgb_prob", "probe"]
def sets(d):
    y = d.label.to_numpy() == 1
    return {"all": np.ones(len(d), bool), "MOMA-lethal": d.moma_lethal.to_numpy(bool), "discordant": (d.base_rate.to_numpy() >= 0.5) != y}
def scoreable(y):
    return len(y) >= 20 and 0 < y.sum() < len(y)
def aucs(d, cols):
    y = d.label.to_numpy() == 1
    if not scoreable(y):
        return None
    return {c: auc_(y, d[c].to_numpy(dtype=float)) for c in cols if c in d}


rows_model, rows_line, rows_pair = [], [], []
for (graph, tokens), lst in sorted(fam.items()):
    # ensemble: mean probability over all models on the (line, gene) index; every model must cover the same samples
    base = attach(lst[0][2].set_index(["line", "gene"]).sort_index().reset_index())
    idx = list(zip(base.line, base.gene))
    P = np.stack([d.set_index(["line", "gene"]).gnn_prob.reindex(idx).to_numpy() for _, _, d, _ in lst])
    assert np.isfinite(P).all(), "a model misses samples of another model"
    base["gnn"] = P.mean(0)
    # per-model distribution
    for (seed, fold, d, fp), p in zip(lst, P):
        dd = base.assign(gnn=p)
        for ln, x in dd[dd.has_label].groupby("line"):
            for sname, sm in sets(x).items():
                a = aucs(x[sm], ["gnn"])
                rows_model.append(dict(graph=graph, tokens=tokens, seed=seed, fold=fold, line=ln, set=sname, n=int(sm.sum()),
                                       pos=int((x[sm].label == 1).sum()), AUROC_gnn=a["gnn"] if a else np.nan, file=fp))
    # ensemble per cell line, with bootstrap
    for ln, x in base.groupby("line"):
        if not x.has_label.all():
            say(f"{graph}/{tokens} {ln}: no DepMap label, n {len(x)} predicted, excluded from metrics"); continue
        for sname, sm in sets(x).items():
            xs = x[sm].reset_index(drop=True); y = xs.label.to_numpy() == 1
            rec = dict(graph=graph, tokens=tokens, n_models=len(lst), line=ln, set=sname, n=len(xs), pos=int(y.sum()), prevalence=float(y.mean()) if len(xs) else np.nan,
                       n_base_rate_missing=int(xs.n_base_rate_missing.iloc[0]) if len(xs) else 0)
            if not scoreable(y):
                rec["scored"] = False; rows_line.append(rec); continue
            rec["scored"] = True
            cols = [c for c in SCORES if c in xs and not (sname == "discordant" and c == "base_rate")]
            for c in cols:
                rec[f"AUROC_{c}"] = auc_(y, xs[c]); rec[f"AUPRC_{c}"] = ap_(y, xs[c])
            pm = [r["AUROC_gnn"] for r in rows_model if r["graph"] == graph and r["tokens"] == tokens and r["line"] == ln and r["set"] == sname and np.isfinite(r["AUROC_gnn"])]
            rec.update(AUROC_gnn_model_min=np.min(pm) if pm else np.nan, AUROC_gnn_model_median=np.median(pm) if pm else np.nan, AUROC_gnn_model_max=np.max(pm) if pm else np.nan)
            B = {c: [] for c in cols}; Bd = {c: [] for c in cols if c != "gnn"}
            for _ in range(args.nb):
                bi = rng.integers(0, len(xs), len(xs)); yb = y[bi]
                if not (0 < yb.sum() < len(yb)):
                    continue
                av = {c: auc_(yb, xs[c].to_numpy()[bi]) for c in cols}
                for c in cols:
                    B[c].append(av[c])
                    if c != "gnn":
                        Bd[c].append(av["gnn"] - av[c])
            for c in cols:
                rec[f"AUROC_{c}_lo"], rec[f"AUROC_{c}_hi"] = (np.percentile(B[c], [2.5, 97.5]) if B[c] else (np.nan, np.nan))
            for c in Bd:
                lo, hi = np.percentile(Bd[c], [2.5, 97.5]) if Bd[c] else (np.nan, np.nan)
                rows_pair.append(dict(graph=graph, tokens=tokens, line=ln, set=sname, n=len(xs), comparator=c, AUROC_gnn=rec["AUROC_gnn"], AUROC_comp=rec[f"AUROC_{c}"],
                                      diff=rec["AUROC_gnn"] - rec[f"AUROC_{c}"], boot_lo=lo, boot_hi=hi, excludes_0=bool(lo > 0 or hi < 0), n_boot=len(Bd[c])))
            rec["n_boot"] = len(B["gnn"])
            rows_line.append(rec)

pm = pd.DataFrame(rows_model); pl = pd.DataFrame(rows_line); pp = pd.DataFrame(rows_pair)
for name, df in (("per_model", pm), ("per_line", pl), ("paired", pp)):
    with open(f"{OUT}/{name}.tsv", "w") as fh:
        fh.write(COND + "\n"); df.to_csv(fh, sep="\t", index=False)

pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)
say("\n# per cell line, ensemble (AUROC with sample-bootstrap interval; per-model min/median/max)")
show = [c for c in ["graph", "tokens", "n_models", "line", "set", "n", "pos", "prevalence", "scored", "AUROC_gnn", "AUROC_gnn_lo", "AUROC_gnn_hi", "AUROC_gnn_model_min", "AUROC_gnn_model_median",
                    "AUROC_gnn_model_max", "AUROC_base_rate", "AUROC_moma", "AUROC_hgb_nomed_prob", "AUROC_hgb_prob", "AUROC_probe", "AUPRC_gnn", "AUPRC_base_rate", "AUPRC_moma", "n_base_rate_missing"] if c in pl]
say(pl[show].round(3).to_string(index=False))
say("\n# paired differences ensemble GNN - comparator per cell line (bootstrap interval of the difference) and sign counts")
if len(pp):
    say(pp.round(3).to_string(index=False))
    sc = pp[pp.set != "x"].groupby(["graph", "tokens", "set", "comparator"]).agg(n_lines=("diff", "size"), gnn_higher=("diff", lambda v: int((v > 0).sum())),
                                                                              gnn_higher_excl0=("excludes_0", lambda v: int(((pp.loc[v.index, "diff"] > 0) & v).sum()))).reset_index()
    say(sc.to_string(index=False))
say("\n# falsifier of the plan (section 7): F1 = set all, GNN > base rate with interval excluding 0, in >= 5 of 6 labeled cell lines;"
    " F2 = set discordant, GNN AUROC > 0.5 with interval excluding 0.5, in >= 5 of 6")
for (graph, tokens), g in pl.groupby(["graph", "tokens"]):
    f1 = pp[(pp.graph == graph) & (pp.tokens == tokens) & (pp.set == "all") & (pp.comparator == "base_rate")]; n1 = int(((f1["diff"] > 0) & f1.excludes_0).sum())
    f2 = g[(g.set == "discordant") & (g.scored == True)]; n2 = int(((f2.AUROC_gnn > 0.5) & (f2.AUROC_gnn_lo > 0.5)).sum()) if len(f2) else 0
    nl = g[g.set == "all"].line.nunique()
    say(f"{graph}/{tokens}: F1 met in {n1}/{nl} cell lines, F2 met in {n2}/{nl} (discordant sets scored: {len(f2)}); FALSIFIED = {n1 >= 5 or n2 >= 5}")
say(f"\ndone {time.time()-t0:.0f}s -> {OUT}")
