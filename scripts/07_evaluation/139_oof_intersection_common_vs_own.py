"""Pooled out-of-fold AUROC on the samples shared with the own-medium dataset: each common-medium GNN grid against the own-medium GNN grid
(design result/common_medium_20261030/gnn_design_20261031.md section 2 items (2) and (3b), falsifiers section 3; user approval decision log 2026-11-03 ruling (3)).

Sample set per common medium M: raw/intersection_samples.tsv of script 111 (own-medium dataset_v2 GNN samples intersected with the common-medium
dataset GNN samples, 81b fluxome-change filter): DMEM/F-12 17,708, RPMI-1640 18,427. Own-only and common-only samples: counts only, no metric.
Per (M, graph in {A (A′), B}, split in {gene, line}) and per sample: GNN = mean gnn_prob over seeds 0-2 of the one out-of-fold prediction per seed
(pred_test.tsv, folds 0-4) in the common grid and in the own grid; MOMA = score_common / score_own of 111 (1 - clipped relative growth rate);
gradient-boosting baseline without medium input = hgb_prob of the non-val rows of <flat root>/<split>/fold<k>.tsv of the same dataset
(fold-matched; own flat_v2_expr_dsv2/fluxonly, common flat_v2_expr/fluxonly); per-gene base rate = 85 'prior' (cell-line-disjoint split only:
mean dependency of the gene over the non-test cell lines of the fold, NaN filled with the global mean; undefined for the gene-disjoint split
because the test gene has no training rows, as in 85). Label = DepMap dependency >= 0.5.
Readouts: one AUROC on the pooled out-of-fold predictions of the shared samples (pooled, NOT the fold-mean AUROC of 85 / comparison_grid);
paired difference common minus own pooled, with a percentile interval from a bootstrap over cell lines (2000 resamples, seed 0, the same
resampled cell lines for every column of one (M, split)); per cell line AUROC and difference (>= 20 samples with both classes, as 111);
cell-line-disjoint split only: per-fold AUROC on the per-fold shared samples, difference common minus own, mean and 95% t interval over the
5 folds (df 4, as 85/129).
Falsifiers / gates (any failure -> AssertionError, nothing written): every pred_test set equal across seeds (samples and fold), no sample
twice within a seed, every shared sample has exactly one out-of-fold prediction per grid per seed; config.json dataset/out_root/graph/split/seed
as expected; flat non-val rows of every fold equal the GNN test set of that fold; labels identical across own, common, 111; MOMA of 111 equal to
1 - clip(growth_ratio) of each samples.tsv; own and common columns computed on the identical sample array with 3 seeds each.
Reproduction gate (rule A4): 85 per-fold AUROC_gnn / AUROC_hgb_prob / AUROC_prior (stratum all) from the shipped eval folders and 111's
pooled MOMA AUROC on the shared samples are re-derived from the same inputs and must agree to 1e-9.
Usage: python 139_oof_intersection_common_vs_own.py [--out DIR] [--nb 2000]
Env: crcgem_gnn (pandas, numpy, scipy, scikit-learn). CPU only.
"""
import argparse, glob, json, os, time
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

R = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
OWN = f"{R}/result/gnn_20261030_tolpatch"
MEDIA = ["DMEMF12_Gibco11320", "RPMI1640_Gibco11875"]
CM = f"{R}/result/common_medium_20261030"
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=f"{CM}/oof_intersection_{time.strftime('%Y%m%d_%H%M')}")
ap.add_argument("--nb", type=int, default=2000)
args = ap.parse_args()
OUT = args.out
assert not os.path.exists(OUT), f"{OUT} exists (no overwrite)"
t0 = time.time()
GRID = {"own": dict(runs=f"{OWN}/runs_v2", dataset=f"{OWN}/dataset_v2", flat=f"{OWN}/flat_v2_expr_dsv2/fluxonly", tag="tolpatch",
                    eval=f"{OWN}/eval_grid_nomedHGB_20261031_1030")}
for M in MEDIA:
    T = f"{R}/result/gnn_fixed_{M}_20261030_tolpatch"
    GRID[M] = dict(runs=f"{T}/runs_v2", dataset=f"{T}/dataset", flat=f"{T}/flat_v2_expr/fluxonly", tag=f"cm{M}", eval=f"{T}/eval_v2_nomedHGB_20261102_2023")
GV = {"A": "nomed_fixedcos30", "B": "nomed_meanmax_fixedcos30"}
SEEDS, FOLDS = [0, 1, 2], [0, 1, 2, 3, 4]
FALS, REPRO = [], []


def fals(check, grid, expected, observed, ok):
    FALS.append(dict(check=check, grid=grid, expected=expected, observed=observed, result="pass" if ok else "FAIL"))
    assert ok, f"FALSIFIER FAILED: {check} {grid} expected {expected} observed {observed}"


def auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_min(y, s):   # 111 rule: >= 20 samples, both classes
    y = np.asarray(y) == 1; return auc(y, s) if len(y) >= 20 and 0 < y.sum() < len(y) else np.nan


smp = {g: pd.read_csv(f"{GRID[g]['dataset']}/samples.tsv", sep="\t", comment="#") for g in GRID}
DEP = {g: smp[g].pivot_table(index="line", columns="symbol", values="dependency") for g in GRID}


def load_grid(g, graph, split):
    """per seed: DataFrame line, gene, symbol, label, gnn_prob, fold; checks config.json per fold."""
    out = {}
    for s in SEEDS:
        rd = f"{GRID[g]['runs']}/{graph}_{split}_{GV[graph]}_s{s}_{GRID[g]['tag']}"
        parts = []
        for k in FOLDS:
            cfg = json.load(open(f"{rd}/fold{k}/config.json"))
            ok = (cfg["dataset"].rstrip("/") == GRID[g]["dataset"] and cfg["out_root"].rstrip("/") == GRID[g]["runs"] and cfg["graph"] == graph
                  and cfg["split"] == split and cfg["seed"] == s and cfg["fold"] == k and cfg["no_medium"] is True)
            assert ok, f"config mismatch {rd}/fold{k}"
            d = pd.read_csv(f"{rd}/fold{k}/pred_test.tsv", sep="\t"); d["fold"] = k
            sj = json.load(open(f"{rd}/fold{k}/split.json")); assert sj["n_test"] == len(d), f"{rd}/fold{k} n_test"
            parts.append(d[["line", "gene", "symbol", "label", "gnn_prob", "fold"]])
        d = pd.concat(parts, ignore_index=True)
        nd = int(d.duplicated(["line", "gene"]).sum())
        fals("no sample predicted twice within one seed (all folds of the grid)", f"{g} {graph} {split} s{s}", 0, nd, nd == 0)
        out[s] = d.set_index(["line", "gene"]).sort_index()
    base = out[SEEDS[0]]
    for s in SEEDS[1:]:
        assert out[s].index.equals(base.index) and (out[s].fold == base.fold).all() and (out[s].label == base.label).all(), f"{g} {graph} {split}: seeds differ in test sets"
    return out


def test_lines(g, graph, split, k):
    return set(json.load(open(f"{GRID[g]['runs']}/{graph}_{split}_{GV[graph]}_s0_{GRID[g]['tag']}/fold{k}/split.json"))["test_lines"])


def load_flat(g, split, folds_of):
    """hgb_prob of the non-val rows; asserts non-val rows == GNN test set of the fold (folds_of: Series fold indexed by line, gene)."""
    parts = []
    for k in FOLDS:
        f = pd.read_csv(f"{GRID[g]['flat']}/{split}/fold{k}.tsv", sep="\t", comment="#")
        f = f[~f.is_val.astype(bool)].set_index(["line", "gene"])
        gnn_k = set(folds_of.index[folds_of == k])
        fals("flat non-val rows of the fold == GNN test set of the fold (fold-matched gradient-boosting baseline)", f"{g} {split} fold{k}", len(gnn_k),
             f"{len(f)} rows, {len(set(f.index) ^ gnn_k)} differing, NaN hgb {int(f.hgb_prob.isna().sum())}", set(f.index) == gnn_k and f.hgb_prob.notna().all() and not f.index.duplicated().any())
        parts.append(f[["hgb_prob"]].assign(fold=k))
    return pd.concat(parts).sort_index()


def per_gene_base_rate(g, graph, idx_df):
    """85 'prior' for the cell-line-disjoint split: per fold, mean dependency over the non-test cell lines."""
    pr = pd.Series(np.nan, index=idx_df.index)
    for k in FOLDS:
        tl = test_lines(g, graph, "line", k); trl = [l for l in DEP[g].index if l not in tl]
        p = DEP[g].loc[trl].mean(); m = idx_df.fold == k
        pr[m] = idx_df.symbol[m].map(p).fillna(float(DEP[g].loc[trl].stack().mean())).to_numpy()
    return pr


# ---------------- reproduction gate: 85 per-fold values and 111 pooled MOMA ----------------
inter_all = pd.read_csv(f"{CM}/raw/intersection_samples.tsv", sep="\t", comment="#")
ov111 = pd.read_csv(f"{CM}/overall.tsv", sep="\t", comment="#").set_index("medium")
cache = {}
for g in GRID:
    pfs = pd.read_csv(f"{GRID[g]['eval']}/per_fold_seed.tsv", sep="\t"); pfs = pfs[pfs.stratum == "all"]
    for graph in ("A", "B"):
        for split in ("gene", "line"):
            P = load_grid(g, graph, split); H = load_flat(g, split, P[0].fold)
            cache[(g, graph, split)] = (P, H)
            pr = per_gene_base_rate(g, graph, P[0]) if split == "line" else None
            maxd = 0.0; n = 0
            for s in SEEDS:
                for k in FOLDS:
                    d = P[s][P[s].fold == k]; y = d.label.to_numpy() == 1
                    row = pfs[(pfs.graph == graph) & (pfs.split == split) & (pfs.seed == s) & (pfs.fold == k) & pfs.variant.str.startswith(GV[graph])]
                    assert len(row) == 1, (g, graph, split, s, k, len(row))
                    vals = [(auc(y, d.gnn_prob), row.AUROC_gnn.iloc[0]), (auc(y, H.loc[d.index, "hgb_prob"]), row.AUROC_hgb_prob.iloc[0])]
                    if split == "line":
                        vals.append((auc(y, pr[d.index]), row.AUROC_prior.iloc[0]))
                    for a, b in vals:
                        maxd = max(maxd, abs(a - b)); n += 1
            REPRO.append(dict(quantity=f"85 per-fold AUROC_gnn, AUROC_hgb_prob{', AUROC_prior' if split == 'line' else ''} (stratum all, every seed x fold)",
                              grid=g, graph=graph, split=split, source=f"{GRID[g]['eval']}/per_fold_seed.tsv", n_values=n, max_abs_diff=maxd, result="pass" if maxd < 1e-9 else "FAIL"))
            assert maxd < 1e-9, REPRO[-1]
for M in MEDIA:
    I = inter_all[inter_all.medium == M]
    for gname, col, key in (("own", "score_own", "pooled_auroc_own_on_intersection"), (M, "score_common", "pooled_auroc_common_on_intersection")):
        # MOMA score as 111 computed it in memory: 1 - clip(growth_ratio) of the dataset samples.tsv (the written raw table differs by <= 1.1e-16
        # in some rows, which changes ties; the raw-table value is recorded as an informational row)
        x = 1 - smp[gname].set_index(["line", "gene"]).loc[list(zip(I.line, I.gene)), "growth_ratio"].clip(0, 1).to_numpy()
        a = auc(I.label_own == 1, x); b = float(ov111.loc[M, key]); d = abs(a - b)
        REPRO.append(dict(quantity=f"111 {key} (MOMA; score = 1 - clip(growth_ratio) of samples.tsv on the 111 shared samples)", grid=M, graph="-", split="-", source=f"{CM}/overall.tsv", n_values=1, max_abs_diff=d, result="pass" if d < 1e-9 else "FAIL"))
        assert d < 1e-9, REPRO[-1]
        a2 = auc(I.label_own == 1, I[col]); n_ulp = int((x != I[col].to_numpy()).sum())
        REPRO.append(dict(quantity=f"informational: same AUROC from the written {col} column of raw/intersection_samples.tsv ({n_ulp} rows differ from samples.tsv by <= {np.abs(x - I[col].to_numpy()).max():.1e}, float round-off changes ties)",
                          grid=M, graph="-", split="-", source=f"{CM}/raw/intersection_samples.tsv", n_values=1, max_abs_diff=abs(a2 - b), result="informational"))
print(f"reproduction gate passed ({len(REPRO)} rows) {time.time()-t0:.0f}s", flush=True)

# ---------------- assemble the shared-sample tables ----------------
rng = np.random.default_rng(0)
pooled, percl, perfold, carry, rawrows, diag = [], [], [], [], [], []
for M in MEDIA:
    I = inter_all[inter_all.medium == M].set_index(["line", "gene"]).sort_index()
    fals("shared samples: count of 111 raw/intersection_samples.tsv equals 111 overall.tsv", M, int(ov111.loc[M, "intersection"]), len(I), len(I) == int(ov111.loc[M, "intersection"]) and not I.index.duplicated().any())
    fals("label identical own vs common in 111 table", M, 0, int((I.label_own != I.label_common).sum()), (I.label_own == I.label_common).all())
    moma = {}
    for gname, col in (("own", "score_own"), (M, "score_common")):
        sm = smp[gname].set_index(["line", "gene"]); x = 1 - sm.loc[I.index, "growth_ratio"].clip(0, 1)
        dmax = float(np.abs(x.to_numpy() - I[col].to_numpy()).max()); moma[col] = x.to_numpy()
        fals("MOMA score of 111 == 1 - clip(growth_ratio) of the dataset samples.tsv (all shared samples)", f"{M} {col}", "max |diff| < 1e-9", f"{dmax:.2e} over {len(I)}", dmax < 1e-9)
    for split in ("gene", "line"):
        cols = {}
        y = (I.label_own == 1).to_numpy()
        for graph in ("A", "B"):
            for gname, side in (("own", "own"), (M, "common")):
                P, H = cache[(gname, graph, split)]
                union = P[0].index
                fals("pred_test union size = GNN sample set of the dataset", f"{gname} {graph} {split}", int(ov111.loc[M, "own_gnn_samples" if side == "own" else "common_gnn_samples"]), len(union),
                     len(union) == int(ov111.loc[M, "own_gnn_samples" if side == "own" else "common_gnn_samples"]))
                cnt = {s: P[s].index.value_counts().reindex(I.index).fillna(0).astype(int) for s in SEEDS}
                for s in SEEDS:
                    c = cnt[s].value_counts().to_dict()
                    fals("every shared sample has exactly one out-of-fold prediction (seed)", f"{gname} {graph} {split} s{s}", f"{{1: {len(I)}}}", str(c), c == {1: len(I)})
                for s in SEEDS:
                    lab = P[s].loc[I.index, "label"].to_numpy() == 1
                    assert (lab == y).all(), f"label mismatch {gname} {graph} {split} s{s}"
                gm = np.mean([P[s].loc[I.index, "gnn_prob"].to_numpy() for s in SEEDS], 0)
                cols[f"gnn_{graph}_{side}"] = gm; cols[f"fold_{graph}_{side}"] = P[0].loc[I.index, "fold"].to_numpy()
                hgb = H.loc[I.index, "hgb_prob"].to_numpy(); assert not np.isnan(hgb).any()
                if f"hgb_{side}" in cols:
                    assert np.array_equal(cols[f"hgb_{side}"], hgb)   # HGB does not depend on the graph
                cols[f"hgb_{side}"] = hgb
                if split == "line":
                    pg = per_gene_base_rate(gname, graph, P[0]).loc[I.index].to_numpy(); assert not np.isnan(pg).any()
                    cols[f"base_{side}"] = pg
        cols["moma_own"] = moma["score_own"]; cols["moma_common"] = moma["score_common"]   # = 111 scores recomputed from samples.tsv (reproduces 111 overall.tsv exactly)
        D = pd.DataFrame(cols, index=I.index); D["label"] = y.astype(int); D["symbol"] = I.symbol_own.to_numpy()
        # fold carry-over (observed) and line-split fold identity
        for graph in ("A", "B"):
            same = int((D[f"fold_{graph}_own"] == D[f"fold_{graph}_common"]).sum())
            carry.append(dict(medium=M, graph=graph, split=split, shared_samples=len(D), same_test_fold=same, fraction=same / len(D)))
            if split == "line":
                for k in FOLDS:
                    fals("cell-line-disjoint split: identical test cell lines per fold own vs common", f"{M} {graph} fold{k}", "equal", "equal" if test_lines("own", graph, "line", k) == test_lines(M, graph, "line", k) else "differ",
                         test_lines("own", graph, "line", k) == test_lines(M, graph, "line", k))
                fals("cell-line-disjoint split: every shared sample in the same fold in both grids", f"{M} {graph}", len(D), same, same == len(D))
        if split == "line":
            # 85 builds the dependency table from the dataset's samples.tsv, whose rows differ between media (exclusions), so the value can differ
            FALS.append(dict(check="informational: per-gene base rate (85 prior) own vs common, same test cell lines, dependency table from each dataset samples.tsv",
                             grid=M, expected="-", observed=f"max |diff| {np.abs(D.base_own - D.base_common).max():.2e}, rows differing {int((D.base_own != D.base_common).sum())} of {len(D)}", result="informational"))
        # consistency gate: same sample array, 3 seeds, own pooled recomputed here
        fals("consistency gate: own and common columns on the identical shared-sample array, seed mean over 3 seeds each", f"{M} {split}", len(I), len(D), D.index.equals(I.index))
        PAIRS = [("gnn_A", "GNN A′"), ("gnn_B", "GNN B"), ("moma", "MOMA"), ("hgb", "gradient-boosting baseline without medium input")]
        if split == "line":
            PAIRS.append(("base", "per-gene base rate (85 prior, dependency table of each dataset samples.tsv)"))
        # check the rank AUROC against sklearn once per column
        for c in [c for c in D.columns if c.startswith(("gnn_", "moma_", "hgb_", "base"))]:
            assert abs(auc(y, D[c]) - roc_auc_score(y, D[c])) < 1e-12
        lines = D.index.get_level_values("line").to_numpy(); ul = np.unique(lines); pos = {l: np.where(lines == l)[0] for l in ul}
        scols = [c for c in D.columns if c.startswith(("gnn_", "moma_", "hgb_", "base_"))]
        point = {c: auc(y, D[c]) for c in scols}
        boot = np.empty((args.nb, len(scols)))
        for b in range(args.nb):
            ix = np.concatenate([pos[l] for l in rng.choice(ul, len(ul))]); yb = y[ix]
            boot[b] = [auc(yb, D[c].to_numpy()[ix]) for c in scols]
        B = pd.DataFrame(boot, columns=scols)
        for pre, name in PAIRS:
            dc = B[f"{pre}_common"] - B[f"{pre}_own"]
            pooled.append(dict(medium=M, split=split, predictor=name, n_shared=len(D), positives=int(y.sum()), cell_lines=len(ul),
                               auroc_common=point[f"{pre}_common"], auroc_common_lo=B[f"{pre}_common"].quantile(.025), auroc_common_hi=B[f"{pre}_common"].quantile(.975),
                               auroc_own=point[f"{pre}_own"], auroc_own_lo=B[f"{pre}_own"].quantile(.025), auroc_own_hi=B[f"{pre}_own"].quantile(.975),
                               diff_common_minus_own=point[f"{pre}_common"] - point[f"{pre}_own"], diff_lo=dc.quantile(.025), diff_hi=dc.quantile(.975),
                               boot_fraction_diff_gt0=float((dc > 0).mean()), n_boot=args.nb))
        if split == "gene":
            pooled.append(dict(medium=M, split=split, predictor="per-gene base rate", n_shared=len(D), positives=int(y.sum()), cell_lines=len(ul),
                               note="not defined for the gene-disjoint split: the test gene has no training rows (85 computes it for the cell-line-disjoint split only)"))
        # per cell line
        for l in ul:
            x = D.iloc[pos[l]]; r = dict(medium=M, split=split, cell_line=l, n_shared=len(x), positives=int(x.label.sum()))
            for pre, name in PAIRS:
                a, b = auc_min(x.label, x[f"{pre}_common"]), auc_min(x.label, x[f"{pre}_own"])
                r[f"{pre}_common"], r[f"{pre}_own"], r[f"{pre}_diff"] = a, b, a - b
            percl.append(r)
        # cell-line-disjoint split: per-fold paired difference on the per-fold shared samples
        if split == "line":
            for graph in ("A", "B"):
                for pre, name in [(f"gnn_{graph}", f"GNN {graph}{chr(8242) if graph == 'A' else ''}"), ("moma", "MOMA"), ("hgb", "gradient-boosting baseline without medium input"), ("base", "per-gene base rate (85 prior)")]:
                    dv = []
                    for k in FOLDS:
                        x = D[D[f"fold_{graph}_own"] == k]; a, b = auc(x.label, x[f"{pre}_common"]), auc(x.label, x[f"{pre}_own"]); dv.append(a - b)
                        perfold.append(dict(medium=M, graph_folds=graph, predictor=name, fold=k, n_shared=len(x), positives=int(x.label.sum()), auroc_common=a, auroc_own=b, diff_common_minus_own=a - b))
                    dv = np.array(dv); h = stats.t.ppf(0.975, len(dv) - 1) * dv.std(ddof=1) / np.sqrt(len(dv))
                    perfold.append(dict(medium=M, graph_folds=graph, predictor=name, fold="mean", n_shared=len(D), positives=int(D.label.sum()), auroc_common=np.nan, auroc_own=np.nan,
                                        diff_common_minus_own=dv.mean(), t_lo=dv.mean() - h, t_hi=dv.mean() + h, folds_common_higher=int((dv > 0).sum())))
        # diagnostic (pooled is not the fold mean): per grid, AUROC within each of its own test folds on the shared samples, fold mean, and
        # median seed-mean gnn_prob per fold (cross-fold score shift enters the pooled AUROC, not the fold mean; Forman & Scholz 2010)
        for graph in ("A", "B"):
            for side in ("own", "common"):
                f = D[f"fold_{graph}_{side}"]; s = D[f"gnn_{graph}_{side}"]
                pf = [auc(y[f == k], s[f == k]) for k in FOLDS]
                diag.append(dict(medium=M, split=split, graph=graph, grid=side, pooled_auroc=point[f"gnn_{graph}_{side}"], fold_mean_auroc=float(np.mean(pf)),
                                 pooled_minus_fold_mean=point[f"gnn_{graph}_{side}"] - float(np.mean(pf)), per_fold_auroc=";".join(f"{x:.4f}" for x in pf),
                                 per_fold_n=";".join(str(int((f == k).sum())) for k in FOLDS), per_fold_median_gnn_prob=";".join(f"{s[f == k].median():.3f}" for k in FOLDS)))
        rawrows.append(D.reset_index().assign(medium=M, split=split))
        print(f"{M} {split}: n {len(D)}; " + "; ".join(f"{p['predictor'][:6]} {p.get('auroc_common', np.nan):.4f}/{p.get('auroc_own', np.nan):.4f} d {p.get('diff_common_minus_own', np.nan):+.4f} [{p.get('diff_lo', np.nan):+.4f},{p.get('diff_hi', np.nan):+.4f}]"
                                                    for p in pooled if p["medium"] == M and p["split"] == split) + f"; {time.time()-t0:.0f}s", flush=True)

# own-only / common-only counts (no metric)
counts = ov111[["own_gnn_samples", "common_gnn_samples", "intersection", "own_only", "common_only"]].reset_index()

# ---------------- write ----------------
os.makedirs(f"{OUT}/raw")
HDR = (f"# script_python/139_oof_intersection_common_vs_own.py; design result/common_medium_20261030/gnn_design_20261031.md section 2 (2), (3b), section 3; user approval decision log 2026-11-03 ruling (3); date {time.strftime('%Y-%m-%d %H:%M %Z')}\n"
       f"# pooled out-of-fold AUROC on the samples shared with the own-medium dataset (one AUROC over the pooled out-of-fold predictions; this is NOT the fold-mean AUROC of script 85 / comparison_grid)\n"
       f"# shared samples: {CM}/raw/intersection_samples.tsv (script 111; own {OWN}/dataset_v2 GNN samples x common-medium dataset GNN samples, 81b fluxome-change filter); own-only and common-only samples: counts only, no metric\n"
       f"# GNN runs: own {OWN}/runs_v2 (*_tolpatch), common result/gnn_fixed_<medium>_20261030_tolpatch/runs_v2 (*_cm<medium>); graphs A′ (A_*_nomed_fixedcos30, attn_max) and B (B_*_nomed_meanmax_fixedcos30, mean_max); protocol v3 fixed 30 epochs cosine, no medium input; seeds 0-2 averaged per sample (seed mean of gnn_prob), folds 0-4\n"
       f"# gradient-boosting baseline without medium input (HGB, script 90 fluxonly, fold-matched to the GNN folds of the same dataset): own {OWN}/flat_v2_expr_dsv2/fluxonly, common <T>/flat_v2_expr/fluxonly; MOMA score = 1 - clipped relative growth rate, recomputed from each dataset samples.tsv as 111 does in memory (111 raw score_own / score_common equal within 1.1e-16)\n"
       f"# per-gene base rate = 85 prior (cell-line-disjoint split only; dependency table from each dataset samples.tsv, so own and common values can differ slightly); label = DepMap CRISPRGeneDependency >= 0.5; bootstrap over cell lines {args.nb} resamples, seed 0, percentile 95% interval, the same resamples for every column of one (medium, split); per-cell-line AUROC needs >= 20 samples with both classes (111 rule)\n"
       f"# gene-disjoint folds differ between own and common grids (about 60% same test fold, see fold_carryover.tsv), so the pairing is at the sample level, not per fold\n")


def write(df, name, note):
    for ext, sep in (("tsv", "\t"), ("csv", ",")):
        with open(f"{OUT}/{name}.{ext}", "w") as fh:
            fh.write(HDR + f"# {note}; n rows {len(df)}\n"); df.to_csv(fh, sep=sep, index=False)


write(pd.DataFrame(pooled), "pooled", "pooled out-of-fold AUROC on the samples shared with the own-medium dataset, common medium vs own medium per predictor; diff = common minus own; lo/hi = bootstrap over cell lines")
write(pd.DataFrame(percl), "per_cell_line", "per cell line AUROC on the shared samples, common, own and difference common minus own (NaN when < 20 samples or one class)")
write(pd.DataFrame(perfold), "per_fold_cell_line_disjoint", "cell-line-disjoint split: per-fold AUROC on the per-fold shared samples (test cell lines identical in both grids); fold 'mean' row = mean difference, 95% t interval over 5 folds (df 4); graph_folds = graph whose fold assignment is used (identical folds for A′ and B)")
write(pd.DataFrame(carry), "fold_carryover", "fraction of shared samples whose test fold is the same in the own and the common grid (observed from pred_test.tsv, seed 0)")
write(pd.DataFrame(diag), "diagnostic_pooled_vs_fold_mean", "diagnostic, GNN only: pooled out-of-fold AUROC vs the mean of per-fold AUROC within each grid own test folds on the shared samples, per-fold n and median seed-mean gnn_prob; gene-disjoint folds differ between grids, so the fold means are not paired")
write(counts, "sample_set_counts", "GNN sample sets (from 111 overall.tsv): own, common, shared, own-only, common-only; own-only and common-only get no metric")
write(pd.DataFrame(FALS), "falsifiers", "falsifier and gate checks; any FAIL stops the script before writing, so every row here passed")
write(pd.DataFrame(REPRO), "reproduction", "reproduction gate: shipped values re-derived from the same inputs (85 per_fold_seed.tsv, 111 overall.tsv)")
write(pd.concat(rawrows, ignore_index=True), "raw/shared_sample_scores", "per shared sample and split: seed-mean GNN (A′, B) common/own, out-of-fold fold per grid, HGB common/own, MOMA common/own, per-gene base rate (cell-line-disjoint), label")
print(f"falsifier rows {len(FALS)} all pass; reproduction rows {len(REPRO)} all pass; done {time.time()-t0:.0f}s -> {OUT}")
