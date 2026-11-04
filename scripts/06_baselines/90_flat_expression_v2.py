"""E4 (discussion 2026-10-29, review_gem.md section 8.5; user approved 2026-10-29 ~13:xx): per-line expression in place of
the medium, flat HGB on the masked GNN input, same samples, folds and splits as scripts 81 and 86. Script 86 is unchanged.

Question. Does line-specific molecular state (expression of the Human-GEM genes) add anything beyond the knockout
fluxome, and in particular does it move the metrics that are at chance, within-line AUROC (gene split) and
within-gene AUROC (line split)?

Variants (--variant, one or more):
  fluxonly   slog(v_KO - v_WT) of the unmasked reactions active anywhere (7,147), nothing per line.
  med        fluxonly + medium columns varying across training lines, z-scored on them (= script 86's HGB).
  expr       fluxonly + per-line expression, log2(TPM+1) of the 2,830 Human-GEM ENSG genes
             (data/depmap/crc_trainset/crc_expression.ENSG.parquet, the tINIT input), z-scored on the training lines
             with the genes whose sd over training lines > 1e-6 (input-only statistics, no labels).
  expr_med   fluxonly + expression + medium.
Model: HistGradientBoostingClassifier as in 86 (max_iter 300, lr 0.1, internal early stopping, random_state = fold).
Comparators written per fold: MOMA (score 1 - growth ratio), line-propensity comparator (mean binary label of the
fold's TRAINING samples of the same line, as script 85), both on the same test samples.
Strata in meta.json: all, MOMA-lethal (G < 0.05), G > 0.99, and for the gene split rel_nn_dist >= 0.2 (read from
--rel-root/A_gene_s0/fold<k>/pred_test.tsv, script 81's column) and rel >= 0.2 & lethal. Within-line AUROC (gene
split, lines with both classes and >= 10 samples) and within-gene AUROC (line split, genes with both classes in the
test fold) for HGB, MOMA and the line-propensity comparator.
Output: result/gnn_20261028/flat_v2_expr/<variant>/<split>/fold<k>.tsv with the columns of script 86 (line, symbol,
gene, dependency, label, growth_ratio, hgb_prob, ridge_growth (NaN here), is_val) plus lineprior and rel_nn_dist, and
fold<k>_meta.json. Script 85 can read a variant folder through --flat-root. Existing outputs are never overwritten
unless --overwrite is given (a run into an existing folder stops).
Smoke: --smoke (gene fold 0, 1,500 training samples, max_iter 10) checks IO and the metric code in ~1-2 min.
Usage: OMP_NUM_THREADS=8 nice -n 10 python 90_flat_expression_v2.py --variant expr med fluxonly expr_med
Env: crcgem_gnn. CPU only.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
ap = argparse.ArgumentParser()
ap.add_argument("--variant", nargs="+", default=["expr"], choices=["fluxonly", "med", "expr", "expr_med"])
ap.add_argument("--splits", nargs="+", default=["gene", "line"])
ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--nfold", type=int, default=5)
ap.add_argument("--mask-file", default=f"{ROOT}/result/gnn_20261028/growth_mask_v2_20261029.json")
ap.add_argument("--dataset", default=f"{ROOT}/result/gnn_20261028/dataset")
ap.add_argument("--expression", default=f"{ROOT}/data/depmap/crc_trainset/crc_expression.ENSG.parquet")
ap.add_argument("--rel-root", default=f"{ROOT}/result/gnn_20261028/runs_v2", help="folder holding A_gene_s0/fold<k>/pred_test.tsv")
ap.add_argument("--rel-run", default="A_gene_s0", help="run folder under --rel-root supplying rel_nn_dist (default unchanged; tolpatch grid: A_gene_nomed_fixedcos30_s0_tolpatch)")
ap.add_argument("--n-samples", type=int, default=18573, help="expected sample count after the fluxome-change filter (asserted; 2026-10-28 dataset 18573; pass the measured count of another dataset)")
ap.add_argument("--out", default=f"{ROOT}/result/gnn_20261028/flat_v2_expr")
ap.add_argument("--max-iter", type=int, default=300)
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()
if args.smoke:
    args.splits, args.folds, args.max_iter = ["gene"], [0], 10
    args.out = args.out.rstrip("/") + "_smoke"
D = args.dataset; t0 = time.time()
def say(*a): print(*a, flush=True)
def slog(x): return np.sign(x) * np.log1p(np.abs(x))
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    return float(roc_auc_score(y, s)) if len(y) >= 10 and y.min() != y.max() else float("nan")

# ---------------- data, identical to 81/86 ----------------
meta = json.load(open(f"{D}/scaffold_meta.json")); NR = len(meta["reactions"])
gene_rxns = meta["gene_rxns"]; BIO = meta["reactions"].index(meta["biomass"])
MASKJ = json.load(open(args.mask_file)); MASK = np.array(sorted(set(MASKJ["mask_index"])), int)
assert BIO in set(MASK.tolist()); KEEP = np.ones(NR, bool); KEEP[MASK] = False
lines = sorted(f[:-4] for f in os.listdir(f"{D}/lines")); L = {}
for ln in lines:
    z = np.load(f"{D}/lines/{ln}.npz"); d = {k: z[k] for k in z.files}; th = 1e-6 / float(d["mu_ref"]); d["th"] = th
    d["vwt"] = np.where(np.abs(d["vwt"]) > th, d["vwt"], 0).astype(np.float32)
    d["V"] = np.where(np.abs(d["V"]) > th, d["V"], 0).astype(np.float32); L[ln] = d
samples = [(ln, i) for ln in lines for i in range(len(L[ln]["genes"])) if np.abs(L[ln]["V"][i] - L[ln]["vwt"]).max() > L[ln]["th"]]
sym = np.array([L[ln]["symbols"][i] for ln, i in samples]); gid = np.array([L[ln]["genes"][i] for ln, i in samples])
dep = np.array([float(L[ln]["dependency"][i]) for ln, i in samples], np.float32); lab = (dep >= 0.5).astype(int)
sline = np.array([ln for ln, _ in samples]); allidx = np.arange(len(samples))
G = np.array([L[ln]["V"][i, BIO] for ln, i in samples])   # growth ratio, never a feature
Vall = np.stack([L[ln]["V"][i] for ln, i in samples]); VW = np.stack([L[ln]["vwt"] for ln, _ in samples])
anyact = (np.abs(Vall).max(0) > 0) | (np.abs(VW).max(0) > 0); cols = np.where(KEEP & anyact)[0]
XD = slog(Vall[:, cols] - VW[:, cols]).astype(np.float32)
del Vall, VW
assert len(samples) == args.n_samples, (len(samples), args.n_samples)
say(f"samples {len(samples)}; fluxome features {len(cols)} (mask {len(MASK)}); {time.time()-t0:.0f}s")
def med_feat(lb):
    return np.concatenate([(lb == 0), (lb <= -999), np.where((lb < 0) & (lb > -999), slog(-lb * 1000), 0)]).astype(np.float32)
MEDRAW = np.stack([med_feat(L[ln]["medium_lb"]) for ln in lines]).astype(np.float64); lpos = {ln: k for k, ln in enumerate(lines)}
# per-line expression, the tINIT input matrix (genes x lines), aligned to the dataset lines
EX = pd.read_parquet(args.expression)
missing = [ln for ln in lines if ln not in EX.columns]
assert not missing, f"lines without expression: {missing}"
EXRAW = EX[lines].to_numpy(dtype=np.float64).T   # lines x genes
say(f"expression {EXRAW.shape[1]} genes x {EXRAW.shape[0]} lines from {args.expression}")


def group_folds(groups, k):
    ug, inv = np.unique(groups, return_inverse=True); n = np.bincount(inv)
    order = sorted(range(len(ug)), key=lambda j: (-n[j], ug[j]))
    fn = np.zeros(k); fo = np.zeros(len(ug), int)
    for t, j in enumerate(order):
        f = t if t < k else int(np.argmin(fn)); fo[j] = f; fn[f] += n[j]
    return fo[inv]


def val_groups(groups, frac, seed, keep_largest=4):
    ug, inv = np.unique(groups, return_inverse=True); n = np.bincount(inv)
    big = set(np.argsort(-n)[:keep_largest])
    cand = [j for j in np.random.default_rng(seed).permutation(len(ug)) if j not in big]
    sel, tot = set(), 0
    for j in cand:
        if tot >= frac * len(groups):
            break
        sel.add(j); tot += n[j]
    return np.isin(inv, list(sel))


def split_fold(split, fold):
    if split == "gene":
        gkey = np.array([",".join(map(str, sorted(gene_rxns.get(g, [])))) for g in gid]); fo = group_folds(gkey, args.nfold)
        test, rest = allidx[fo == fold], allidx[fo != fold]; v = val_groups(gkey[rest], 0.2, fold); return rest[~v], rest[v], test
    perm = list(np.random.default_rng(0).permutation(lines)); lf = {ln: k % args.nfold for k, ln in enumerate(perm)}
    test_lines = sorted(ln for ln in lines if lf[ln] == fold); rest_lines = sorted(ln for ln in lines if lf[ln] != fold)
    val_lines = sorted(np.random.default_rng(fold).choice(rest_lines, int(round(0.2 * len(rest_lines))), replace=False))
    test = allidx[np.isin(sline, test_lines)]; val = allidx[np.isin(sline, val_lines)]
    train = allidx[~np.isin(sline, test_lines + list(val_lines))]; return train, val, test


def zscore_lines(RAW, tr_rows):
    sd = RAW[tr_rows].std(0); keep = sd > 1e-6
    Z = (RAW[:, keep] - RAW[tr_rows][:, keep].mean(0)) / sd[keep]
    return Z, int(keep.sum())


def grouped_auc(df, by, score):
    r = [auc(g.label, g[score]) for _, g in df.groupby(by)]
    r = [x for x in r if x == x]
    return (float(np.mean(r)) if r else float("nan")), len(r)


def rel_for(split, fold):
    f = f"{args.rel_root}/{args.rel_run}/fold{fold}/pred_test.tsv"
    if split != "gene" or not os.path.exists(f):
        return None
    r = pd.read_csv(f, sep="\t", usecols=["line", "gene", "rel_nn_dist"])
    return dict(zip(zip(r.line, r.gene), r.rel_nn_dist))


for variant in args.variant:
    for split in args.splits:
        od = f"{args.out}/{variant}/{split}"
        if os.path.exists(od) and os.listdir(od) and not args.overwrite:
            say(f"STOP: {od} exists and is not empty; pass --overwrite or choose --out"); sys.exit(2)
        os.makedirs(od, exist_ok=True)
        for fold in args.folds:
            tf = time.time(); train, val, test = split_fold(split, fold)
            if args.smoke:
                train = np.random.default_rng(0).choice(train, 1500, replace=False)
            tr_rows = np.isin(lines, sorted(set(sline[train])))
            parts, nmed, nexp = [XD], 0, 0
            if variant in ("med", "expr_med"):
                MEDZ, nmed = zscore_lines(MEDRAW, tr_rows); parts.append(MEDZ[[lpos[ln] for ln in sline]].astype(np.float32))
            if variant in ("expr", "expr_med"):
                EXZ, nexp = zscore_lines(EXRAW, tr_rows); parts.append(EXZ[[lpos[ln] for ln in sline]].astype(np.float32))
            X = np.concatenate(parts, 1)
            hgb = HistGradientBoostingClassifier(max_iter=args.max_iter, learning_rate=0.1, early_stopping=True, validation_fraction=0.1,
                                                 n_iter_no_change=20, random_state=fold).fit(X[train], lab[train])
            ph = hgb.predict_proba(X)[:, 1]; th_ = time.time() - tf
            # comparators on the same samples
            lp = pd.Series(lab[train]).groupby(sline[train]).mean()
            lineprior = pd.Series(sline).map(lp).fillna(lab[train].mean()).to_numpy()
            moma = 1 - np.clip(G, 0, 1)
            rel = rel_for(split, fold)
            relv = np.array([rel.get((ln, g), np.nan) for ln, g in zip(sline, gid)]) if rel else np.full(len(samples), np.nan)
            ev = np.concatenate([val, test]); isval = np.concatenate([np.ones(len(val), bool), np.zeros(len(test), bool)])
            df = pd.DataFrame(dict(line=sline[ev], symbol=sym[ev], gene=gid[ev], dependency=dep[ev], label=lab[ev], growth_ratio=G[ev],
                                   hgb_prob=ph[ev], ridge_growth=np.nan, is_val=isval, lineprior=lineprior[ev], rel_nn_dist=relv[ev]))
            df.to_csv(f"{od}/fold{fold}.tsv", sep="\t", index=False)
            T = df[~df.is_val].copy(); T["moma"] = 1 - np.clip(T.growth_ratio, 0, 1)
            strata = {"all": np.ones(len(T), bool), "lethal": T.growth_ratio.to_numpy() < 0.05, "G>0.99": T.growth_ratio.to_numpy() > 0.99}
            if rel:
                strata["rel>=0.2"] = T.rel_nn_dist.to_numpy() >= 0.2
                strata["rel>=0.2&lethal"] = strata["rel>=0.2"] & strata["lethal"]
            res = {}
            for sn, m in strata.items():
                S = T[m]; row = dict(n=int(m.sum()), pos_rate=float(S.label.mean()) if len(S) else float("nan"))
                for nm, sc in [("hgb", "hgb_prob"), ("moma", "moma"), ("lineprior", "lineprior")]:
                    row[f"auroc_{nm}"] = auc(S.label, S[sc])
                    if split == "gene":
                        row[f"within_line_{nm}"], row["n_lines"] = grouped_auc(S, "line", sc)
                    else:
                        row[f"within_gene_{nm}"], row["n_genes"] = grouped_auc(S, "gene", sc)
                res[sn] = row
            metaj = dict(variant=variant, split=split, fold=fold, smoke=args.smoke, n_train=len(train), n_val=len(val), n_test=len(test),
                         n_features=int(X.shape[1]), n_flux_features=len(cols), n_medium_cols=nmed, n_expression_cols=nexp,
                         hgb_iters=int(hgb.n_iter_), hgb_auroc_val=auc(lab[val], ph[val]), strata=res, seconds_hgb=round(th_),
                         seconds_total=round(time.time() - tf), mask_file=args.mask_file, expression=args.expression,
                         rel_source=f"{args.rel_root}/A_gene_s0/fold{fold}/pred_test.tsv" if rel else None, date=time.strftime("%Y-%m-%d %H:%M"))
            json.dump(metaj, open(f"{od}/fold{fold}_meta.json", "w"), indent=1)
            a = res["all"]; w = "within_line" if split == "gene" else "within_gene"
            say(f"{variant} {split} fold {fold}: train {len(train)} val {len(val)} test {len(test)} feat {X.shape[1]} iters {hgb.n_iter_}; "
                f"all AUROC hgb {a['auroc_hgb']:.3f} moma {a['auroc_moma']:.3f} lineprior {a['auroc_lineprior']:.3f}; "
                f"{w} hgb {a[f'{w}_hgb']:.3f} moma {a[f'{w}_moma']:.3f}; lethal hgb {res['lethal']['auroc_hgb']:.3f} "
                f"{w} {res['lethal'][f'{w}_hgb']:.3f}; {time.time()-tf:.0f}s")
say(f"done {time.time()-t0:.0f}s -> {args.out}")
