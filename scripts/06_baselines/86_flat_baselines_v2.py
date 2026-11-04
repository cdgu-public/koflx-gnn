"""Flat (non-graph) baselines on the masked GNN input, per fold, same samples and splits as script 81 (2026-10-29).

Per split (gene, line) and fold 0-4, with the split code of scripts 73/80/81 (group_folds / val_groups, line folds):
- HGB: HistGradientBoostingClassifier on slog(v_KO - v_WT) of the reactions that are active anywhere and are not in
  the growth mask (7,147) plus the medium columns that vary across the training lines, z-scored on them (as 81).
  Trained on the fold's training part with the binary label (dependency >= 0.5); sklearn's internal early stopping
  (validation_fraction 0.1 of the training part) because the classifier takes no external validation set.
- Growth probe: Ridge(alpha 10) from [slog(v_KO), slog(v_KO - v_WT)] of the same reactions to the MOMA growth ratio,
  trained on the training part. Reported as R2 and Spearman on the test part. Documents that the growth ratio is
  recoverable from the fluxome without being an input.
Output: result/gnn_20261028/flat_v2/<split>/fold<k>.tsv (line, symbol, gene, dependency, label, growth_ratio,
hgb_prob, ridge_growth, is_val) and fold<k>_meta.json (sizes, R2, Spearman, AUROC of 1 - ridge_growth and of
hgb_prob on test, timings). Script 85 reads these.
Usage: OMP_NUM_THREADS=16 python 86_flat_baselines_v2.py [--mask-file ...] [--splits gene line] [--folds 0 1 2 3 4]
Env: crcgem_gnn. CPU only.
"""
import argparse, json, os, time
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--mask-file", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/growth_mask_v2_20261029.json")
ap.add_argument("--splits", nargs="+", default=["gene", "line"])
ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--nfold", type=int, default=5)
ap.add_argument("--out", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/flat_v2")
ap.add_argument("--dataset", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/dataset",
                help="dataset folder (script 64 / 64d); default = the 2026-10-28 dataset, unchanged behavior (2026-10-30 tolpatch: result/gnn_20261030_tolpatch/dataset)")
args = ap.parse_args()
ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"; D = args.dataset.rstrip("/")
assert not (os.path.exists(f"{args.out}/gene/fold0.tsv") or os.path.exists(f"{args.out}/line/fold0.tsv")) or args.out.endswith("flat_v2"), f"{args.out} already holds baselines (no overwrite)"
t0 = time.time()
def say(*a): print(*a, flush=True)
def slog(x): return np.sign(x) * np.log1p(np.abs(x))

# ---------------- data, identical to 73/81 ----------------
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
G = np.array([L[ln]["V"][i, BIO] for ln, i in samples])   # growth ratio (target of the probe; never a feature)
Vall = np.stack([L[ln]["V"][i] for ln, i in samples]); VW = np.stack([L[ln]["vwt"] for ln, _ in samples])
anyact = (np.abs(Vall).max(0) > 0) | (np.abs(VW).max(0) > 0); cols = np.where(KEEP & anyact)[0]
XD = slog(Vall[:, cols] - VW[:, cols]).astype(np.float32); XK = slog(Vall[:, cols]).astype(np.float32)
del Vall, VW
say(f"samples {len(samples)}; features {len(cols)} reactions (mask {len(MASK)} removed); {time.time()-t0:.0f}s")
def med_feat(lb):
    return np.concatenate([(lb == 0), (lb <= -999), np.where((lb < 0) & (lb > -999), slog(-lb * 1000), 0)]).astype(np.float32)
MEDRAW = np.stack([med_feat(L[ln]["medium_lb"]) for ln in lines]).astype(np.float64); lpos = {ln: k for k, ln in enumerate(lines)}


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


for split in args.splits:
    os.makedirs(f"{args.out}/{split}", exist_ok=True)
    for fold in args.folds:
        tf = time.time(); train, val, test = split_fold(split, fold)
        tr_rows = np.isin(lines, sorted(set(sline[train]))); sd = MEDRAW[tr_rows].std(0); keep = sd > 1e-6
        MEDZ = (MEDRAW[:, keep] - MEDRAW[tr_rows][:, keep].mean(0)) / sd[keep]
        MEDS = MEDZ[[lpos[ln] for ln in sline]].astype(np.float32)
        X = np.concatenate([XD, MEDS], 1)
        hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.1,
                                             n_iter_no_change=20, random_state=fold).fit(X[train], lab[train])
        ph = hgb.predict_proba(X)[:, 1]; th_ = time.time() - tf
        XP = np.concatenate([XK, XD], 1)
        rg = Ridge(alpha=10.0).fit(XP[train], G[train]); pg = rg.predict(XP)
        r2 = 1 - ((pg[test] - G[test])**2).sum() / ((G[test] - G[test].mean())**2).sum()
        ev = np.concatenate([val, test]); isval = np.concatenate([np.ones(len(val), bool), np.zeros(len(test), bool)])
        pd.DataFrame(dict(line=sline[ev], symbol=sym[ev], gene=gid[ev], dependency=dep[ev], label=lab[ev], growth_ratio=G[ev],
                          hgb_prob=ph[ev], ridge_growth=pg[ev], is_val=isval)).to_csv(f"{args.out}/{split}/fold{fold}.tsv", sep="\t", index=False)
        metaj = dict(split=split, fold=fold, n_train=len(train), n_val=len(val), n_test=len(test), n_features_hgb=int(X.shape[1]),
                     n_medium_cols=int(keep.sum()), hgb_iters=int(hgb.n_iter_), hgb_auroc_test=float(roc_auc_score(lab[test], ph[test])),
                     hgb_auroc_val=float(roc_auc_score(lab[val], ph[val])), probe_r2_test=float(r2),
                     probe_spearman_test=float(spearmanr(pg[test], G[test])[0]),
                     probe_auroc_label_test=float(roc_auc_score(lab[test], -pg[test])), moma_auroc_test=float(roc_auc_score(lab[test], -G[test])),
                     probe_auroc_moma_lethal_test=float(roc_auc_score(G[test] < 0.05, -pg[test])), seconds_hgb=round(th_), seconds_total=round(time.time() - tf),
                     mask_file=args.mask_file, mask_reactions=MASKJ["mask_reactions"])
        json.dump(metaj, open(f"{args.out}/{split}/fold{fold}_meta.json", "w"), indent=1)
        say(f"{split} fold {fold}: train {len(train)} val {len(val)} test {len(test)}; HGB iters {hgb.n_iter_} test AUROC {metaj['hgb_auroc_test']:.3f} "
            f"(MOMA {metaj['moma_auroc_test']:.3f}); probe R2 {r2:.3f} Spearman {metaj['probe_spearman_test']:.3f} AUROC-vs-label {metaj['probe_auroc_label_test']:.3f}; {time.time()-tf:.0f}s")
say(f"done {time.time()-t0:.0f}s")
