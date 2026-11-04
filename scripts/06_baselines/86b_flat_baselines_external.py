"""Gradient-boosting baseline and linear probe of growth, refit per colorectal cell-line-disjoint fold and applied to the
external validation set of non-colorectal cell lines (term pending ratification; plan
result/gnn_20261028/discussion_20261030/external_eval_plan.md section 5), 2026-10-30. Script 86 saves no models, so the
fits are repeated here with its code (data preparation, split_fold, hyperparameters, random_state = fold) and the
colorectal fold test AUROC is written to fold<k>_meta.json so that the refit can be checked against
result/gnn_20261028/flat_v2/line/fold<k>_meta.json (hgb_auroc_test, probe_r2_test).

Two gradient-boosting variants per fold: (a) hgb_nomed_prob = masked delta fluxome only (matched to the no-medium GNN;
the plan's paired comparator); (b) hgb_prob = masked delta fluxome + medium columns z-scored over the fold's training
cell lines (dictionary definition, script 86). Probe: Ridge(alpha 10) on [slog(v_KO), slog(v_KO - v_WT)] -> growth ratio.
Feature columns = reactions active anywhere in the COLORECTAL dataset and not in the growth mask (7,147), applied as-is
to the external fluxomes (same scaffold index, verified by 64c). External samples = those passing the 81b fluxome-change
filter. NO metric on the external samples is computed or printed here.
Usage: OMP_NUM_THREADS=8 python 86b_flat_baselines_external.py [--folds 0 1 2 3 4] [--external-dataset ...] [--out ...]
Env: crcgem_gnn. CPU only. Output: <out>/line/fold<k>.tsv (external rows: line, symbol, gene, dependency, label, has_label,
growth_ratio, hgb_prob, hgb_nomed_prob, ridge_growth) and fold<k>_meta.json.
"""
import argparse, hashlib, json, os, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
ap = argparse.ArgumentParser()
ap.add_argument("--mask-file", default=f"{ROOT}/result/gnn_20261028/growth_mask_v2_20261029.json")
ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
ap.add_argument("--nfold", type=int, default=5)
ap.add_argument("--train-dataset", default=f"{ROOT}/result/gnn_20261028/dataset")
ap.add_argument("--external-dataset", default=f"{ROOT}/result/external_20261030/dataset")
ap.add_argument("--out", default=f"{ROOT}/result/external_20261030/flat")
args = ap.parse_args()
D = args.train_dataset; t0 = time.time()
def say(*a): print(*a, flush=True)
def slog(x): return np.sign(x) * np.log1p(np.abs(x))
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()


def load(Dd, meta, KEEP):
    lines = sorted(f[:-4] for f in os.listdir(f"{Dd}/lines")); L = {}
    for ln in lines:
        z = np.load(f"{Dd}/lines/{ln}.npz"); d = {k: z[k] for k in z.files}; th = 1e-6 / float(d["mu_ref"]); d["th"] = th
        d["vwt"] = np.where(np.abs(d["vwt"]) > th, d["vwt"], 0).astype(np.float32)
        d["V"] = np.where(np.abs(d["V"]) > th, d["V"], 0).astype(np.float32); L[ln] = d
    samples = [(ln, i) for ln in lines for i in range(len(L[ln]["genes"])) if np.abs(L[ln]["V"][i] - L[ln]["vwt"]).max() > L[ln]["th"]]
    return lines, L, samples


# ---------------- colorectal training data, identical to 86 ----------------
meta = json.load(open(f"{D}/scaffold_meta.json")); NR = len(meta["reactions"]); BIO = meta["reactions"].index(meta["biomass"])
MASKJ = json.load(open(args.mask_file)); MASK = np.array(sorted(set(MASKJ["mask_index"])), int)
assert BIO in set(MASK.tolist()); KEEP = np.ones(NR, bool); KEEP[MASK] = False
lines, L, samples = load(D, meta, KEEP)
sym = np.array([L[ln]["symbols"][i] for ln, i in samples]); gid = np.array([L[ln]["genes"][i] for ln, i in samples])
dep = np.array([float(L[ln]["dependency"][i]) for ln, i in samples], np.float32); lab = (dep >= 0.5).astype(int)
sline = np.array([ln for ln, _ in samples]); allidx = np.arange(len(samples))
G = np.array([L[ln]["V"][i, BIO] for ln, i in samples])
Vall = np.stack([L[ln]["V"][i] for ln, i in samples]); VW = np.stack([L[ln]["vwt"] for ln, _ in samples])
anyact = (np.abs(Vall).max(0) > 0) | (np.abs(VW).max(0) > 0); cols = np.where(KEEP & anyact)[0]
XD = slog(Vall[:, cols] - VW[:, cols]).astype(np.float32); XK = slog(Vall[:, cols]).astype(np.float32)
del Vall, VW
say(f"colorectal samples {len(samples)}; features {len(cols)} reactions (mask {len(MASK)} removed); {time.time()-t0:.0f}s")
def med_feat(lb):
    return np.concatenate([(lb == 0), (lb <= -999), np.where((lb < 0) & (lb > -999), slog(-lb * 1000), 0)]).astype(np.float32)
MEDRAW = np.stack([med_feat(L[ln]["medium_lb"]) for ln in lines]).astype(np.float64); lpos = {ln: k for k, ln in enumerate(lines)}

# ---------------- external samples on the same feature columns ----------------
E = args.external_dataset
emeta = json.load(open(f"{E}/scaffold_meta.json")); assert emeta["reactions"] == meta["reactions"], "external scaffold index differs"
elines, EL, esamples = load(E, emeta, KEEP)
eV = np.stack([EL[ln]["V"][i] for ln, i in esamples]); eVW = np.stack([EL[ln]["vwt"] for ln, _ in esamples])
eXD = slog(eV[:, cols] - eVW[:, cols]).astype(np.float32); eXK = slog(eV[:, cols]).astype(np.float32)
eG = eV[:, BIO]; del eV, eVW
edep = np.array([float(EL[ln]["dependency"][i]) for ln, i in esamples], np.float32); ehas = np.isfinite(edep)
ext = pd.DataFrame(dict(line=[ln for ln, _ in esamples], symbol=[str(EL[ln]["symbols"][i]) for ln, i in esamples],
                        gene=[str(EL[ln]["genes"][i]) for ln, i in esamples], dependency=np.where(ehas, edep, np.nan),
                        label=np.where(ehas, (edep >= 0.5).astype(int), -1), has_label=ehas, growth_ratio=eG))
EMED = np.stack([med_feat(EL[ln]["medium_lb"]) for ln in elines]).astype(np.float64); elpos = {ln: k for k, ln in enumerate(elines)}
say(f"external samples {len(esamples)} over {len(elines)} cell lines ({int((~ehas).sum())} without label); {time.time()-t0:.0f}s")


def split_fold(fold):
    perm = list(np.random.default_rng(0).permutation(lines)); lf = {ln: k % args.nfold for k, ln in enumerate(perm)}
    test_lines = sorted(ln for ln in lines if lf[ln] == fold); rest_lines = sorted(ln for ln in lines if lf[ln] != fold)
    val_lines = sorted(np.random.default_rng(fold).choice(rest_lines, int(round(0.2 * len(rest_lines))), replace=False))
    test = allidx[np.isin(sline, test_lines)]; val = allidx[np.isin(sline, val_lines)]
    train = allidx[~np.isin(sline, test_lines + list(val_lines))]; return train, val, test, test_lines, val_lines


os.makedirs(f"{args.out}/line", exist_ok=True)
for fold in args.folds:
    assert not os.path.exists(f"{args.out}/line/fold{fold}.tsv"), f"fold{fold}.tsv exists (no overwrite)"
    tf = time.time(); train, val, test, test_lines, val_lines = split_fold(fold)
    tr_rows = np.isin(lines, sorted(set(sline[train]))); sd = MEDRAW[tr_rows].std(0); keep = sd > 1e-6; mu = MEDRAW[tr_rows][:, keep].mean(0)
    MEDZ = (MEDRAW[:, keep] - mu) / sd[keep]; MEDS = MEDZ[[lpos[ln] for ln in sline]].astype(np.float32)
    eMEDS = ((EMED[:, keep] - mu) / sd[keep])[[elpos[ln] for ln in ext.line]].astype(np.float32)
    X = np.concatenate([XD, MEDS], 1); eX = np.concatenate([eXD, eMEDS], 1)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=20, random_state=fold).fit(X[train], lab[train])
    ph_test = hgb.predict_proba(X[test])[:, 1]; ph_ext = hgb.predict_proba(eX)[:, 1]; t_hgb = time.time() - tf
    hgb0 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=20, random_state=fold).fit(XD[train], lab[train])
    ph0_test = hgb0.predict_proba(XD[test])[:, 1]; ph0_ext = hgb0.predict_proba(eXD)[:, 1]
    XP = np.concatenate([XK, XD], 1); eXP = np.concatenate([eXK, eXD], 1)
    rg = Ridge(alpha=10.0).fit(XP[train], G[train]); pg_test = rg.predict(XP[test]); pg_ext = rg.predict(eXP)
    r2 = 1 - ((pg_test - G[test])**2).sum() / ((G[test] - G[test].mean())**2).sum()
    out = ext.assign(hgb_prob=ph_ext, hgb_nomed_prob=ph0_ext, ridge_growth=pg_ext)
    assert np.isfinite(out[["hgb_prob", "hgb_nomed_prob", "ridge_growth"]].to_numpy()).all()
    metaj = dict(script="script_python/86b_flat_baselines_external.py", date=time.ctime(), split="line", fold=fold, n_train=len(train), n_val=len(val),
                 n_test=len(test), test_lines=test_lines, val_lines=list(val_lines), n_features_hgb=int(X.shape[1]), n_features_hgb_nomed=int(XD.shape[1]),
                 n_medium_cols=int(keep.sum()), hgb_iters=int(hgb.n_iter_), hgb_nomed_iters=int(hgb0.n_iter_),
                 colorectal_check=dict(hgb_auroc_test=float(roc_auc_score(lab[test], ph_test)), hgb_nomed_auroc_test=float(roc_auc_score(lab[test], ph0_test)),
                                       probe_r2_test=float(r2), probe_spearman_test=float(spearmanr(pg_test, G[test])[0]),
                                       reference="result/gnn_20261028/flat_v2/line/fold%d_meta.json (hgb_auroc_test, probe_r2_test)" % fold),
                 external=dict(dataset=E, samples_md5=md5(f"{E}/samples.tsv"), n=int(len(out)), n_without_label=int((~out.has_label).sum()),
                               note="no metric on external samples computed here (pre-specified plan: one evaluation run, script 85b)"),
                 mask_file=args.mask_file, seconds_hgb=round(t_hgb), seconds_total=round(time.time() - tf))
    with open(f"{args.out}/line/fold{fold}.tsv", "w") as fh:
        fh.write(f"# 86b gradient-boosting baseline (hgb_prob: delta fluxome + medium; hgb_nomed_prob: delta fluxome only) and growth probe (ridge_growth), "
                 f"refit on colorectal cell-line-disjoint fold {fold} training part ({len(train)} samples, {int(tr_rows.sum())} cell lines), applied to external dataset {E}; "
                 f"features {len(cols)} reactions (+{int(keep.sum())} medium cols); mask {args.mask_file}; {time.ctime()}; n {len(out)}; label -1 = no DepMap row\n")
        out.to_csv(fh, sep="\t", index=False)
    json.dump(metaj, open(f"{args.out}/line/fold{fold}_meta.json", "w"), indent=1)
    say(f"fold {fold}: colorectal check HGB test AUROC {metaj['colorectal_check']['hgb_auroc_test']:.4f} (no-medium {metaj['colorectal_check']['hgb_nomed_auroc_test']:.4f}), "
        f"probe R2 {r2:.4f}; external rows {len(out)} written; {time.time()-tf:.0f}s")
say(f"done {time.time()-t0:.0f}s -> {args.out}")
