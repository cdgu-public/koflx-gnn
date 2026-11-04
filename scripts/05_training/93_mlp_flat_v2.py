"""Arms 1 / 1b: MLP on exactly the flat HGB input of script 86 (slog(v_KO - v_WT) of the 7,147 unmasked active-anywhere
reactions + the medium columns that vary across training lines, z-scored on them), same samples and splits as 81/86.
User-approved 2026-10-29 (status.md 13:40 entry; review_gnn_gap.md section 3). 81/81b/85/86 untouched; data code copied from 86.

--recipe gnn81 (Arm 1): the training recipe of script 81. Soft target (clipped dependency) with BCEWithLogits, balanced
    WeightedRandomSampler on the binary label with replacement (n_train draws per epoch), AdamW lr 1e-3 wd 1e-4, batch 32,
    early stopping on validation BCE, min 10 epochs, patience 8, max --max-epochs. Folder token mlp81.
--recipe hgblike (Arm 1b): hard label (dependency >= 0.5), plain shuffling (no balancing), stopping on validation AUROC,
    everything else identical. Folder token mlphgb.
Architecture (both): Linear(F, 512) LeakyReLU(0.1) Dropout 0.2 Linear(512, 128) LeakyReLU(0.1) Dropout 0.2 Linear(128, 1).
Features z-scored per column on the training rows (sd floor 1e-6; label-free). The HGB of 86 is scale invariant, an MLP is not.
Per-epoch log.tsv: train_loss, val_bce, val_auroc, val_auprc, train_auroc_probe (fixed 2,000 training samples, eval mode).
Output: <out-root>/A_<split>_<token>_s<seed>/fold<k>/{pred_val.tsv, pred_test.tsv, log.tsv, model.pt, config.json, split.json}.
  The leading "A" exists only because 85_eval_gnn_v2.py accepts folder names starting with A or B (85:63-64); it does not mean
  graph A'. The variant token (mlp81 / mlphgb) identifies the arm. rel_nn_dist and dup_in_train are copied from
  runs_v2/A_<split>_s0/fold<k>/pred_test.tsv (sample and split properties, identical across models; asserted on the row set).
  Evaluate with: python 85_eval_gnn_v2.py <out-root> --out <out-root>/eval_mlp
--no-medium (2026-10-30): the medium columns are dropped, X = the reaction columns only, so that the MLP is a like-for-like comparator
  of the no-medium GNN (user ruling 2026-10-30 08:40 KST: medium input excluded from the final model; with medium columns the MLP
  could infer cell-line identity on the gene-disjoint split, where every line is in training). Folder token _nomed after the split.
--protocol fixed_cosine (2026-10-30, same semantics as 81b_gnn_train_v2.py; lock file result/gnn_20261028/protocol_lock_v3_20261030.json,
  control arms control_mlp_gnn81 / control_mlp_hgblike): fixed budget of --max-epochs epochs, no early stopping, no validation-based
  checkpoint selection (the recipe's stopping metric is not used), AdamW lr cosine-annealed per epoch from --lr to --lr-min
  (Loshchilov & Hutter 2017, arXiv:1608.03983), model.pt = last epoch, per-epoch ckpt/ep<k>.pt for 100_monitor_v3.py (monitoring
  only), log.tsv gains an lr column, folder token _fixedcos<max_epochs> before _s<seed>, an existing fold folder is refused,
  config.json records protocol, lr_min, lr_schedule, checkpoint, script and is checked against --protocol-lock at the end ('none'
  skips, smoke only). Target and sampler of the recipe are unchanged (gnn81: soft target + balanced sampler; hgblike: hard label +
  plain shuffling). The default --protocol es_val is the unchanged path (stopping on validation BCE or AUROC per recipe).
Usage: CUDA_VISIBLE_DEVICES=0 python 93_mlp_flat_v2.py --split gene --fold 0 --recipe gnn81 --seed 0 [--max-epochs 30]
       CUDA_VISIBLE_DEVICES=0 python 93_mlp_flat_v2.py --split gene --fold 0 --recipe gnn81 --seed 0 --no-medium --protocol fixed_cosine --max-epochs 30
Smoke: add --smoke (600 training rows, 2 epochs) and --out-root <scratch>.
Env: crcgem_gnn. GPU memory: X is 18,573 x 7,200 float32 = 0.53 GB on the device; < 1.5 GB total.
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["gene", "line"], default="gene")
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--recipe", choices=["gnn81", "hgblike"], required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--nfold", type=int, default=5)
ap.add_argument("--max-epochs", type=int, default=30)
ap.add_argument("--min-epochs", type=int, default=10)
ap.add_argument("--patience", type=int, default=8)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--hidden", nargs=2, type=int, default=[512, 128])
ap.add_argument("--train-probe", type=int, default=2000)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--mask-file", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/growth_mask_v2_20261029.json")
ap.add_argument("--ref-root", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/runs_v2", help="source of rel_nn_dist/dup_in_train")
ap.add_argument("--out-root", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/runs_v2_mlp")
ap.add_argument("--no-medium", action="store_true", help="drop the medium columns (like-for-like comparator of the no-medium GNN)")
ap.add_argument("--protocol", choices=["es_val", "fixed_cosine"], default="es_val",
                help="es_val = recipe stopping rule (validation BCE for gnn81, validation AUROC for hgblike, best checkpoint); fixed_cosine = fixed budget of "
                     "--max-epochs epochs, cosine annealing --lr -> --lr-min per epoch, last-epoch checkpoint, per-epoch ckpt/ (2026-10-30, as 81b)")
ap.add_argument("--lr-min", type=float, default=1e-5, help="fixed_cosine only: final learning rate of the cosine schedule")
ap.add_argument("--protocol-lock", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/protocol_lock_v3_20261030.json",
                help="fixed_cosine only: lock file checked against config.json at the end of the run; 'none' skips (smoke tests)")
ap.add_argument("--dataset", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/dataset",
                help="dataset folder (script 64 / 64d); default = the 2026-10-28 dataset, unchanged behavior. 2026-10-30 tolpatch: result/gnn_20261030_tolpatch/dataset; recorded in config.json and checked by the lock")
ap.add_argument("--ref-run", default="", help="run folder under --ref-root whose fold<k>/pred_test.tsv supplies rel_nn_dist / dup_in_train; default A_<split>_s0 (unchanged behavior); tolpatch grid: A_gene_nomed_fixedcos30_s0_tolpatch")
args = ap.parse_args()
FIXED = args.protocol == "fixed_cosine"
ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"; D = args.dataset.rstrip("/")
TOKEN = {"gnn81": "mlp81", "hgblike": "mlphgb"}[args.recipe]
OUT = f"{args.out_root}/A_{args.split}{'_nomed' if args.no_medium else ''}_{TOKEN}{f'_fixedcos{args.max_epochs}' if FIXED else ''}_s{args.seed}{args.tag}/fold{args.fold}"
if FIXED and os.path.exists(OUT):
    raise SystemExit(f"REFUSED: fold folder exists, fixed_cosine runs never overwrite: {OUT}")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(1000 * args.seed + args.fold); rng = np.random.default_rng(1000 * args.seed + args.fold)
dev = torch.device(args.device); t0 = time.time()
def say(*a): print(*a, flush=True)
def slog(x): return np.sign(x) * np.log1p(np.abs(x))

# ---------------- data, identical to 73/81/86 ----------------
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
dep = np.array([float(L[ln]["dependency"][i]) for ln, i in samples], np.float32); lab = (dep >= 0.5).astype(np.float32)
tgt = np.clip(dep, 0, 1).astype(np.float32)
sline = np.array([ln for ln, _ in samples]); allidx = np.arange(len(samples))
Vall = np.stack([L[ln]["V"][i] for ln, i in samples]); VW = np.stack([L[ln]["vwt"] for ln, _ in samples])
anyact = (np.abs(Vall).max(0) > 0) | (np.abs(VW).max(0) > 0); cols = np.where(KEEP & anyact)[0]
XD = slog(Vall[:, cols] - VW[:, cols]).astype(np.float32)
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


train, val, test = split_fold(args.split, args.fold)
if args.smoke:
    train = np.sort(rng.choice(train, 600, replace=False)); args.max_epochs = 2; args.min_epochs = 1
tr_rows = np.isin(lines, sorted(set(sline[train]))); sd = MEDRAW[tr_rows].std(0); keep = sd > 1e-6
MEDZ = (MEDRAW[:, keep] - MEDRAW[tr_rows][:, keep].mean(0)) / sd[keep]
if args.no_medium:
    keep = np.zeros_like(keep); MEDZ = MEDZ[:, :0]   # no medium columns: X = reaction columns only (n_medium_cols 0)
X = np.concatenate([XD, MEDZ[[lpos[ln] for ln in sline]].astype(np.float32)], 1)
mu, s = X[train].mean(0), X[train].std(0); s = np.where(s > 1e-6, s, 1.0)
X = ((X - mu) / s).astype(np.float32)
say(f"X {X.shape}; train {len(train)} val {len(val)} test {len(test)}; pos rate train {lab[train].mean():.3f} val {lab[val].mean():.3f} test {lab[test].mean():.3f}")
json.dump(dict(split=args.split, fold=args.fold, n_train=len(train), n_val=len(val), n_test=len(test),
               pos_rate=dict(train=float(lab[train].mean()), val=float(lab[val].mean()), test=float(lab[test].mean()))), open(f"{OUT}/split.json", "w"), indent=1)
Xt = torch.tensor(X, device=dev); Yt = torch.tensor(tgt if args.recipe == "gnn81" else lab, device=dev)
probe_ids = np.sort(rng.choice(train, min(args.train_probe, len(train)), replace=False))

net = nn.Sequential(nn.Linear(X.shape[1], args.hidden[0]), nn.LeakyReLU(0.1), nn.Dropout(0.2),
                    nn.Linear(args.hidden[0], args.hidden[1]), nn.LeakyReLU(0.1), nn.Dropout(0.2), nn.Linear(args.hidden[1], 1)).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4); lossf = nn.BCEWithLogitsLoss()
cfg = dict(vars(args), n_train=len(train), n_val=len(val), n_test=len(test), n_features=int(X.shape[1]), n_medium_cols=int(keep.sum()),
           n_params=sum(p.numel() for p in net.parameters()), token=TOKEN, note="A_ prefix only for 85 folder parsing, not graph A'")
if FIXED:
    cfg.update(lr_schedule="cosine_annealing_per_epoch_no_restarts", checkpoint="last_epoch", early_stopping=False,
               weight_decay=1e-4, optimizer="AdamW", ckpt_dir="ckpt", protocol_date="2026-10-30", script="93_mlp_flat_v2.py")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs, eta_min=args.lr_min)
    os.makedirs(f"{OUT}/ckpt", exist_ok=True)
else:
    for k in ("protocol", "lr_min", "protocol_lock"):   # default path: config.json keys unchanged (no_medium is recorded)
        cfg.pop(k)
json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1)


def predict(ids):
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(Xt[torch.tensor(ids, device=dev)]).squeeze(1)).cpu().numpy()


def bce(ids, p):
    p = np.clip(p, 1e-7, 1 - 1e-7); t = tgt[ids]
    return float(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean())


def epoch_batches():
    if args.recipe == "gnn81":   # balanced sampling with replacement, n_train draws, as 81:262-264
        y = lab[train]; w = np.where(y == 1, 0.5 / y.sum(), 0.5 / (len(y) - y.sum()))
        g = torch.Generator().manual_seed(int(rng.integers(1 << 31)))
        order = train[torch.multinomial(torch.tensor(w, dtype=torch.double), len(train), True, generator=g).numpy()]
    else:
        order = rng.permutation(train)
    for k in range(0, len(order), args.batch):
        yield order[k:k + args.batch]


log, best, bad = [], np.inf, 0
for ep in range(args.max_epochs):
    net.train(); te = time.time(); tl, nb = 0.0, 0
    for ids in epoch_batches():
        it = torch.tensor(ids, device=dev); loss = lossf(net(Xt[it]).squeeze(1), Yt[it])
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); nb += 1
    vp = predict(val); vl = bce(val, vp); pp = predict(probe_ids)
    log.append(dict(epoch=ep, train_loss=tl / nb, val_bce=vl, val_auroc=roc_auc_score(lab[val], vp), val_auprc=average_precision_score(lab[val], vp),
                    train_auroc_probe=roc_auc_score(lab[probe_ids], pp), epoch_s=round(time.time() - te, 1)))
    if FIXED:
        log[-1]["lr"] = opt.param_groups[0]["lr"]   # learning rate used in this epoch
    say(log[-1]); pd.DataFrame(log).to_csv(f"{OUT}/log.tsv", sep="\t", index=False)
    if FIXED:
        # fixed budget (as 81b): every epoch is checkpointed for post-hoc monitoring; model.pt is always the latest (= last) epoch;
        # validation metrics drive nothing (no selection, no patience)
        torch.save(net.state_dict(), f"{OUT}/ckpt/ep{ep:02d}.pt"); torch.save(net.state_dict(), f"{OUT}/model.pt")
        sched.step()
        continue
    crit = vl if args.recipe == "gnn81" else -log[-1]["val_auroc"]
    if crit < best:
        best, bad = crit, 0; torch.save(net.state_dict(), f"{OUT}/model.pt")
    else:
        bad += 1
        if bad >= args.patience and ep + 1 >= args.min_epochs:
            break

net.load_state_dict(torch.load(f"{OUT}/model.pt", weights_only=True))
ref = pd.read_csv(f"{args.ref_root}/{args.ref_run or f'A_{args.split}_s0'}/fold{args.fold}/pred_test.tsv", sep="\t")[["line", "gene", "dup_in_train", "rel_nn_dist"]]
for name, ids in (("val", val), ("test", test)):
    p = predict(ids)
    o = pd.DataFrame(dict(line=sline[ids], symbol=sym[ids], gene=gid[ids], dependency=dep[ids], label=lab[ids], gnn_prob=p))
    if name == "test":
        o = o.merge(ref, on=["line", "gene"], how="left")
        assert len(o) == len(ids) and o.dup_in_train.notna().all(), "reference pred_test.tsv does not cover the test rows"
        say(f"test: n {len(ids)} AUROC {roc_auc_score(lab[ids], p):.3f} AUPRC {average_precision_score(lab[ids], p):.3f}")
    o.to_csv(f"{OUT}/pred_{name}.tsv", sep="\t", index=False)
if FIXED and args.protocol_lock != "none":
    # protocol lock (review_stopping.md section 3, item 3): this fold's config.json must match one registered control arm
    import importlib.util
    _sp = importlib.util.spec_from_file_location("chk101", f"{ROOT}/script_python/101_check_protocol_v3.py")
    _chk = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_chk)
    _ok, _arm, _mism = _chk.check_config(json.load(open(f"{OUT}/config.json")), json.load(open(args.protocol_lock)))
    say(f"protocol lock check ({os.path.basename(args.protocol_lock)}): {'PASS' if _ok else 'MISMATCH'} arm={_arm}" + ("" if _ok else " " + "; ".join(_mism)))
    if not _ok:
        say("PROTOCOL LOCK MISMATCH: outputs kept, exit 3"); raise SystemExit(3)
elif FIXED:
    say("protocol lock check skipped (--protocol-lock none)")
say(f"done {time.time() - t0:.0f}s")
