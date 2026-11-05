"""GNN v2 on the MOMA knockout fluxomes (dataset of script 64), graphs A′ and B. Script 73 (v1) is frozen.

81b (2026-10-29) = script 81 plus the PC3 fallback readout of scheme_v2.md section 2.3 / 2.7, selected by
--readout {attn_max,mean_max}. Default attn_max reproduces script 81 exactly (same code path, same parameter
creation order, same output folder). mean_max replaces the gated attention pooling by global mean pooling (no gate
network, no AttentionalAggregation parameters); global max pooling and the medium branch are unchanged, the head
input stays 3*hidden. With mean_max the attention diagnostics (att_entropy, att_max) are NaN and the output folder
carries _meanmax. Pass criteria for the mean_max pre-check (no attention): logit_std > 0.05, head_live >= 0.5,
val AUROC > 0.6 at epoch 1. Script 81 itself is not modified.
Fallback pre-check: CUDA_VISIBLE_DEVICES=<g> python 81b_gnn_train_v2.py --graph B --split gene --fold 3 --readout mean_max
  --max-epochs 2 --min-epochs 1 --max-train-batches 150 --tag _precheck

2026-10-30, --protocol fixed_cosine (user decision 2026-10-30 07:44 KST after result/gnn_20261028/discussion_20261030/
review_stopping.md section 2; lock file result/gnn_20261028/protocol_lock_v3_20261030.json):
- fixed training budget of --max-epochs epochs (30 for the grid, 45 for the falsifier dry-test), no early stopping,
  no validation-based checkpoint selection; AdamW learning rate annealed per epoch by cosine annealing without restarts
  (Loshchilov & Hutter 2017, arXiv:1608.03983) from --lr to --lr-min; model.pt = the last epoch; validation metrics are
  logged every epoch (log.tsv gains an lr column) and drive nothing.
- per-epoch checkpoints ckpt/ep<k>.pt for the post-hoc monitoring script 100_monitor_v3.py (monitor_val.tsv /
  monitor_test.tsv; monitoring only, never read by 85_eval_gnn_v2.py; checker 101_check_protocol_v3.py).
- output folder carries the token _fixedcos<max_epochs> before _s<seed>; an existing fold folder is refused (no overwrite).
- config.json records protocol, lr_min, lr_schedule, checkpoint and is checked against the lock file at the end of the run
  (--protocol-lock none skips the check, smoke tests only).
The default --protocol es_valbce is the unchanged script-81 path (early stopping on validation BCE; same code, same
RNG consumption, same output names, same config.json keys).

PROPOSED 2026-10-29 (scheme review, result/gnn_20261028/scheme_review_20261029/scheme_v2.md), on top of script 80:
- Growth mask (user rule: the growth rate is not an input). The reactions whose KO flux is a fixed multiple of the
  biomass flux are removed from both graphs: the biomass reaction MAR13082 (its value in the dataset IS the MOMA
  growth ratio, because script 64 divides every flux by the WT biomass flux), biomass transport/exchange MAR10023/
  MAR10024, the four biomass pool reactions MAR10062-MAR10065, and the sole-route cofactor supplies that the check
  found coupled in >= 99% of samples (MAR08615-17, MAR09167, MAR09608, MAR04603). Set = --mask-file (13 reactions,
  check_leak_dup.py). Removed as nodes (B: reaction node and its edges; A′: never enters the active set). The sample
  set (knockouts that change the fluxome) is unchanged.
- --seed: initialization and sampler seed (splits do not change with the seed). Output folder carries _s<seed>.
- --no-medium: ablation, medium branch receives a constant.
- pred_*.tsv gets rel_nn_dist: Chebyshev distance of the sample's fluxome to the nearest training fluxome of the
  same line, divided by the sample's own |v_KO - v_WT|_inf (0 = exact duplicate; evaluation strata).

Changes from v1 (script 73; fix of the graph-B constant-output collapse, diagnosis 2026-10-28, script 79 and an independent review):
- Head and medium branch: ReLU -> LeakyReLU(0.1). With v1 the head input is nearly constant across samples
  (graph B: between-sample std 0.1% of scale at init), the per-unit gradient is batch noise, AdamW random-walks
  each ReLU unit into the absorbing dead state (pre-activation <= 0 for every sample => zero gradient forever);
  once all 64 head units are dead the GNN receives no gradient (B gene folds 1 and 3, B line fold 1). The medium
  branch died the same way in every v1 model, A′ included (0/64 live units): the medium was never used.
- Medium input: only the columns that vary across the training lines (53 of 4,980; the rest are constant), z-scored
  over the training lines (label-free input statistics).
- Readout: attention pooling concatenated with global max pooling (both graphs), so that the head sees a
  sample-dependent input from the first step (graph B at init: between-sample std 0.04 for max pooling vs 4e-4
  for attention pooling).
- Per-epoch diagnostics on a fixed validation batch (logit std, live head units, attention entropy / max weight).
- Output: result/gnn_20261028/runs_v2/.

Changes from v0 (script 65, kept as run):
- Splits (all 59 lines; no separate held-out lines, user 2026-10-28):
  --split gene (primary, vs MOMA): groups = the set of scaffold reactions the knockout disables (546 groups on the 2026-10-28 dataset, 549 on dataset_v2;
  no union-find merge, which had chained 119 genes / 36% of samples into one group); the 5 largest groups
  one per fold, the rest size-balanced (group_folds; positive rate 0.31-0.53 per fold); validation = random
  groups of the outer-training part (seed = fold; the 4 largest kept in training) up to 20% of samples. --split line (secondary): lines in random order (seed 0) cut into 5 folds; validation = 20% of
  the outer-training lines (seed = fold). Test samples whose fluxome equals a training fluxome are flagged
  (dup_in_train).
- Input: fluxome and medium only. The context-model presence flag (from the transcriptome via tINIT) is left
  out; --presence adds it back (ablation).
- Pooling: attention pooling (AttentionalAggregation) + medium MLP -> head.
- Target: DepMap dependency as a soft label, BCE; balanced sampling of the binary label (>= 0.5) per epoch,
  no pos_weight.
- Early stopping on validation BCE (soft target): min 10 epochs, patience 8, max 50.
Graphs and node/edge features otherwise as v0: B = full scaffold bipartite graph, edge S_ij*v_j (WT, KO);
A′ = Mass Flow Graph of the reactions active in WT or KO [ref: doi: 10.1038/s41540-018-0067-y], currency
metabolites (Stelzer 2011 Table A1) as node features. Flux features v_WT, v_KO, v_KO - v_WT (/ WT biomass
flux, signed log1p); fluxes below the QP tolerance (1e-6 raw) set to 0. Samples: knockouts that change the
line's fluxome. Evaluation label: dependency >= 0.5.
Outputs per fold: pred_val.tsv, pred_test.tsv, log.tsv, model.pt, config.json, split.json.
Usage: CUDA_VISIBLE_DEVICES=0 python 80_gnn_train_v2.py --graph B --split gene --fold 0 [--presence]
Pre-check (collapsed fold): python 80_gnn_train_v2.py --graph B --split gene --fold 3 --max-epochs 2 --max-train-batches 150 --tag _precheck
Env: crcgem_gnn. Output: result/gnn_20261028/runs_v2/<graph>_<split>[_presence]/fold<k>/
"""
import argparse, json, os, time
import numpy as np, pandas as pd, scipy.sparse as sp, torch
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import average_precision_score, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--graph", choices=["A", "B"], required=True)
ap.add_argument("--split", choices=["gene", "line"], required=True)
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--nfold", type=int, default=5)
ap.add_argument("--presence", action="store_true", help="ablation: add the context-model presence flag")
ap.add_argument("--max-epochs", type=int, default=50)
ap.add_argument("--min-epochs", type=int, default=10)
ap.add_argument("--patience", type=int, default=8)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--max-train-batches", type=int, default=0, help="debug: cap batches per epoch")
ap.add_argument("--tag", default="")
ap.add_argument("--seed", type=int, default=0, help="initialization / sampler seed; splits are seed-independent")
ap.add_argument("--no-medium", action="store_true", help="ablation: medium branch receives a constant")
ap.add_argument("--mask-file", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/growth_mask_v2_20261029.json",
                help="json with mask_index: reactions removed from both graphs (biomass and reactions coupled to it)")
ap.add_argument("--out-root", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/runs_v2")
ap.add_argument("--readout", choices=["attn_max", "mean_max"], default="attn_max",
                help="attn_max = script 81 (gated attention + global max pooling); mean_max = PC3 fallback (global mean + global max pooling, no gate network)")
ap.add_argument("--protocol", choices=["es_valbce", "fixed_cosine"], default="es_valbce",
                help="es_valbce = script 81 (early stopping on validation BCE, best-validation checkpoint); fixed_cosine = fixed budget of "
                     "--max-epochs epochs, cosine annealing --lr -> --lr-min per epoch, last-epoch checkpoint, per-epoch ckpt/ (2026-10-30)")
ap.add_argument("--lr-min", type=float, default=1e-5, help="fixed_cosine only: final learning rate of the cosine schedule")
ap.add_argument("--protocol-lock", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/protocol_lock_v3_20261030.json",
                help="fixed_cosine only: lock file checked against config.json at the end of the run; 'none' skips (smoke tests)")
ap.add_argument("--dataset", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/dataset",
                help="dataset folder (script 64 / 64d); default = the 2026-10-28 dataset, unchanged behavior. 2026-10-30 tolpatch: result/gnn_20261030_tolpatch/dataset; recorded in config.json and checked by the lock")
args = ap.parse_args()
FIXED = args.protocol == "fixed_cosine"
ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
D = args.dataset.rstrip("/")
OUT = f"{args.out_root}/{args.graph}_{args.split}{'_presence' if args.presence else ''}{'_nomed' if args.no_medium else ''}{'_meanmax' if args.readout == 'mean_max' else ''}{f'_fixedcos{args.max_epochs}' if FIXED else ''}_s{args.seed}{args.tag}/fold{args.fold}"
if FIXED and os.path.exists(OUT):
    raise SystemExit(f"REFUSED: fold folder exists, fixed_cosine runs never overwrite: {OUT}")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(1000 * args.seed + args.fold); rng = np.random.default_rng(1000 * args.seed + args.fold)
dev = torch.device("cuda:0")
t0 = time.time()

def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))

# ---------------- data ----------------
meta = json.load(open(f"{D}/scaffold_meta.json"))
S = sp.load_npz(f"{D}/scaffold_S.npz").tocsc()
NR, NM = S.shape[1], S.shape[0]
mpos = {m: i for i, m in enumerate(meta["metabolites"])}
cur_idx = np.array([mpos[m] for m in meta["currency"] if m in mpos])
is_cur = np.zeros(NM, bool); is_cur[cur_idx] = True
gene_rxns = meta["gene_rxns"]
# growth mask: reactions whose flux is a fixed multiple of the biomass flux never enter a graph (user rule)
MASKJ = json.load(open(args.mask_file)); MASK = np.array(sorted(set(MASKJ["mask_index"])), int)
BIO = meta["reactions"].index(meta["biomass"])
assert BIO in set(MASK.tolist()), "mask must contain the biomass reaction"
assert all(meta["reactions"][i] == r for i, r in zip(MASKJ["mask_index"], MASKJ["mask_reactions"])), "mask index/id mismatch"
KEEP = np.ones(NR, bool); KEEP[MASK] = False
print(f"growth mask: {len(MASK)} reactions removed from the graphs: {MASKJ['mask_reactions']}", flush=True)
lines = sorted(f[:-4] for f in os.listdir(f"{D}/lines"))
L = {ln: np.load(f"{D}/lines/{ln}.npz") for ln in lines}
L = {ln: {k: z[k] for k in z.files} for ln, z in L.items()}
for ln in lines:
    th = 1e-6 / float(L[ln]["mu_ref"])
    L[ln]["th"] = th
    L[ln]["vwt"] = np.where(np.abs(L[ln]["vwt"]) > th, L[ln]["vwt"], 0).astype(np.float32)
    L[ln]["V"] = np.where(np.abs(L[ln]["V"]) > th, L[ln]["V"], 0).astype(np.float32)
samples = [(ln, i) for ln in lines for i in range(len(L[ln]["genes"]))
           if np.abs(L[ln]["V"][i] - L[ln]["vwt"]).max() > L[ln]["th"]]
sym = np.array([L[ln]["symbols"][i] for ln, i in samples])
gid = np.array([L[ln]["genes"][i] for ln, i in samples])
dep = np.array([float(L[ln]["dependency"][i]) for ln, i in samples], np.float32)
tgt = np.clip(dep, 0, 1).astype(np.float32)
lab = (dep >= 0.5).astype(np.float32)
sline = np.array([ln for ln, _ in samples])
def med_feat(lb):
    return np.concatenate([(lb == 0), (lb <= -999), np.where((lb < 0) & (lb > -999), slog(-lb * 1000), 0)]).astype(np.float32)

# ---------------- split ----------------
def group_folds(groups, k):
    """Whole groups to k folds: the k largest groups one per fold, then the rest largest first into the
    currently smallest fold (sizes balanced; positive rate not balanced). A size-and-positive-rate balancing
    (superseded 2026-10-28) put the two largest groups, complex I and RNA polymerase, alone into one test fold."""
    ug, inv = np.unique(groups, return_inverse=True); n = np.bincount(inv)
    order = sorted(range(len(ug)), key=lambda j: (-n[j], ug[j]))
    fn = np.zeros(k); fo = np.zeros(len(ug), int)
    for t, j in enumerate(order):
        f = t if t < k else int(np.argmin(fn)); fo[j] = f; fn[f] += n[j]
    return fo[inv]


def val_groups(groups, frac, seed, keep_largest=4):
    """Validation = random groups (seeded), excluding the keep_largest largest, until frac of the samples."""
    ug, inv = np.unique(groups, return_inverse=True); n = np.bincount(inv)
    big = set(np.argsort(-n)[:keep_largest])
    cand = [j for j in np.random.default_rng(seed).permutation(len(ug)) if j not in big]
    sel, tot = set(), 0
    for j in cand:
        if tot >= frac * len(groups):
            break
        sel.add(j); tot += n[j]
    return np.isin(inv, list(sel))


allidx = np.arange(len(samples))
if args.split == "gene":
    gkey = np.array([",".join(map(str, sorted(gene_rxns.get(g, [])))) for g in gid])
    fo = group_folds(gkey, args.nfold)
    test, rest = allidx[fo == args.fold], allidx[fo != args.fold]
    v = val_groups(gkey[rest], 0.2, args.fold)
    val, train = rest[v], rest[~v]
    assert not set(gkey[train]) & set(gkey[test]) and not set(gkey[val]) & set(gkey[test]) and not set(gkey[train]) & set(gkey[val])
else:
    perm = list(np.random.default_rng(0).permutation(lines)); lf = {ln: k % args.nfold for k, ln in enumerate(perm)}
    test_lines = sorted(ln for ln in lines if lf[ln] == args.fold)
    rest_lines = sorted(ln for ln in lines if lf[ln] != args.fold)
    val_lines = sorted(np.random.default_rng(args.fold).choice(rest_lines, int(round(0.2 * len(rest_lines))), replace=False))
    test = allidx[np.isin(sline, test_lines)]
    val = allidx[np.isin(sline, val_lines)]
    train = allidx[~np.isin(sline, test_lines + list(val_lines))]
# medium features: columns varying across the training lines, z-scored over the training lines (input-only statistics)
MEDM = np.stack([med_feat(L[ln]["medium_lb"]) for ln in lines]).astype(np.float64)
tr_rows = np.isin(lines, sorted(set(sline[train])))
med_sd = MEDM[tr_rows].std(0); med_keep = med_sd > 1e-6
MEDM = (MEDM[:, med_keep] - MEDM[tr_rows][:, med_keep].mean(0)) / med_sd[med_keep]
if args.no_medium:
    MEDM = np.zeros((len(lines), 1))  # ablation: the medium branch sees a constant
MED = {ln: MEDM[k].astype(np.float32) for k, ln in enumerate(lines)}
print(f"medium: {int(med_keep.sum())} varying columns of {len(med_keep)}; {len(np.unique(MEDM[tr_rows], axis=0))} distinct media in training lines", flush=True)
train_fx = {L[samples[i][0]]["V"][samples[i][1]].tobytes() for i in train}
def dup(ids):
    return np.array([L[samples[i][0]]["V"][samples[i][1]].tobytes() in train_fx for i in ids])
print(f"samples {len(samples)}: train {len(train)} val {len(val)} test {len(test)}; pos rate train "
      f"{lab[train].mean():.3f} val {lab[val].mean():.3f} test {lab[test].mean():.3f}; test dup_in_train "
      f"{dup(test).mean():.3f}", flush=True)
json.dump(dict(split=args.split, fold=args.fold, n_train=len(train), n_val=len(val), n_test=len(test),
               test_lines=sorted(set(sline[test])), val_lines=sorted(set(sline[val])),
               pos_rate=dict(train=float(lab[train].mean()), val=float(lab[val].mean()), test=float(lab[test].mean()))),
          open(f"{OUT}/split.json", "w"), indent=1)

# ---------------- graph builders ----------------
NPRES = 1 if args.presence else 0
if args.graph == "B":
    # full scaffold bipartite graph, the same for every sample (scope section 4; ruling 2026-10-27)
    R = np.where(KEEP)[0]   # growth mask: masked reaction nodes and their edges are absent
    Ssub = S[:, R].tocoo()
    M_used = np.unique(Ssub.row); mloc = {m: k for k, m in enumerate(M_used)}
    nM, nRB = len(M_used), len(R)
    em = np.array([mloc[m] for m in Ssub.row]); er = Ssub.col; ecoef = Ssub.data.astype(np.float32)
    EI = np.concatenate([np.stack([em, er + nM]), np.stack([er + nM, em])], 1)
    EDIR = np.concatenate([np.zeros(len(em)), np.ones(len(em))]).astype(np.float32)
    F_IN = 5 + NPRES   # [is metabolite, is currency, v_WT, v_KO, v_KO - v_WT] (+ presence)
    met_x = np.zeros((nM, F_IN), np.float32); met_x[:, 0] = 1; met_x[:, 1] = is_cur[M_used]
    EI_t = torch.tensor(EI, dtype=torch.long)
    print(f"graph B fixed: {nM} metabolites, {nRB} reactions, {EI.shape[1]} directed edges", flush=True)

    def build(idx):
        ln, i = samples[idx]; z = L[ln]
        vw, vk = z["vwt"][R], z["V"][i][R]
        cols = [np.zeros(nRB), np.zeros(nRB), slog(vw), slog(vk), slog(vk - vw)]
        if NPRES:
            cols.append(z["present"][R].astype(np.float32))
        rx = np.stack(cols, 1).astype(np.float32)
        ew, ek = ecoef * vw[er], ecoef * vk[er]
        ea = np.stack([np.tile(slog(ew), 2), np.tile(slog(ek), 2), EDIR], 1).astype(np.float32)
        return Data(x=torch.tensor(np.concatenate([met_x, rx])), edge_index=EI_t, edge_attr=torch.tensor(ea),
                    med=torch.tensor(MED[ln])[None], y=torch.tensor([tgt[idx]]), idx=torch.tensor([idx]))
    E_IN = 3
else:
    noncur = ~is_cur

    def mfg(Sa, v):
        P = Sa.multiply(v[None, :]).tocsr()          # mets x act
        prod = P.maximum(0); cons = (-P).maximum(0)
        tot = np.asarray(prod.sum(1)).ravel(); tot[tot == 0] = 1
        W = (sp.diags(1 / tot) @ prod).T @ cons       # act x act
        return W.tocoo()

    def build(idx):
        ln, i = samples[idx]; z = L[ln]
        vw_all, vk_all = z["vwt"], z["V"][i]
        A = np.where(((vw_all != 0) | (vk_all != 0)) & KEEP)[0]  # reactions active in WT or KO, growth mask applied
        vw, vk = vw_all[A], vk_all[A]
        Sa = S[:, A]
        Snc = sp.diags(noncur.astype(float)) @ Sa
        Ww, Wk = mfg(Snc, vw), mfg(Snc, vk)
        W = sp.coo_matrix((np.concatenate([Ww.data, np.zeros(len(Wk.data))]), (np.concatenate([Ww.row, Wk.row]), np.concatenate([Ww.col, Wk.col]))), shape=Ww.shape).tocsr()
        W2 = sp.coo_matrix((np.concatenate([np.zeros(len(Ww.data)), Wk.data]), (np.concatenate([Ww.row, Wk.row]), np.concatenate([Ww.col, Wk.col]))), shape=Ww.shape).tocsr()
        W, W2 = W.tocoo(), W2.tocoo()  # same sparsity pattern after csr summation
        Sc = Sa[cur_idx, :]
        cw = (Sc.multiply(vw[None, :])).T.toarray(); ck = (Sc.multiply(vk[None, :])).T.toarray()
        cols = [slog(vw), slog(vk), slog(vk - vw)] + ([z["present"][A].astype(np.float32)] if NPRES else [])
        x = np.concatenate([np.stack(cols, 1), slog(cw), slog(ck)], 1).astype(np.float32)
        ea = np.stack([slog(W.data), slog(W2.data)], 1).astype(np.float32)
        return Data(x=torch.tensor(x), edge_index=torch.tensor(np.stack([W.row, W.col]), dtype=torch.long),
                    edge_attr=torch.tensor(ea), med=torch.tensor(MED[ln])[None], y=torch.tensor([tgt[idx]]),
                    idx=torch.tensor([idx]))
    F_IN, E_IN = 3 + NPRES + 2 * len(cur_idx), 2


class DS(torch.utils.data.Dataset):
    def __init__(self, ids): self.ids = ids
    def __len__(self): return len(self.ids)
    def __getitem__(self, k): return build(int(self.ids[k]))


def loader(ids, balanced):
    kw = {}
    if balanced:
        # balanced sampling of the binary label (dependency >= 0.5) per epoch, with replacement
        y = lab[ids]; w = np.where(y == 1, 0.5 / y.sum(), 0.5 / (len(y) - y.sum()))
        g = torch.Generator().manual_seed(int(rng.integers(1 << 31)))
        kw["sampler"] = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(ids), True, generator=g)
    return torch.utils.data.DataLoader(DS(ids), batch_size=args.batch, num_workers=args.workers,
                                       collate_fn=Batch.from_data_list, persistent_workers=False, prefetch_factor=4, **kw)


class Net(nn.Module):
    def __init__(self, f_in, e_in, m_in, h):
        super().__init__()
        self.inp = nn.Linear(f_in, h)
        self.convs = nn.ModuleList([GATv2Conv(h, h, heads=1, edge_dim=e_in) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(3)])
        self.readout = args.readout
        if self.readout == "attn_max":
            self.pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1)),
                                               nn=nn.Linear(h, h))
        # mean_max (PC3 fallback, scheme 2.3): no gate network; the first pooled block is global mean pooling of x
        self.med = nn.Sequential(nn.Linear(m_in, h), nn.LeakyReLU(0.1))
        self.head = nn.Sequential(nn.Linear(3 * h, h), nn.LeakyReLU(0.1), nn.Dropout(0.2), nn.Linear(h, 1))

    def embed(self, d):
        x = self.inp(d.x)
        for c, n in zip(self.convs, self.norms):
            x = n(x + torch.relu(c(x, d.edge_index, d.edge_attr)))
        return x

    def pooled(self, x, d):
        first = self.pool(x, d.batch) if self.readout == "attn_max" else global_mean_pool(x, d.batch)
        return torch.cat([first, global_max_pool(x, d.batch), self.med(d.med)], 1)

    def forward(self, d):
        x = self.embed(d)
        return self.head(self.pooled(x, d)).squeeze(1)

    def diag(self, d):
        """On one batch (eval mode): logit std across graphs, fraction of head units with a positive pre-activation
        in any graph, attention entropy (mean over graphs; uniform = ln n_nodes) and max attention weight
        (NaN for the mean_max readout, which has no attention)."""
        x = self.embed(d)
        if self.readout == "attn_max":
            a = pyg_softmax(self.pool.gate_nn(x).squeeze(-1), d.batch)
            ent = torch.stack([-(a[d.batch == q] * a[d.batch == q].clamp_min(1e-30).log()).sum() for q in range(d.num_graphs)])
            att_entropy, att_max = round(ent.mean().item(), 3), round(a.max().item(), 4)
        else:
            att_entropy, att_max = float("nan"), float("nan")
        g = self.pooled(x, d)
        h1 = self.head[0](g); logit = self.head(g).squeeze(1)
        return dict(logit_std=round(logit.std().item(), 4), head_live=round((h1 > 0).any(0).float().mean().item(), 3),
                    att_entropy=att_entropy, att_max=att_max)


net = Net(F_IN, E_IN, len(MED[lines[0]]), args.hidden).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss()


def predict(ids):
    net.eval(); out_i, out_p = [], []
    with torch.no_grad():
        for b in loader(ids, False):
            b = b.to(dev); out_p.append(torch.sigmoid(net(b)).cpu().numpy()); out_i.append(b.idx.cpu().numpy())
    return np.concatenate(out_i), np.concatenate(out_p)


def bce(i, p):
    p = np.clip(p, 1e-7, 1 - 1e-7); t = tgt[i]
    return float(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean())


cfg = dict(vars(args), n_train=len(train), n_val=len(val), n_test=len(test), f_in=F_IN, e_in=E_IN, lines=lines)
if FIXED:
    cfg.update(lr_schedule="cosine_annealing_per_epoch_no_restarts", checkpoint="last_epoch", early_stopping=False,
               weight_decay=1e-4, optimizer="AdamW", ckpt_dir="ckpt", protocol_date="2026-10-30")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs, eta_min=args.lr_min)
    os.makedirs(f"{OUT}/ckpt", exist_ok=True)
else:
    for k in ("protocol", "lr_min", "protocol_lock"):   # default path: config.json keys unchanged from script 81
        cfg.pop(k)
json.dump(cfg, open(f"{OUT}/config.json", "w"), indent=1)
diag_batch = next(iter(loader(val[:args.batch], False))).to(dev)
log, best, bad = [], np.inf, 0
for ep in range(args.max_epochs):
    net.train(); te = time.time(); tl, nb = 0.0, 0
    for b in loader(train, True):
        b = b.to(dev); loss = lossf(net(b), b.y)
        opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); nb += 1
        if args.max_train_batches and nb >= args.max_train_batches:
            break
    vi, vp = predict(val)
    vl = bce(vi, vp)
    with torch.no_grad():
        dg = net.diag(diag_batch)
    log.append(dict(epoch=ep, train_loss=tl / nb, val_bce=vl, val_auroc=roc_auc_score(lab[vi], vp),
                    val_auprc=average_precision_score(lab[vi], vp), **dg,
                    epoch_s=round(time.time() - te), peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)))
    if FIXED:
        log[-1]["lr"] = opt.param_groups[0]["lr"]   # learning rate used in this epoch
    print(log[-1], flush=True)
    pd.DataFrame(log).to_csv(f"{OUT}/log.tsv", sep="\t", index=False)
    if FIXED:
        # fixed budget: every epoch is checkpointed for post-hoc monitoring; model.pt is always the latest (= last) epoch;
        # validation metrics drive nothing (no selection, no patience)
        torch.save(net.state_dict(), f"{OUT}/ckpt/ep{ep:02d}.pt"); torch.save(net.state_dict(), f"{OUT}/model.pt")
        sched.step()
    elif vl < best:
        best, bad = vl, 0; torch.save(net.state_dict(), f"{OUT}/model.pt")
    else:
        bad += 1
        if bad >= args.patience and ep + 1 >= args.min_epochs:
            break
def rel_nn_dist(ids):
    """Chebyshev distance to the nearest training fluxome of the same line / the sample's own |v_KO - v_WT|_inf."""
    out = np.full(len(ids), np.nan); ids = np.asarray(ids)
    for ln in lines:
        trl = [samples[i][1] for i in train if samples[i][0] == ln]; sel = np.where(sline[ids] == ln)[0]
        if not trl or not len(sel):
            continue
        Vt = L[ln]["V"]; A_ = Vt[[samples[i][1] for i in ids[sel]]]; B_ = Vt[trl]; vw = L[ln]["vwt"]
        c = np.where((np.abs(A_ - vw).max(0) > 0) | (np.abs(B_ - vw).max(0) > 0))[0]
        dmin = np.abs(A_[:, None, c] - B_[None, :, c]).max(2).min(1)
        out[sel] = dmin / np.abs(A_[:, c] - vw[c]).max(1)
    return out


net.load_state_dict(torch.load(f"{OUT}/model.pt", weights_only=True))
for name, ids in (("val", val), ("test", test)):
    ii, pp = predict(ids)
    pd.DataFrame(dict(line=sline[ii], symbol=sym[ii], gene=gid[ii], dependency=dep[ii], label=lab[ii], gnn_prob=pp,
                      dup_in_train=dup(ii), rel_nn_dist=rel_nn_dist(ii))).to_csv(f"{OUT}/pred_{name}.tsv", sep="\t", index=False)
    if name != "val":
        print(f"{name}: n {len(ii)} AUROC {roc_auc_score(lab[ii], pp):.3f} AUPRC {average_precision_score(lab[ii], pp):.3f}", flush=True)
if FIXED and args.protocol_lock != "none":
    # protocol lock (review_stopping.md section 3, item 3): this fold's config.json must match one registered arm
    import importlib.util
    _sp = importlib.util.spec_from_file_location("chk101", f"{ROOT}/script_python/101_check_protocol_v3.py")
    _chk = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_chk)
    _ok, _arm, _mism = _chk.check_config(json.load(open(f"{OUT}/config.json")), json.load(open(args.protocol_lock)))
    print(f"protocol lock check ({os.path.basename(args.protocol_lock)}): {'PASS' if _ok else 'MISMATCH'} arm={_arm}" + ("" if _ok else " " + "; ".join(_mism)), flush=True)
    if not _ok:
        print("PROTOCOL LOCK MISMATCH: outputs kept, exit 3", flush=True); raise SystemExit(3)
elif FIXED:
    print("protocol lock check skipped (--protocol-lock none)", flush=True)
print(f"done {time.time() - t0:.0f}s", flush=True)
