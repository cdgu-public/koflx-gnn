"""E2 / E3 controls on GNN v2 (script 81 derived, 2026-10-29; 81, 81b, 85, 86 untouched). review_gem.md section 8.5,
user-approved 2026-10-29 13:40 (status.md). Identical to 81 unless one of the flags below is set.

E3  --shuffle-graph relabel|rewire   (A' only) graph-shuffled control, fixed per sample (rng seed 1000*seed + sample idx,
    so the same shuffled graph is used in every epoch and at test). relabel: node permutation applied to edge_index only,
    node features stay in place => topology (degrees, MFG weights) preserved, the correspondence between a node's features
    and its structural position destroyed (primary control). rewire: edge targets permuted (source and edge attribute kept)
    => topology destroyed, out-degrees kept. Folder token _shuf<mode>.
E2a --no-early-stop                 fit check: train to --max-epochs, log per-epoch train AUROC/BCE on a fixed training
    probe subset (--train-probe N, eval mode), keep model.pt = best validation BCE (as 81, pred_test.tsv for script 85) and
    model_last.pt / pred_test_last.tsv = last epoch. Folder token _fitchk.
Arm 2 --rxn-embed D                 learnable identity embedding (nn.Embedding over the 12,931 scaffold reactions + 8,461 metabolites,
    indexed by scaffold index; A' uses the reaction rows, B both) concatenated to the node features. Folder token _rid<D>.
Arm 5 --stop-metric auroc            model selection and patience on validation AUROC instead of validation BCE. Token _stopauroc.
WT-only --wt-only                    (A' only) the same predictor given only the wild-type flux graph (FlowGAT-type control, user 2026-10-29).
    Nodes = reactions active in WT plus the reactions the knockout disables (scaffold_meta gene_rxns), growth mask applied.
    Node features [slog v_WT, disabled flag (1 on the knockout's reactions), 0] + currency columns from v_WT (the KO currency
    columns are 0); edges = WT Mass Flow Graph, edge attribute [slog W_WT, 0]. The KO fluxome is never read. F_IN, E_IN and the
    network are unchanged. The disabled-reaction flag is the only knockout information (without it every knockout of a cell
    line would get the same prediction); it is the knocked-out-reaction flag that the 2026-10-27 ruling removed from A' when
    v_KO carried the perturbation. Token _wtonly.
Disabled flag --disabled-flag        (A' only, 2026-11-01, user ruling "바로 시작"; design result/gnn_20261030_tolpatch/design_flag_arm_20261101.md)
    the baseline A' input plus ONE extra node-feature column: the disabled-reaction flag of the WT-only arm (1 on the knockout's
    disabled reactions, scaffold_meta gene_rxns, growth mask applied). Node set (WT-active or KO-active), edges, edge attributes,
    the KO flux columns and the currency columns are the baseline's; F_IN grows by 1 (input layer +64 parameters), nothing else
    changes. Separates the WT-only result into (flag effect) vs (KO fluxome effect). Not combinable with --wt-only. Token _disflag.
E2b --permute-labels                memorization control: soft target and binary label permuted jointly across the
    TRAINING samples only (validation and test labels untouched; pred_test.tsv label column is the true label). Combine with
    --no-early-stop and --train-subset N (fixed random subset of the training set, sampler and probe restricted to it).
    Folder token _permlab (+ _sub<N>).
--device                             cuda:0 (default) or cpu (smoke only).
--protocol fixed_cosine              (2026-10-30, same semantics as 81b_gnn_train_v2.py; lock file result/gnn_20261028/
    protocol_lock_v3_20261030.json, control arms) fixed budget of --max-epochs epochs, no early stopping, no validation-based
    checkpoint selection, AdamW lr cosine-annealed per epoch from --lr to --lr-min (Loshchilov & Hutter 2017, arXiv:1608.03983),
    model.pt = last epoch, per-epoch ckpt/ep<k>.pt for 100_monitor_v3.py (monitoring only), log.tsv gains an lr column, folder
    token _fixedcos<max_epochs> before _s<seed>, an existing fold folder is refused, config.json records protocol, lr_min,
    lr_schedule, checkpoint, script and is checked against --protocol-lock at the end ('none' skips, smoke only). Not combinable
    with --no-early-stop or --stop-metric auroc (stopping-rule controls). The default --protocol es_valbce is the unchanged path.
Output: <out-root>/<graph>_<split>[_nomed][_shuf<mode>][_fitchk][_permlab][_sub<N>]_s<seed><tag>/fold<k>/, read by
85_eval_gnn_v2.py as variant tokens (folders with "precheck" or "smoke" in the tag are skipped by 85 only for "precheck";
smoke runs go to a separate --out-root).
Usage: CUDA_VISIBLE_DEVICES=0 python 90_gnn_e2e3_v2.py --graph A --split gene --fold 0 --shuffle-graph relabel --max-epochs 30
       CUDA_VISIBLE_DEVICES=0 python 90_gnn_e2e3_v2.py --graph A --split gene --fold 0 --no-early-stop --max-epochs 30
       CUDA_VISIBLE_DEVICES=0 python 90_gnn_e2e3_v2.py --graph A --split gene --fold 0 --permute-labels --no-early-stop --train-subset 2000 --max-epochs 30
Script 81 docstring follows.
"""
"""GNN v2 on the MOMA knockout fluxomes (dataset of script 64), graphs A′ and B. Script 73 (v1) is frozen.

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

Changes from v1 (script 73; fix of the graph-B constant-output collapse, diagnosis 2026-10-28 scripts 79 + fable):
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
  --split gene (primary, vs MOMA): groups = the set of scaffold reactions the knockout disables (546 groups;
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
from torch_geometric.nn import global_max_pool
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
ap.add_argument("--shuffle-graph", choices=["", "relabel", "rewire"], default="", help="E3: graph-shuffled control (A' only)")
ap.add_argument("--no-early-stop", action="store_true", help="E2a: train to --max-epochs; keep best-val and last models")
ap.add_argument("--permute-labels", action="store_true", help="E2b: permute soft target and label within the training set")
ap.add_argument("--train-subset", type=int, default=0, help="E2b: fixed random subset of the training set (0 = all)")
ap.add_argument("--train-probe", type=int, default=2000, help="per-epoch train AUROC/BCE on this many fixed training samples (0 = off)")
ap.add_argument("--rxn-embed", type=int, default=0, help="Arm 2: learnable identity embedding of this width per scaffold reaction (B: and per metabolite); 0 = off")
ap.add_argument("--stop-metric", choices=["bce", "auroc"], default="bce", help="Arm 5: model selection / patience on validation BCE (81) or validation AUROC")
ap.add_argument("--wt-only", action="store_true", help="WT-only arm (A' only): graph and node features from the wild-type fluxome; the knockout enters only as a node flag on its disabled reactions")
ap.add_argument("--disabled-flag", action="store_true", help="baseline-plus-flag arm (A' only): baseline input plus one node column = disabled-reaction flag (the WT-only arm's flag); node set and all other features unchanged")
ap.add_argument("--protocol", choices=["es_valbce", "fixed_cosine"], default="es_valbce",
                help="es_valbce = script 81 (early stopping on validation BCE, best-validation checkpoint); fixed_cosine = fixed budget of "
                     "--max-epochs epochs, cosine annealing --lr -> --lr-min per epoch, last-epoch checkpoint, per-epoch ckpt/ (2026-10-30, as 81b)")
ap.add_argument("--lr-min", type=float, default=1e-5, help="fixed_cosine only: final learning rate of the cosine schedule")
ap.add_argument("--protocol-lock", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/protocol_lock_v3_20261030.json",
                help="fixed_cosine only: lock file checked against config.json at the end of the run; 'none' skips (smoke tests)")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out-root", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/runs_v2")
ap.add_argument("--dataset", default="/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/gnn_20261028/dataset",
                help="dataset folder (script 64 / 64d); default = the 2026-10-28 dataset, unchanged behavior. 2026-10-30 tolpatch: result/gnn_20261030_tolpatch/dataset; recorded in config.json and checked by the lock")
args = ap.parse_args()
FIXED = args.protocol == "fixed_cosine"
assert not (FIXED and (args.no_early_stop or args.stop_metric != "bce")), "fixed_cosine has no stopping rule: --no-early-stop / --stop-metric are es_valbce controls"
ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
D = args.dataset.rstrip("/")
assert not (args.shuffle_graph and args.graph == "B"), "E3 shuffle is defined for A' only"
VAR = (f"{'_presence' if args.presence else ''}{'_nomed' if args.no_medium else ''}{'_shuf' + args.shuffle_graph if args.shuffle_graph else ''}"
       f"{'_fitchk' if args.no_early_stop and not args.permute_labels else ''}{'_permlab' if args.permute_labels else ''}"
       f"{'_sub' + str(args.train_subset) if args.train_subset else ''}{'_rid' + str(args.rxn_embed) if args.rxn_embed else ''}"
       f"{'_stopauroc' if args.stop_metric == 'auroc' else ''}{'_wtonly' if args.wt_only else ''}{'_disflag' if args.disabled_flag else ''}"
       f"{f'_fixedcos{args.max_epochs}' if FIXED else ''}")
assert not (args.wt_only and args.graph == "B"), "the WT-only arm is defined for A' only"
assert not (args.disabled_flag and args.graph == "B"), "the disabled-flag arm is defined for A' only"
assert not (args.disabled_flag and args.wt_only), "--disabled-flag adds the flag to the baseline input; the WT-only arm already carries it"
OUT = f"{args.out_root}/{args.graph}_{args.split}{VAR}_s{args.seed}{args.tag}/fold{args.fold}"
if FIXED and os.path.exists(OUT):
    raise SystemExit(f"REFUSED: fold folder exists, fixed_cosine runs never overwrite: {OUT}")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(1000 * args.seed + args.fold); rng = np.random.default_rng(1000 * args.seed + args.fold)
dev = torch.device(args.device)
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
# ---------------- E2 controls: training-set subset and label permutation (test and validation untouched) ----------------
if args.train_subset:
    train = np.sort(np.random.default_rng(1000 * args.seed + args.fold).choice(train, min(args.train_subset, len(train)), replace=False))
    print(f"E2 train subset: {len(train)} training samples", flush=True)
if args.permute_labels:
    pp_ = np.random.default_rng(1000 * args.seed + args.fold + 7).permutation(len(train))
    tgt[train], lab[train] = tgt[train][pp_], lab[train][pp_]   # joint permutation; dep (reported in pred files) untouched
    print(f"E2 permuted labels on {len(train)} training samples; train pos rate after permutation {lab[train].mean():.3f}", flush=True)
DIS = [np.asarray(gene_rxns.get(g, []), int) for g in gid]   # scaffold reaction indices the knockout disables (WT-only arm node flag)
if args.wt_only or args.disabled_flag:
    print(f"{'WT-only' if args.wt_only else 'disabled-flag'} arm: {sum(len(d) == 0 for d in DIS)} of {len(DIS)} samples have an empty disabled set; "
          f"median disabled reactions {int(np.median([len(d) for d in DIS]))}", flush=True)
probe_ids = (np.sort(np.random.default_rng(1000 * args.seed + args.fold + 11).choice(train, min(args.train_probe, len(train)), replace=False))
             if args.train_probe else np.array([], int))
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
    RID_B = torch.tensor(np.concatenate([M_used + NR, R]), dtype=torch.long)   # identity index per node (Arm 2)
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
        return Data(x=torch.tensor(np.concatenate([met_x, rx])), edge_index=EI_t, edge_attr=torch.tensor(ea), rid=RID_B,
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
        if args.wt_only:   # WT-only arm: the KO fluxome is never read; nodes = WT-active reactions plus the knockout's disabled reactions
            dis_ = np.zeros(NR, bool); dis_[DIS[idx]] = True
            A = np.where(((vw_all != 0) | dis_) & KEEP)[0]
            vk_all = np.zeros_like(vw_all)   # => slog(v_KO) = 0, KO mass-flow weights empty, KO currency columns 0
        else:
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
        if args.disabled_flag:   # baseline-plus-flag arm: one extra column, 1 on the knockout's disabled reactions that are nodes (node set unchanged)
            dis_ = np.zeros(NR, bool); dis_[DIS[idx]] = True
            cols.append(dis_[A].astype(np.float32))
        x = np.concatenate([np.stack(cols, 1), slog(cw), slog(ck)], 1).astype(np.float32)
        if args.wt_only:   # node columns become [slog v_WT, disabled flag, 0]; WT currency columns kept, KO currency columns are 0
            x[:, 1] = dis_[A]; x[:, 2] = 0
        ea = np.stack([slog(W.data), slog(W2.data)], 1).astype(np.float32)
        erow, ecol = W.row, W.col
        if args.shuffle_graph:   # E3: fixed per sample; features x and edge attributes ea untouched
            g_ = np.random.default_rng(1000 * args.seed + idx)
            if args.shuffle_graph == "relabel":
                perm_ = g_.permutation(len(A)); erow, ecol = perm_[erow], perm_[ecol]
            else:
                ecol = g_.permutation(ecol)
        return Data(x=torch.tensor(x), edge_index=torch.tensor(np.stack([erow, ecol]), dtype=torch.long),
                    edge_attr=torch.tensor(ea), med=torch.tensor(MED[ln])[None], y=torch.tensor([tgt[idx]]),
                    idx=torch.tensor([idx]), rid=torch.tensor(A, dtype=torch.long))
    F_IN, E_IN = 3 + NPRES + int(args.disabled_flag) + 2 * len(cur_idx), 2


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
    def __init__(self, f_in, e_in, m_in, h, n_id=0, d_id=0):
        super().__init__()
        self.emb = nn.Embedding(n_id, d_id) if d_id else None   # Arm 2: identity embedding, concatenated to the node features
        self.inp = nn.Linear(f_in + d_id, h)
        self.convs = nn.ModuleList([GATv2Conv(h, h, heads=1, edge_dim=e_in) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(3)])
        self.pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1)),
                                           nn=nn.Linear(h, h))
        self.med = nn.Sequential(nn.Linear(m_in, h), nn.LeakyReLU(0.1))
        self.head = nn.Sequential(nn.Linear(3 * h, h), nn.LeakyReLU(0.1), nn.Dropout(0.2), nn.Linear(h, 1))

    def embed(self, d):
        x = self.inp(torch.cat([d.x, self.emb(d.rid)], 1) if self.emb is not None else d.x)
        for c, n in zip(self.convs, self.norms):
            x = n(x + torch.relu(c(x, d.edge_index, d.edge_attr)))
        return x

    def forward(self, d):
        x = self.embed(d)
        g = torch.cat([self.pool(x, d.batch), global_max_pool(x, d.batch), self.med(d.med)], 1)
        return self.head(g).squeeze(1)

    def diag(self, d):
        """On one batch (eval mode): logit std across graphs, fraction of head units with a positive pre-activation
        in any graph, attention entropy (mean over graphs; uniform = ln n_nodes) and max attention weight."""
        x = self.embed(d)
        a = pyg_softmax(self.pool.gate_nn(x).squeeze(-1), d.batch)
        ent = torch.stack([-(a[d.batch == q] * a[d.batch == q].clamp_min(1e-30).log()).sum() for q in range(d.num_graphs)])
        g = torch.cat([self.pool(x, d.batch), global_max_pool(x, d.batch), self.med(d.med)], 1)
        h1 = self.head[0](g); logit = self.head(g).squeeze(1)
        return dict(logit_std=round(logit.std().item(), 4), head_live=round((h1 > 0).any(0).float().mean().item(), 3),
                    att_entropy=round(ent.mean().item(), 3), att_max=round(a.max().item(), 4))


net = Net(F_IN, E_IN, len(MED[lines[0]]), args.hidden, n_id=NR + NM, d_id=args.rxn_embed).to(dev)
print(f"parameters {sum(p.numel() for p in net.parameters())} (identity embedding {args.rxn_embed} x {NR + NM if args.rxn_embed else 0})", flush=True)
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


cfg = dict(vars(args), n_train=len(train), n_val=len(val), n_test=len(test), f_in=F_IN, e_in=E_IN, n_train_probe=len(probe_ids), lines=lines)
if FIXED:
    # readout attn_max = the script-81 readout this script inherits (gated attention + global max pooling); script names the writer
    cfg.update(lr_schedule="cosine_annealing_per_epoch_no_restarts", checkpoint="last_epoch", early_stopping=False,
               weight_decay=1e-4, optimizer="AdamW", ckpt_dir="ckpt", protocol_date="2026-10-30", script="90_gnn_e2e3_v2.py", readout="attn_max")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs, eta_min=args.lr_min)
    os.makedirs(f"{OUT}/ckpt", exist_ok=True)
else:
    for k in ("protocol", "lr_min", "protocol_lock"):   # default path: config.json keys unchanged
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
    if len(probe_ids):
        ti, tp = predict(probe_ids)
        dg = dict(dg, train_auroc_probe=round(float(roc_auc_score(lab[ti], tp)), 4), train_bce_probe=round(bce(ti, tp), 4))
    log.append(dict(epoch=ep, train_loss=tl / nb, val_bce=vl, val_auroc=roc_auc_score(lab[vi], vp),
                    val_auprc=average_precision_score(lab[vi], vp), **dg,
                    epoch_s=round(time.time() - te), peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)))
    if FIXED:
        log[-1]["lr"] = opt.param_groups[0]["lr"]   # learning rate used in this epoch
    print(log[-1], flush=True)
    pd.DataFrame(log).to_csv(f"{OUT}/log.tsv", sep="\t", index=False)
    if FIXED:
        # fixed budget (as 81b): every epoch is checkpointed for post-hoc monitoring; model.pt is always the latest (= last) epoch;
        # validation metrics drive nothing (no selection, no patience)
        torch.save(net.state_dict(), f"{OUT}/ckpt/ep{ep:02d}.pt"); torch.save(net.state_dict(), f"{OUT}/model.pt")
        sched.step()
        continue
    crit = vl if args.stop_metric == "bce" else -float(log[-1]["val_auroc"])   # Arm 5: selection and patience on val AUROC
    if crit < best:
        best, bad = crit, 0; torch.save(net.state_dict(), f"{OUT}/model.pt")
    else:
        bad += 1
        if bad >= args.patience and ep + 1 >= args.min_epochs and not args.no_early_stop:
            break
if args.no_early_stop:
    torch.save(net.state_dict(), f"{OUT}/model_last.pt")
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


if args.no_early_stop:   # last-epoch model first (E2a), then 81's best-validation model
    ii, pp = predict(test)
    pd.DataFrame(dict(line=sline[ii], symbol=sym[ii], gene=gid[ii], dependency=dep[ii], label=lab[ii], gnn_prob=pp,
                      dup_in_train=dup(ii))).to_csv(f"{OUT}/pred_test_last.tsv", sep="\t", index=False)
    print(f"test (last epoch {len(log) - 1}): AUROC {roc_auc_score(lab[ii], pp):.3f}", flush=True)
net.load_state_dict(torch.load(f"{OUT}/model.pt", weights_only=True))
if len(probe_ids):
    ii, pp = predict(probe_ids)
    pd.DataFrame(dict(line=sline[ii], symbol=sym[ii], gene=gid[ii], dependency=dep[ii], label_used=lab[ii], gnn_prob=pp)).to_csv(
        f"{OUT}/pred_train_probe.tsv", sep="\t", index=False)
    print(f"train probe (best-val model): n {len(ii)} AUROC vs label used in training {roc_auc_score(lab[ii], pp):.3f}", flush=True)
for name, ids in (("val", val), ("test", test)):
    ii, pp = predict(ids)
    pd.DataFrame(dict(line=sline[ii], symbol=sym[ii], gene=gid[ii], dependency=dep[ii], label=lab[ii], gnn_prob=pp,
                      dup_in_train=dup(ii), rel_nn_dist=rel_nn_dist(ii))).to_csv(f"{OUT}/pred_{name}.tsv", sep="\t", index=False)
    if name != "val":
        print(f"{name}: n {len(ii)} AUROC {roc_auc_score(lab[ii], pp):.3f} AUPRC {average_precision_score(lab[ii], pp):.3f}", flush=True)
if FIXED and args.protocol_lock != "none":
    # protocol lock (review_stopping.md section 3, item 3): this fold's config.json must match one registered control arm
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
