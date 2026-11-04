"""Predict-only script for saved GNN v2 fold models (script 81b: model.pt + config.json [+ split.json]) on a dataset built
by script 64 / 64c, written for the external validation set of non-colorectal cell lines (term pending ratification;
plan result/gnn_20261028/discussion_20261030/external_eval_plan.md), 2026-10-30.

No training, no metric. The graph builders, the fluxome threshold (1e-6 / mu_ref), the fluxome-change sample filter, the
growth mask and the Net class are copied verbatim from 81b (sections "data", "graph builders", "Net"); state_dict is
loaded strictly, so an architecture mismatch fails loudly. Medium: no-medium models get the constant; a with-medium model
gets the medium features z-scored with the statistics of ITS training cell lines (config.json lines minus split.json
test_lines and val_lines, as 81b) applied to the new cell lines' medium vectors (ablation only; the models of record take
no medium).
Output per fold: <out>/fold<k>/pred_external.tsv, '#' header lines (model path and md5, config summary, dataset and its
samples.tsv md5, n, filter, date), columns line, symbol, gene, dependency, label, has_label, growth_ratio, gnn_prob.
rel_nn_dist is NOT written: it is defined to training fluxomes of the same cell line and no external cell line has any.
Usage: python 99_gnn_predict_external.py --model-root <run folder with fold*/> --dataset <dataset folder> --out <folder>
       [--folds 0 1 ...] [--device auto|cpu|cuda] [--subset N (smoke test only)] [--workers 4]
Env: crcgem_gnn.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp, torch
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn import global_max_pool, global_mean_pool

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def med_feat(lb):
    return np.concatenate([(lb == 0), (lb <= -999), np.where((lb < 0) & (lb > -999), slog(-lb * 1000), 0)]).astype(np.float32)


def load_dataset(D, mask_file):
    """Dataset folder of script 64/64c -> dict (81b 'data' section: threshold, fluxome-change filter, growth mask)."""
    meta = json.load(open(f"{D}/scaffold_meta.json"))
    S = sp.load_npz(f"{D}/scaffold_S.npz").tocsc()
    NR, NM = S.shape[1], S.shape[0]
    mpos = {m: i for i, m in enumerate(meta["metabolites"])}
    cur_idx = np.array([mpos[m] for m in meta["currency"] if m in mpos])
    is_cur = np.zeros(NM, bool); is_cur[cur_idx] = True
    MASKJ = json.load(open(mask_file)); MASK = np.array(sorted(set(MASKJ["mask_index"])), int)
    BIO = meta["reactions"].index(meta["biomass"])
    assert BIO in set(MASK.tolist()), "mask must contain the biomass reaction"
    assert all(meta["reactions"][i] == r for i, r in zip(MASKJ["mask_index"], MASKJ["mask_reactions"])), "mask index/id mismatch"
    KEEP = np.ones(NR, bool); KEEP[MASK] = False
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
    return dict(meta=meta, S=S, NR=NR, NM=NM, cur_idx=cur_idx, is_cur=is_cur, MASK=MASK, MASKJ=MASKJ, BIO=BIO, KEEP=KEEP,
                lines=lines, L=L, samples=samples)


def make_build(ds, graph, presence, MED):
    """81b 'graph builders' section, verbatim up to the closure variables. Returns (build, F_IN, E_IN, info)."""
    S, L, samples, KEEP, is_cur, cur_idx = ds["S"], ds["L"], ds["samples"], ds["KEEP"], ds["is_cur"], ds["cur_idx"]
    NPRES = 1 if presence else 0
    if graph == "B":
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
                        med=torch.tensor(MED[ln])[None], y=torch.tensor([0.0]), idx=torch.tensor([idx]))
        E_IN = 3
        info = dict(graph="B", n_metabolite_nodes=int(nM), n_reaction_nodes=int(nRB), n_directed_edges=int(EI.shape[1]))
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
                        edge_attr=torch.tensor(ea), med=torch.tensor(MED[ln])[None], y=torch.tensor([0.0]),
                        idx=torch.tensor([idx]))
        F_IN, E_IN = 3 + NPRES + 2 * len(cur_idx), 2
        info = dict(graph="A", n_currency=int(len(cur_idx)))
    return build, F_IN, E_IN, info


class Net(nn.Module):
    """81b Net, verbatim (readout passed in instead of read from args)."""
    def __init__(self, f_in, e_in, m_in, h, readout):
        super().__init__()
        self.inp = nn.Linear(f_in, h)
        self.convs = nn.ModuleList([GATv2Conv(h, h, heads=1, edge_dim=e_in) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(3)])
        self.readout = readout
        if self.readout == "attn_max":
            self.pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1)),
                                               nn=nn.Linear(h, h))
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


def medium_features(cfg, fold_dir, ds):
    """Medium input per new cell line. No-medium models: constant. Else z-score with the training cell lines of this fold."""
    if cfg.get("no_medium", False):
        return {ln: np.zeros(1, np.float32) for ln in ds["lines"]}, "constant (no-medium model)"
    TR = cfg.get("dataset", f"{ROOT}/result/gnn_20261028/dataset")   # the training dataset (medium statistics are training-cell-line statistics); 81b configs since 2026-10-30 record it
    sj = json.load(open(f"{fold_dir}/split.json"))
    tr_lines = [ln for ln in cfg["lines"] if ln not in set(sj["test_lines"]) | set(sj["val_lines"])]
    MEDM = np.stack([med_feat(np.load(f"{TR}/lines/{ln}.npz")["medium_lb"]) for ln in tr_lines]).astype(np.float64)
    sd = MEDM.std(0); keep = sd > 1e-6; mu = MEDM[:, keep].mean(0)
    out = {ln: ((med_feat(ds["L"][ln]["medium_lb"]).astype(np.float64)[keep] - mu) / sd[keep]).astype(np.float32) for ln in ds["lines"]}
    return out, f"z-scored over {len(tr_lines)} training cell lines of this fold, {int(keep.sum())} varying columns (WITH-MEDIUM MODEL: ablation)"


def predict_fold(fold_dir, ds, out_dir, device, workers, batch, subset=0):
    cfg = json.load(open(f"{fold_dir}/config.json"))
    graph, presence, readout, hidden = cfg["graph"], cfg.get("presence", False), cfg.get("readout", "attn_max"), cfg.get("hidden", 64)
    MED, med_note = medium_features(cfg, fold_dir, ds)
    build, F_IN, E_IN, info = make_build(ds, graph, presence, MED)
    assert F_IN == cfg["f_in"] and E_IN == cfg["e_in"], f"feature width mismatch: built {F_IN}/{E_IN}, config {cfg['f_in']}/{cfg['e_in']}"
    net = Net(F_IN, E_IN, len(MED[ds["lines"][0]]), hidden, readout).to(device)
    sd = torch.load(f"{fold_dir}/model.pt", weights_only=True, map_location=device)
    net.load_state_dict(sd, strict=True); net.eval()
    ids = np.arange(len(ds["samples"]))
    if subset:
        ids = ids[:subset]

    class DS(torch.utils.data.Dataset):
        def __len__(self): return len(ids)
        def __getitem__(self, k): return build(int(ids[k]))
    dl = torch.utils.data.DataLoader(DS(), batch_size=batch, num_workers=workers, collate_fn=Batch.from_data_list, shuffle=False)
    out_i, out_p = [], []
    with torch.no_grad():
        for b in dl:
            b = b.to(device); out_p.append(torch.sigmoid(net(b)).cpu().numpy()); out_i.append(b.idx.cpu().numpy())
    ii, pp = np.concatenate(out_i), np.concatenate(out_p)
    assert np.array_equal(np.sort(ii), np.sort(ids)) and np.isfinite(pp).all(), "prediction set or finiteness check failed"
    L, samples, BIO = ds["L"], ds["samples"], ds["BIO"]
    rows = []
    for k, i in enumerate(ii):
        ln, j = samples[i]; z = L[ln]; d = float(z["dependency"][j]); hl = bool(np.isfinite(d))
        rows.append(dict(line=ln, symbol=str(z["symbols"][j]), gene=str(z["genes"][j]), dependency=d if hl else np.nan,
                         label=int(d >= 0.5) if hl else -1, has_label=hl, growth_ratio=float(z["V"][j, BIO]), gnn_prob=float(pp[k])))
    df = pd.DataFrame(rows).sort_values(["line", "gene"]).reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)
    smp_md5 = md5(f"{ds['path']}/samples.tsv")
    with open(f"{out_dir}/pred_external.tsv", "w") as fh:
        fh.write(f"# script_python/99_gnn_predict_external.py; model {fold_dir}/model.pt md5 {md5(f'{fold_dir}/model.pt')}; {time.ctime()}\n")
        fh.write(f"# config: graph {graph}, split {cfg.get('split')}, fold {cfg.get('fold')}, seed {cfg.get('seed')}, readout {readout}, "
                 f"presence {presence}, no_medium {cfg.get('no_medium', False)}, hidden {hidden}, max_epochs {cfg.get('max_epochs')}, "
                 f"f_in {F_IN}, e_in {E_IN}; mask {cfg['mask_file']} ({len(ds['MASK'])} reactions); medium: {med_note}\n")
        fh.write(f"# dataset {ds['path']} (samples.tsv md5 {smp_md5}); fluxome-change filter |v_KO - v_WT|_inf > 1e-6/mu_ref as 81b; "
                 f"n {len(df)}{' (SUBSET smoke test)' if subset else ''} of {len(samples)} passing samples over {len(ds['lines'])} cell lines; "
                 f"label -1 = no DepMap row; rel_nn_dist undefined (no training fluxome of these cell lines) and not written; no metric computed\n")
        df.to_csv(fh, sep="\t", index=False)
    return df, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", required=True, help="run folder holding fold<k>/{model.pt,config.json,split.json}")
    ap.add_argument("--dataset", default=f"{ROOT}/result/external_20261030/dataset")
    ap.add_argument("--out", required=True, help="output folder; fold<k>/pred_external.tsv is written below it (no overwrite)")
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--subset", type=int, default=0, help="smoke test only: first N samples")
    ap.add_argument("--mask-file", default="", help="default: the mask_file recorded in the first fold's config.json")
    args = ap.parse_args()
    t0 = time.time()
    folds = args.folds if args.folds is not None else sorted(int(d[4:]) for d in os.listdir(args.model_root) if d.startswith("fold") and os.path.exists(f"{args.model_root}/{d}/model.pt"))
    assert folds, f"no fold*/model.pt under {args.model_root}"
    cfg0 = json.load(open(f"{args.model_root}/fold{folds[0]}/config.json"))
    mask_file = args.mask_file or cfg0["mask_file"]
    device = torch.device("cuda:0" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "")) else "cpu")
    ds = load_dataset(args.dataset, mask_file); ds["path"] = args.dataset.rstrip("/")
    print(f"dataset {args.dataset}: {len(ds['lines'])} cell lines, {len(ds['samples'])} samples pass the fluxome-change filter; "
          f"mask {len(ds['MASK'])}; device {device}; folds {folds}; {time.time()-t0:.0f}s", flush=True)
    for f in folds:
        fd = f"{args.model_root}/fold{f}"; od = f"{args.out}/fold{f}"
        assert not os.path.exists(f"{od}/pred_external.tsv"), f"{od}/pred_external.tsv exists (no overwrite)"
        df, info = predict_fold(fd, ds, od, device, args.workers, args.batch, args.subset)
        print(f"fold {f}: {len(df)} rows written to {od}/pred_external.tsv; gnn_prob finite {np.isfinite(df.gnn_prob).all()}, "
              f"range written to file only; graph info {info}; {time.time()-t0:.0f}s", flush=True)
    print(f"done {time.time()-t0:.0f}s")
