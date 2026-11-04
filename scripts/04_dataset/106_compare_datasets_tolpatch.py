"""Compare the solver-tolerance-patched GNN dataset (script 64d from result/moma_20261030_tolpatch) with the 2026-10-28 dataset
(script 64 from result/moma_20261027): what changed in the training inputs, the sample set, the splits and the evaluation strata.

Measured per cell line and overall (every number from the two dataset folders, no model involved):
  npz keys     vwt, present, medium_lb, mu_ref identical? genes identical? rows of V that differ (count, max |dV|), and which of them are patched
  samples.tsv  rows, status and growth_ratio changes (patched rows only expected), MOMA-lethal count old vs new, label unchanged
  sample set   knockouts passing the 81b fluxome-change filter (|v_KO - v_WT|_inf > 1e-6 / mu_ref), old vs new, entering / leaving
  splits       gene-disjoint fold of every sample (group_folds / val_groups of 81b, seed-independent) old vs new: samples whose test
               fold changes; cell-line-disjoint folds (seeded permutation of cell lines; unchanged by construction, verified)
  strata       MOMA-lethal (G < 0.05), G > 0.99 counts per fold old vs new; exact-duplicate flag (fluxome equal to a training fluxome of
               the fold) old vs new. rel_nn_dist needs the trained fold's training set and is recomputed here exactly as 81b (per cell
               line, Chebyshev distance to the nearest training fluxome / own |v_KO - v_WT|_inf) for the test part of every gene fold.
Output: <out>/{dataset_diff_per_line.tsv, dataset_diff_summary.tsv, strata_per_fold.tsv, sample_set_changes.tsv} with '#' condition lines.
Usage: python 106_compare_datasets_tolpatch.py [--old result/gnn_20261028/dataset] [--new result/gnn_20261030_tolpatch/dataset]
       [--out result/gnn_20261030_tolpatch/dataset_diff]
Env: crcgem_gnn.
"""
import argparse, json, os, time
import numpy as np, pandas as pd

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
ap = argparse.ArgumentParser()
ap.add_argument("--old", default=f"{ROOT}/result/gnn_20261028/dataset")
ap.add_argument("--new", default=f"{ROOT}/result/gnn_20261030_tolpatch/dataset")
ap.add_argument("--out", default=f"{ROOT}/result/gnn_20261030_tolpatch/dataset_diff")
ap.add_argument("--nfold", type=int, default=5)
args = ap.parse_args()
assert not os.path.exists(f"{args.out}/dataset_diff_summary.tsv"), f"{args.out} exists (no overwrite)"
os.makedirs(args.out, exist_ok=True); t0 = time.time()
HDR = (f"# script_python/106_compare_datasets_tolpatch.py; old {args.old}; new {args.new}; filter |v_KO - v_WT|_inf > 1e-6/mu_ref and threshold 1e-6/mu_ref on every flux (81b:125-132); "
       f"splits = 81b group_folds / val_groups (gene) and seeded cell-line permutation (line), nfold {args.nfold}; lethal = growth_ratio < 0.05; date {time.strftime('%Y-%m-%d %H:%M %Z')}\n")


def load(D):
    meta = json.load(open(f"{D}/scaffold_meta.json"))
    lines = sorted(f[:-4] for f in os.listdir(f"{D}/lines")); L = {}
    for ln in lines:
        z = np.load(f"{D}/lines/{ln}.npz"); d = {k: z[k] for k in z.files}; th = 1e-6 / float(d["mu_ref"]); d["th"] = th
        d["vwt_raw"], d["V_raw"] = d["vwt"].copy(), d["V"].copy()
        d["vwt"] = np.where(np.abs(d["vwt"]) > th, d["vwt"], 0).astype(np.float32)
        d["V"] = np.where(np.abs(d["V"]) > th, d["V"], 0).astype(np.float32); L[ln] = d
    smp = pd.read_csv(f"{D}/samples.tsv", sep="\t", comment="#")
    return meta, lines, L, smp


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


def sample_table(meta, lines, L, smp):
    gene_rxns = meta["gene_rxns"]
    rows = []
    for ln in lines:
        d = L[ln]
        for i in range(len(d["genes"])):
            if np.abs(d["V"][i] - d["vwt"]).max() > d["th"]:
                rows.append((ln, str(d["genes"][i]), i))
    S = pd.DataFrame(rows, columns=["line", "gene", "i"])
    S["gkey"] = [",".join(map(str, sorted(gene_rxns.get(g, [])))) for g in S.gene]
    S["fold_gene"] = group_folds(S.gkey.to_numpy(), args.nfold)
    perm = list(np.random.default_rng(0).permutation(lines)); lf = {ln: k % args.nfold for k, ln in enumerate(perm)}
    S["fold_line"] = S.line.map(lf)
    gr = smp.set_index(["line", "gene"]).growth_ratio
    S["growth_ratio"] = gr.reindex(list(zip(S.line, S.gene))).to_numpy()
    S["label"] = smp.set_index(["line", "gene"]).label.reindex(list(zip(S.line, S.gene))).to_numpy()
    return S


def strata_per_fold(S, L, tag):
    out = []
    for f in range(args.nfold):
        test = S[S.fold_gene == f]; rest = S[S.fold_gene != f]
        v = val_groups(rest.gkey.to_numpy(), 0.2, f); train = rest[~v]
        trfx = {}
        for ln, g in train.groupby("line"):
            trfx[ln] = {L[ln]["V"][i].tobytes() for i in g.i}
        dup = np.array([L[ln]["V"][i].tobytes() in trfx.get(ln, set()) for ln, i in zip(test.line, test.i)])
        rel = np.full(len(test), np.nan)
        for ln, g in train.groupby("line"):
            sel = np.where(test.line.to_numpy() == ln)[0]
            if not len(sel):
                continue
            Vt = L[ln]["V"]; A_ = Vt[test.i.to_numpy()[sel]]; B_ = Vt[g.i.to_numpy()]; vw = L[ln]["vwt"]
            c = np.where((np.abs(A_ - vw).max(0) > 0) | (np.abs(B_ - vw).max(0) > 0))[0]
            dmin = np.abs(A_[:, None, c] - B_[None, :, c]).max(2).min(1)
            rel[sel] = dmin / np.abs(A_[:, c] - vw[c]).max(1)
        g_ = test.growth_ratio.to_numpy()
        out.append(dict(dataset=tag, fold=f, n_train=len(train), n_val=int(v.sum()), n_test=len(test), pos_rate_test=float((test.label == 1).mean()),
                        moma_lethal=int((g_ < 0.05).sum()), g_gt_0p99=int((g_ > 0.99).sum()), exact_dup=int(dup.sum()), dup_free=int((~dup).sum()),
                        rel_ge_0p2=int((rel >= 0.2).sum()), rel_ge_0p5=int((rel >= 0.5).sum()), rel_ge_0p2_and_lethal=int(((rel >= 0.2) & (g_ < 0.05)).sum()),
                        rel_lt_0p2=int((rel < 0.2).sum()), rel_nan=int(np.isnan(rel).sum())))
    return out


mo, lo, Lo, so = load(args.old); mn, ln_, Ln, Ln_smp = load(args.new)
assert mo["reactions"] == mn["reactions"] and lo == ln_, "scaffold index or cell-line set differ"
per = []
for ln in lo:
    a, b = Lo[ln], Ln[ln]
    same_genes = np.array_equal(a["genes"], b["genes"])
    pat = Ln_smp[(Ln_smp.line == ln) & (Ln_smp.patched)].gene.tolist() if "patched" in Ln_smp else []
    if same_genes:
        dV = np.abs(a["V_raw"] - b["V_raw"]).max(1); chg = np.where(dV > 0)[0]
        chg_genes = set(map(str, a["genes"][chg]))
    else:
        dV = np.array([np.nan]); chg = []; chg_genes = set()
    so_l = so[so.line == ln].set_index("gene"); sn_l = Ln_smp[Ln_smp.line == ln].set_index("gene")
    common = so_l.index.intersection(sn_l.index)
    gr_chg = common[(so_l.growth_ratio.reindex(common).fillna(-1) != sn_l.growth_ratio.reindex(common).fillna(-1))]
    per.append(dict(line=ln, vwt_identical=bool(np.array_equal(a["vwt_raw"], b["vwt_raw"])), present_identical=bool(np.array_equal(a["present"], b["present"])),
                    medium_identical=bool(np.array_equal(a["medium_lb"], b["medium_lb"])), mu_ref_identical=bool(float(a["mu_ref"]) == float(b["mu_ref"])),
                    genes_identical=same_genes, n_genes_old=len(a["genes"]), n_genes_new=len(b["genes"]), n_V_rows_changed=len(chg),
                    max_abs_dV_over_mu=float(dV.max()) if len(chg) else 0.0, n_patched_rows_samples=len(pat),
                    n_changed_rows_not_patched=len(chg_genes - set(pat)), n_patched_rows_unchanged=len(set(pat) & set(map(str, a["genes"])) - chg_genes),
                    n_samples_old=len(so_l), n_samples_new=len(sn_l), n_growth_ratio_changed=len(gr_chg), n_growth_ratio_changed_unpatched=len(set(gr_chg) - set(pat)),
                    lethal_old=int((so_l.growth_ratio < 0.05).sum()), lethal_new=int((sn_l.growth_ratio < 0.05).sum()),
                    label_identical=bool((so_l.label.reindex(common).to_numpy() == sn_l.label.reindex(common).to_numpy()).all())))
per = pd.DataFrame(per)
with open(f"{args.out}/dataset_diff_per_line.tsv", "w") as fh:
    fh.write(HDR); per.to_csv(fh, sep="\t", index=False)

So = sample_table(mo, lo, Lo, so); Sn = sample_table(mn, ln_, Ln, Ln_smp)
ko, kn = set(zip(So.line, So.gene)), set(zip(Sn.line, Sn.gene))
m = So.merge(Sn, on=["line", "gene"], how="outer", suffixes=("_old", "_new"), indicator=True)
both = m[m._merge == "both"]
chg = pd.concat([m[m._merge != "both"].assign(change=lambda d: d._merge.map({"left_only": "left (old only)", "right_only": "entered (new only)"})),
                 both[both.fold_gene_old != both.fold_gene_new].assign(change="gene fold changed")])
chg = chg.merge(Ln_smp[["line", "gene", "patched"]], on=["line", "gene"], how="left")
with open(f"{args.out}/sample_set_changes.tsv", "w") as fh:
    fh.write(HDR + "# rows: samples that enter or leave the fluxome-change sample set, or whose gene-disjoint test fold changes\n")
    chg[["line", "gene", "change", "patched", "growth_ratio_old", "growth_ratio_new", "fold_gene_old", "fold_gene_new", "label_old", "label_new"]].to_csv(fh, sep="\t", index=False)
st = pd.DataFrame(strata_per_fold(So, Lo, "old") + strata_per_fold(Sn, Ln, "new"))
with open(f"{args.out}/strata_per_fold.tsv", "w") as fh:
    fh.write(HDR + "# strata of the gene-disjoint test fold recomputed as 81b / 85 (rel = relative nearest-neighbor distance to the fold's training fluxomes of the same cell line)\n")
    st.to_csv(fh, sep="\t", index=False)
summ = dict(n_lines=len(lo), lines_all_inputs_identical_except_V=int((per.vwt_identical & per.present_identical & per.medium_identical & per.mu_ref_identical & per.genes_identical).sum()),
            n_V_rows_changed=int(per.n_V_rows_changed.sum()), n_patched_rows_samples=int(per.n_patched_rows_samples.sum()),
            n_changed_rows_not_patched=int(per.n_changed_rows_not_patched.sum()), n_patched_rows_unchanged=int(per.n_patched_rows_unchanged.sum()),
            n_growth_ratio_changed_unpatched=int(per.n_growth_ratio_changed_unpatched.sum()), labels_identical=bool(per.label_identical.all()),
            samples_tsv_rows_old=len(so), samples_tsv_rows_new=len(Ln_smp), lethal_old=int(per.lethal_old.sum()), lethal_new=int(per.lethal_new.sum()),
            gnn_samples_old=len(ko), gnn_samples_new=len(kn), gnn_samples_entered=len(kn - ko), gnn_samples_left=len(ko - kn),
            gnn_samples_gene_fold_changed=int((both.fold_gene_old != both.fold_gene_new).sum()), gnn_samples_line_fold_changed=int((both.fold_line_old != both.fold_line_new).sum()),
            gnn_lethal_old=int((So.growth_ratio < 0.05).sum()), gnn_lethal_new=int((Sn.growth_ratio < 0.05).sum()),
            gnn_patched_samples_new=int(Sn.merge(Ln_smp[["line", "gene", "patched"]], on=["line", "gene"]).patched.sum()) if "patched" in Ln_smp else 0,
            seconds=round(time.time() - t0))
with open(f"{args.out}/dataset_diff_summary.tsv", "w") as fh:
    fh.write(HDR); pd.DataFrame([summ]).to_csv(fh, sep="\t", index=False)
print(json.dumps(summ, indent=1)); print(st.to_string(index=False))
