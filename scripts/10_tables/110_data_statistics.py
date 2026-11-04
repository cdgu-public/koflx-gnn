"""Data statistics of the final GNN dataset (user request 2026-10-30: per-fold counts of cell line-gene pairs, cell lines, genes; overall
cell lines, genes; data statistics tables and figures), for the gene-disjoint and the cell-line-disjoint splits.

Split code = the functions of script_python/81b_gnn_train_v2.py (group_folds, val_groups, cell-line permutation with seed 0, validation cell
lines with seed = fold) copied verbatim; the result is checked against a launched run's split.json (--check-split <fold dir>) when given.
Sample set = knockouts passing the 81b fluxome-change filter (|v_KO - v_WT|_inf > 1e-6 / mu_ref after the 1e-6 / mu_ref threshold).
Per (split, fold, part in train / validation / test): cell line-gene pairs (= knockouts), cell lines, genes, disabled-reaction groups
(scaffold GPR reaction sets, the gene-disjoint grouping unit), label-positive rate (DepMap CRISPRGeneDependency >= 0.5), MOMA-lethal
knockouts (relative growth rate < 0.05), relative nearest-neighbor distance strata of the validation and test parts against the training
part of the same fold (>= 0.2 / < 0.2; exact duplicates = distance 0; not defined for the training part itself), patched rows (script 104).
Overall: cell lines, genes, knockouts, per-cell-line knockout counts, per-gene cell-line counts, and the exclusion funnel from screened
knockouts to GNN samples (no DepMap symbol; non-optimal MOMA status; no fluxome change; excluded by ruling), for the colorectal dataset and
the external validation set of non-colorectal cell lines (no folds there: the models are the colorectal cell-line-disjoint fold models).
Tables (TSV + CSV, '#' header lines) -> <out-tables>/ (raw/ holds the per-sample tables). Figures (plotly; html + svg + png + jpeg; strip /
violin / scatter, no bar charts; panel data next to each panel) -> <out-figures>/. Primordial figures for redrawing by the user.
Usage: python 110_data_statistics.py --dataset result/gnn_20261030_tolpatch/dataset_v2 [--check-split <run>/fold0] [--external-dataset ...]
Env: crcgem (pandas, numpy, plotly 6.3, kaleido 0.2.1).
"""
import argparse, json, os, time
import numpy as np, pandas as pd

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=f"{ROOT}/result/gnn_20261030_tolpatch/dataset_v2")
ap.add_argument("--external-dataset", default=f"{ROOT}/result/external_20261030_tolpatch/dataset")
ap.add_argument("--moma-root", default=f"{ROOT}/result/moma_20261030_tolpatch")
ap.add_argument("--external-moma-root", default=f"{ROOT}/result/oob_20261030_tolpatch/moma")
ap.add_argument("--check-split", nargs="*", default=[], help="launched fold folders (split.json + config.json) to verify the reproduced split against")
ap.add_argument("--out-tables", default=f"{ROOT}/submission_packages/20261029/Tables/Table_data_statistics")
ap.add_argument("--out-figures", default=f"{ROOT}/submission_packages/20261029/Figures/Figure2_data_statistics", help="main Figure 2 panels a, b")
ap.add_argument("--out-figures-supp", default=f"{ROOT}/submission_packages/20261029/Figures/Supplementary/FigureS_data_statistics", help="supplementary panels A..M")
ap.add_argument("--nfold", type=int, default=5)
ap.add_argument("--check-only", action="store_true", help="only the split reproduction check (raw/split_reproduction_check.tsv); other tables untouched")
args = ap.parse_args()
D = args.dataset.rstrip("/"); t0 = time.time()
for p in (args.out_tables, f"{args.out_tables}/raw", args.out_figures):
    os.makedirs(p, exist_ok=True)
assert args.check_only or not os.path.exists(f"{args.out_tables}/fold_statistics.tsv"), "tables exist (no overwrite)"
DATE = time.strftime("%Y-%m-%d %H:%M %Z")
HDR = (f"# script_python/110_data_statistics.py; dataset {D}; external dataset {args.external_dataset}; sample set = 81b fluxome-change filter (|v_KO - v_WT|_inf > 1e-6/mu_ref); "
       f"label = DepMap CRISPRGeneDependency >= 0.5; MOMA-lethal = relative growth rate < 0.05; relative nearest-neighbor distance = Chebyshev distance to the nearest training "
       f"fluxome of the same cell line / own |v_KO - v_WT|_inf (81b rel_nn_dist); splits = 81b group_folds / val_groups (gene-disjoint) and seeded cell-line permutation (cell-line-disjoint), nfold {args.nfold}; date {DATE}\n")


def write(df, name, note="", out=args.out_tables):
    for ext, sep in (("tsv", "\t"), ("csv", ",")):
        with open(f"{out}/{name}.{ext}", "w") as fh:
            fh.write(HDR + f"# {note}; n rows {len(df)}\n"); df.to_csv(fh, sep=sep, index=False)


# ---------------- data, as 81b ----------------
def load(Dd):
    meta = json.load(open(f"{Dd}/scaffold_meta.json")); lines = sorted(f[:-4] for f in os.listdir(f"{Dd}/lines")); L = {}
    for ln in lines:
        z = np.load(f"{Dd}/lines/{ln}.npz"); d = {k: z[k] for k in z.files}; th = 1e-6 / float(d["mu_ref"]); d["th"] = th
        d["vwt"] = np.where(np.abs(d["vwt"]) > th, d["vwt"], 0).astype(np.float32); d["V"] = np.where(np.abs(d["V"]) > th, d["V"], 0).astype(np.float32); L[ln] = d
    smp = pd.read_csv(f"{Dd}/samples.tsv", sep="\t", comment="#")
    samples = [(ln, i) for ln in lines for i in range(len(L[ln]["genes"])) if np.abs(L[ln]["V"][i] - L[ln]["vwt"]).max() > L[ln]["th"]]
    S = pd.DataFrame(dict(line=[ln for ln, _ in samples], i=[i for _, i in samples]))
    S["gene"] = [str(L[ln]["genes"][i]) for ln, i in samples]; S["symbol"] = [str(L[ln]["symbols"][i]) for ln, i in samples]
    S["dependency"] = [float(L[ln]["dependency"][i]) for ln, i in samples]; S["label"] = (S.dependency >= 0.5).astype(int)
    S.loc[S.dependency.isna(), "label"] = -1
    S["gkey"] = [",".join(map(str, sorted(meta["gene_rxns"].get(g, [])))) for g in S.gene]
    S = S.merge(smp[["line", "gene", "status", "growth_ratio"] + [c for c in ("patched", "excluded_reason", "has_label") if c in smp]], on=["line", "gene"], how="left")
    S["moma_lethal"] = S.growth_ratio < 0.05
    return meta, lines, L, smp, S


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


def split_fold(S, lines, split, fold):
    allidx = np.arange(len(S)); gkey = S.gkey.to_numpy(); sline = S.line.to_numpy()
    if split == "gene":
        fo = group_folds(gkey, args.nfold); test, rest = allidx[fo == fold], allidx[fo != fold]
        v = val_groups(gkey[rest], 0.2, fold); return rest[~v], rest[v], test
    perm = list(np.random.default_rng(0).permutation(lines)); lf = {ln: k % args.nfold for k, ln in enumerate(perm)}
    test_lines = sorted(ln for ln in lines if lf[ln] == fold); rest_lines = sorted(ln for ln in lines if lf[ln] != fold)
    val_lines = sorted(np.random.default_rng(fold).choice(rest_lines, int(round(0.2 * len(rest_lines))), replace=False))
    test = allidx[np.isin(sline, test_lines)]; val = allidx[np.isin(sline, val_lines)]
    train = allidx[~np.isin(sline, test_lines + list(val_lines))]; return train, val, test


def rel_dist(S, L, ids, train):
    out = np.full(len(ids), np.nan); dup = np.zeros(len(ids), bool); ids = np.asarray(ids)
    for ln in S.line.unique():
        trl = S.i.to_numpy()[train][S.line.to_numpy()[train] == ln]; sel = np.where(S.line.to_numpy()[ids] == ln)[0]
        if not len(trl) or not len(sel):
            continue
        Vt = L[ln]["V"]; A_ = Vt[S.i.to_numpy()[ids[sel]]]; B_ = Vt[trl]; vw = L[ln]["vwt"]
        c = np.where((np.abs(A_ - vw).max(0) > 0) | (np.abs(B_ - vw).max(0) > 0))[0]
        dmin = np.abs(A_[:, None, c] - B_[None, :, c]).max(2).min(1)
        out[sel] = dmin / np.abs(A_[:, c] - vw[c]).max(1); dup[sel] = dmin == 0
    return out, dup


meta, lines, L, smp, S = load(D)
print(f"dataset {D}: {len(lines)} cell lines, {len(S)} samples, {S.gene.nunique()} genes, {S.gkey.nunique()} disabled-reaction groups; {time.time()-t0:.0f}s", flush=True)

# ---------------- per fold / part ----------------
rows, per_sample = [], []
for split in (() if args.check_only else ("gene", "line")):
    for fold in range(args.nfold):
        train, val, test = split_fold(S, lines, split, fold)
        rel = {"train": (np.full(len(train), np.nan), np.zeros(len(train), bool))}
        rel["validation"], rel["test"] = rel_dist(S, L, val, train), rel_dist(S, L, test, train)
        for part, ids in (("train", train), ("validation", val), ("test", test)):
            d = S.iloc[ids]; r, dup = rel[part]
            rows.append(dict(split={"gene": "gene-disjoint", "line": "cell-line-disjoint"}[split], fold=fold, part=part, cell_line_gene_pairs=len(d), cell_lines=d.line.nunique(),
                             genes=d.gene.nunique(), disabled_reaction_groups=d.gkey.nunique(), label_positive=int((d.label == 1).sum()), label_positive_rate=float((d.label == 1).mean()),
                             moma_lethal=int(d.moma_lethal.sum()), moma_lethal_rate=float(d.moma_lethal.mean()), patched_rows=int(d.patched.sum()) if "patched" in d else 0,
                             rel_ge_0p2=int((r >= 0.2).sum()) if part != "train" else -1, rel_lt_0p2=int((r < 0.2).sum()) if part != "train" else -1,
                             exact_duplicate_of_training=int(dup.sum()) if part != "train" else -1, rel_undefined=int(np.isnan(r).sum()) if part != "train" else -1,
                             median_knockouts_per_cell_line=float(d.groupby("line").size().median()), min_knockouts_per_cell_line=int(d.groupby("line").size().min()),
                             max_knockouts_per_cell_line=int(d.groupby("line").size().max())))
            ps = d[["line", "gene", "symbol", "label", "growth_ratio", "moma_lethal", "gkey"]].copy(); ps["split"] = split; ps["fold"] = fold; ps["part"] = part; ps["rel_nn_dist"] = r
            per_sample.append(ps)
        print(f"{split} fold {fold}: train {len(train)} val {len(val)} test {len(test)}; {time.time()-t0:.0f}s", flush=True)
F = pd.DataFrame(rows)
if not args.check_only:
    write(F, "fold_statistics", "one row per split x fold x part; -1 = not defined for the training part")
    PS = pd.concat(per_sample, ignore_index=True)
    write(PS, "raw/per_sample_fold_assignment", "every GNN sample with its fold and part per split, label, growth ratio, group key, relative nearest-neighbor distance (validation / test vs training)", out=args.out_tables)

# check against launched runs
checks = []
for fd in args.check_split:
    sj = json.load(open(f"{fd}/split.json")); cfg = json.load(open(f"{fd}/config.json"))
    train, val, test = split_fold(S, lines, cfg["split"], cfg["fold"])
    ok = (sj["n_train"], sj["n_val"], sj["n_test"]) == (len(train), len(val), len(test)) and sorted(sj["test_lines"]) == sorted(S.line.iloc[test].unique()) and \
         sorted(sj["val_lines"]) == sorted(S.line.iloc[val].unique()) and os.path.abspath(cfg.get("dataset", "")) == os.path.abspath(D)
    checks.append(dict(fold_dir=fd, split=cfg["split"], fold=cfg["fold"], run_n_train=sj["n_train"], run_n_val=sj["n_val"], run_n_test=sj["n_test"], here_n_train=len(train), here_n_val=len(val),
                       here_n_test=len(test), run_dataset=cfg.get("dataset"), match=ok))
    print(f"split check {fd}: {'MATCH' if ok else 'MISMATCH'}", flush=True)
if checks:
    write(pd.DataFrame(checks), "raw/split_reproduction_check", "reproduced split vs the launched run's split.json / config.json")
if args.check_only:
    raise SystemExit(0)

# ---------------- overall and funnel ----------------
def funnel(Dd, Mroot, smp_, S_, tag):
    metas = {x: json.load(open(f"{Mroot}/{x}/meta.json")) for x in sorted(os.listdir(Mroot)) if x.startswith("ACH-") and os.path.exists(f"{Mroot}/{x}/meta.json")}
    metas = {k: v for k, v in metas.items() if v.get("status") == "done"}
    n_screened = sum(v["n_screened"] for v in metas.values())
    n_mapped = len(smp_); n_optimal = int((smp_.status == "optimal").sum()); n_in_gnn_opt = int(smp_.in_gnn.sum())
    n_excl = int((smp_.excluded_reason != "").sum()) if "excluded_reason" in smp_ else 0
    n_pass = len(S_); labeled = S_[S_.label >= 0]
    steps = [("screened knockouts (knockout disables >= 1 reaction of the cell line's context-specific model)", n_screened, ""),
             ("with a DepMap gene (ENSG -> symbol via struct_universe.tsv, symbol in the DepMap dependency table)", n_mapped, f"removed {n_screened - n_mapped}: no DepMap symbol"),
             ("MOMA solution optimal", n_optimal, f"removed {n_mapped - n_optimal}: solver status not optimal (production run)"),
             ("kept for the GNN input (optimal, DepMap gene, not excluded by ruling)", n_in_gnn_opt, f"removed {n_optimal - n_in_gnn_opt}: excluded by user ruling 2026-10-30 (no optimum under the patch condition){'' if n_excl else ' (none)'}"),
             ("knockout changes the fluxome (|v_KO - v_WT|_inf > 1e-6/mu_ref) = GNN sample set", n_pass, f"removed {n_in_gnn_opt - n_pass}: fluxome equal to wild type within tolerance"),
             ("GNN samples with a DepMap label (metrics)", len(labeled), f"removed {n_pass - len(labeled)}: cell line without DepMap row")]
    fun = pd.DataFrame([dict(dataset=tag, step=k, n=v, exclusion=e) for k, v, e in steps])
    ov = dict(dataset=tag, cell_lines=len(metas), cell_lines_with_label=int(labeled.line.nunique()), genes_screened=int(smp_.gene.nunique()), genes_in_gnn=int(S_.gene.nunique()),
              disabled_reaction_groups=int(S_.gkey.nunique()), knockouts_screened_with_depmap_gene=n_mapped, gnn_samples=n_pass, label_positive=int((labeled.label == 1).sum()),
              label_positive_rate=float((labeled.label == 1).mean()), moma_lethal=int(S_.moma_lethal.sum()), moma_lethal_rate=float(S_.moma_lethal.mean()),
              patched_rows_in_gnn=int(S_.patched.sum()) if "patched" in S_ else 0, knockouts_per_cell_line_median=float(S_.groupby("line").size().median()),
              knockouts_per_cell_line_min=int(S_.groupby("line").size().min()), knockouts_per_cell_line_max=int(S_.groupby("line").size().max()),
              cell_lines_per_gene_median=float(S_.groupby("gene").size().median()), cell_lines_per_gene_min=int(S_.groupby("gene").size().min()), cell_lines_per_gene_max=int(S_.groupby("gene").size().max()),
              genes_in_all_cell_lines=int((S_.groupby("gene").line.nunique() == len(metas)).sum()), genes_in_one_cell_line=int((S_.groupby("gene").line.nunique() == 1).sum()))
    pl = S_.groupby("line").agg(knockouts=("gene", "size"), genes=("gene", "nunique"), label_positive=("label", lambda x: int((x == 1).sum())), moma_lethal=("moma_lethal", "sum"),
                                patched=("patched", "sum") if "patched" in S_ else ("gene", "size")).reset_index().rename(columns={"line": "cell_line"})
    pl["label_positive_rate"] = pl.label_positive / pl.knockouts; pl["assembly"] = pl.cell_line.map(smp_.drop_duplicates("line").set_index("line").assembly)
    pl["screened_knockouts"] = pl.cell_line.map({k: v["n_screened"] for k, v in metas.items()}); pl["dataset"] = tag
    pg = S_.groupby(["gene", "symbol"]).agg(cell_lines=("line", "nunique"), label_positive=("label", lambda x: int((x == 1).sum())), moma_lethal=("moma_lethal", "sum"),
                                            mean_dependency=("dependency", "mean")).reset_index(); pg["dataset"] = tag
    return fun, ov, pl, pg


fun_c, ov_c, pl_c, pg_c = funnel(D, args.moma_root, smp, S, "colorectal (59 cell lines, training and cross-validation)")
outs = [fun_c]; ovs = [ov_c]; pls = [pl_c]; pgs = [pg_c]
if args.external_dataset and os.path.exists(f"{args.external_dataset}/samples.tsv"):
    meta_e, lines_e, L_e, smp_e, S_e = load(args.external_dataset)
    fun_e, ov_e, pl_e, pg_e = funnel(args.external_dataset, args.external_moma_root, smp_e, S_e, "external validation set of non-colorectal cell lines (7 cell lines)")
    outs.append(fun_e); ovs.append(ov_e); pls.append(pl_e); pgs.append(pg_e)
write(pd.concat(outs, ignore_index=True), "exclusion_funnel", "knockouts kept at each step from the MOMA screen to the GNN sample set, with the reason for every exclusion")
write(pd.DataFrame(ovs), "overall_statistics", "dataset-level counts")
write(pd.concat(pls, ignore_index=True), "per_cell_line", "GNN samples per cell line")
write(pd.concat(pgs, ignore_index=True), "per_gene", "cell lines per gene among the GNN samples")

# ---------------- figures (plotly; strip / violin / scatter; primordial) ----------------
# 2026-11-02: one plot per panel file, project palette (module/crcgem/figstyle.py); builders shared with script 131 in module/crcgem/figpanels.py.
# --out-figures = main Figure 2 folder (panels a, b); the supplementary panels (A..M) go to --out-figures-supp. The old facetted panels A..G were retired
# (backup _backup/20261102_figures/Figures/Figure_data_statistics). Rebuilding the figures alone (without this script's heavy pass): script_python/131_build_figures.py --only fig2.
import sys  # noqa: E402
sys.path.insert(0, f"{ROOT}/module")
from crcgem import figpanels as fp  # noqa: E402

PL = pd.concat(pls, ignore_index=True); PG = pd.concat(pgs, ignore_index=True)
fp.data_statistics_panels(PL, PG, F, PS, args.out_figures, args.out_figures_supp, [HDR.lstrip("# ").rstrip()])
with open(f"{args.out_tables}/README.md", "w") as fh:
    fh.write(f"# Data statistics of the GNN dataset ({os.path.basename(D)}), {DATE}\n\nSource: {D} (script 64d from {args.moma_root}); external: {args.external_dataset}.\n"
             f"Built by script_python/110_data_statistics.py. Splits reproduced with the 81b code{' and checked against ' + ', '.join(args.check_split) if args.check_split else ''}.\n"
             f"Files: fold_statistics (per split x fold x part), exclusion_funnel, overall_statistics, per_cell_line, per_gene; raw/ per-sample fold assignment with relative nearest-neighbor distance.\n"
             f"Figures: {args.out_figures} (panel_A..G, html/svg/png/jpeg, *_data.tsv next to each panel). Figure and table numbers are not assigned.\n")
print(f"done {time.time()-t0:.0f}s -> {args.out_tables}, {args.out_figures}")
