"""GNN dataset builder for the solver-tolerance-patched MOMA roots (script 104; user decision 2026-10-30 13:18 KST, option A).
Scripts 64 / 64b / 64c stay unchanged (they read summary.tsv without '#' header lines; the patched summary.tsv has them).

Same per-cell-line construction as 64 / 64b / 64c: flux of WT (reference.parquet) and of every optimal knockout
(flux_matrix.parquet), divided by the cell line's WT biomass flux (reference_growth), placed on the scaffold reaction index
(absent reactions = 0, presence mask); genes ENSG -> symbol via struct_universe.tsv; labels DepMap CRISPRGeneDependency >= 0.5;
MOMA growth ratio of every screened gene in samples.tsv; medium vector = assembly lower bounds after the per-cell-line
requirement caps. As 64c: scaffold_meta.json / scaffold_S.npz are COPIED from --scaffold-from (md5 recorded), a cell line
without a DepMap row keeps dependency NaN / label -1 / has_label False, '#' condition lines, no-overwrite guard.
Additions: summary.tsv read with comment='#'; samples.tsv carries the patch columns (patched, feasibility_tol) when present;
build_condition.json records every cell line's tolpatch counts from meta.json.
Reproduction check of the code path (run before the first patched build): --moma-root result/moma_20261027 --out <scratch>
--lines ACH-000007 ACH-000009, then compare each npz key and the samples.tsv rows with result/gnn_20261028/dataset.
Usage (colorectal, default): python 64d_gnn_build_dataset_tolpatch.py --moma-root result/moma_20261030_tolpatch --out result/gnn_20261030_tolpatch/dataset
       (external validation set): python 64d_gnn_build_dataset_tolpatch.py --moma-root result/oob_20261030_tolpatch/moma
         --out result/external_20261030_tolpatch/dataset --line-map data/depmap/oob_20261028/oob_lines.tsv
         --overrides result/oob_20261028/line_requirement_caps/lower_bound_overrides.tsv --dependency depmap-csv
         --scaffold-from result/gnn_20261030_tolpatch/dataset
Env: crcgem_gnn (pandas + pyarrow, no cobra).
"""
import argparse, hashlib, json, os, shutil, time
import numpy as np, pandas as pd

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
ap = argparse.ArgumentParser()
ap.add_argument("--moma-root", default=f"{ROOT}/result/moma_20261030_tolpatch")
ap.add_argument("--out", default=f"{ROOT}/result/gnn_20261030_tolpatch/dataset")
ap.add_argument("--line-map", default=f"{ROOT}/data/medium/line_medium_map.tsv", help="tsv with ModelID and assembly_id")
ap.add_argument("--overrides", default=f"{ROOT}/data/medium/uptake_caps/line_requirement_caps/lower_bound_overrides.tsv")
ap.add_argument("--dependency", default=f"{ROOT}/data/depmap/crc_trainset/crc_dependency.parquet",
                help="parquet with symbol columns, or 'depmap-csv' (rows of data/depmap/CRISPRGeneDependency.csv mapped by crc_dependency.genekey.tsv)")
ap.add_argument("--scaffold-from", default=f"{ROOT}/result/gnn_20261028/dataset", help="dataset folder whose scaffold files are copied")
ap.add_argument("--lines", nargs="*", default=None, help="default: every done cell line under --moma-root")
ap.add_argument("--exclude", default="", help="tsv (# comments allowed) with columns line, gene, reason: knockouts removed from the GNN input (not in lines/<line>.npz; "
                "samples.tsv keeps the row with in_gnn False and excluded_reason). 2026-10-30 user ruling: the 14 knockouts whose patched re-solve had no optimum")
args = ap.parse_args()
EXC = pd.read_csv(args.exclude, sep="\t", comment="#") if args.exclude else pd.DataFrame(columns=["line", "gene", "reason"])
EXC_SET = {(l, g): r for l, g, r in zip(EXC.line, EXC.gene, EXC.reason)}
M, OUT = args.moma_root.rstrip("/"), args.out.rstrip("/")
M = M if M.startswith("/") else f"{ROOT}/{M}"; OUT = OUT if OUT.startswith("/") else f"{ROOT}/{OUT}"
assert not os.path.exists(f"{OUT}/samples.tsv"), f"{OUT} already holds a dataset (no overwrite)"
os.makedirs(f"{OUT}/lines", exist_ok=True)
t0 = time.time()


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


scaf_md5 = {}
for f in ("scaffold_meta.json", "scaffold_S.npz"):
    shutil.copyfile(f"{args.scaffold_from}/{f}", f"{OUT}/{f}"); scaf_md5[f] = md5(f"{OUT}/{f}")
    assert scaf_md5[f] == md5(f"{args.scaffold_from}/{f}")
meta = json.load(open(f"{OUT}/scaffold_meta.json"))
RID, EX = meta["reactions"], meta["exchanges"]; ridx = {r: i for i, r in enumerate(RID)}
print(f"scaffold copied from {args.scaffold_from}: {len(RID)} reactions, {len(EX)} exchanges, md5 {scaf_md5}", flush=True)

lm = pd.read_csv(args.line_map, sep="\t", comment="#").set_index("ModelID")
LB = pd.read_csv(f"{ROOT}/data/medium/uptake_caps/exchange_lower_bounds_by_assembly.tsv", sep="\t", comment="#")
try:
    OV = pd.read_csv(args.overrides, sep="\t", comment="#")
except pd.errors.EmptyDataError:   # 2026-11-03 (as 58:45-48): 62 writes only the comment line when no cell line needs an override
    OV = pd.DataFrame(columns=["line", "exchange_rxn_id", "lower_bound_new"])
st = pd.read_csv(f"{ROOT}/result/ko_screen_20261026/eval_20261027/struct_universe.tsv", sep="\t", index_col=0)
e2s = st.symbol.dropna().to_dict()

lines = args.lines or sorted(x for x in os.listdir(M) if x.startswith("ACH-") and os.path.exists(f"{M}/{x}/meta.json")
                             and json.load(open(f"{M}/{x}/meta.json")).get("status") == "done")
if args.dependency == "depmap-csv":
    key = pd.read_csv(f"{ROOT}/data/depmap/crc_trainset/crc_dependency.genekey.tsv", sep="\t", dtype=str)
    col2sym = {f"{s} ({e})": s for s, e in zip(key.symbol, key.entrez)}
    raw = pd.read_csv(f"{ROOT}/data/depmap/CRISPRGeneDependency.csv", index_col=0)
    assert set(col2sym) <= set(raw.columns), "gene key columns missing from CRISPRGeneDependency.csv"
    dep = raw.loc[[l for l in lines if l in raw.index], list(col2sym)].rename(columns=col2sym)
    dep.to_parquet(f"{OUT}/external_dependency.parquet")
    dep_src = f"{ROOT}/data/depmap/CRISPRGeneDependency.csv (md5 {md5(f'{ROOT}/data/depmap/CRISPRGeneDependency.csv')}), columns via crc_dependency.genekey.tsv"
else:
    dep = pd.read_parquet(args.dependency); dep_src = args.dependency
no_label = [l for l in lines if l not in dep.index]
print(f"{len(lines)} cell lines; dependency rows for {len(lines) - len(no_label)}; NO DepMap label: {no_label}", flush=True)

rows, patch_counts = [], {}
for line in lines:
    meta_l = json.load(open(f"{M}/{line}/meta.json"))
    mu_ref = meta_l["reference_growth"]
    patch_counts[line] = meta_l.get("tolpatch", {}).get("counts", "no tolpatch block (unpatched root)")
    ref = pd.read_parquet(f"{M}/{line}/reference.parquet")["reference"]
    fx = pd.read_parquet(f"{M}/{line}/flux_matrix.parquet")
    sm = pd.read_csv(f"{M}/{line}/summary.tsv", sep="\t", index_col=0, comment="#")
    assert set(fx.index) <= set(sm.index) and len(fx) == int((sm.status == "optimal").sum()), f"{line}: flux matrix rows vs optimal rows"
    cols = np.array([ridx[r] for r in ref.index])
    vwt = np.zeros(len(RID), np.float32); vwt[cols] = ref.to_numpy() / mu_ref
    present = np.zeros(len(RID), bool); present[cols] = True
    fcols = np.array([ridx[r] for r in fx.columns])
    genes_opt = [g for g in fx.index if e2s.get(g) in dep.columns and (line, g) not in EXC_SET]
    n_exc = sum(1 for g in fx.index if (line, g) in EXC_SET)
    V = np.zeros((len(genes_opt), len(RID)), np.float32)
    V[:, fcols] = fx.loc[genes_opt].to_numpy(np.float32) / mu_ref
    lb = LB[LB.assembly_id == lm.loc[line, "assembly_id"]].set_index("exchange_rxn_id").lower_bound.to_dict()
    assert lb, f"{line}: assembly {lm.loc[line, 'assembly_id']} not in the assembly table"
    lb.update(dict(zip(OV.exchange_rxn_id[OV.line == line], OV.lower_bound_new[OV.line == line])))
    med = np.array([lb.get(r, 0.0) for r in EX], np.float32)
    dsym = [e2s[g] for g in genes_opt]
    has_label = line in dep.index
    dvec = dep.loc[line, dsym].to_numpy(np.float32) if has_label else np.full(len(dsym), np.nan, np.float32)
    np.savez_compressed(f"{OUT}/lines/{line}.npz", vwt=vwt, present=present, V=V, genes=np.array(genes_opt),
                        symbols=np.array(dsym), dependency=dvec, medium_lb=med, mu_ref=mu_ref)
    gset = set(genes_opt)
    for g in sm.index:
        s_ = e2s.get(g)
        if s_ not in dep.columns:
            continue
        d_ = float(dep.loc[line, s_]) if has_label else float("nan")
        rows.append(dict(line=line, gene=g, symbol=s_, status=sm.status[g], growth_ratio=sm.growth_ratio[g],
                         in_gnn=g in gset, dependency=d_, label=int(d_ >= 0.5) if has_label else -1,
                         assembly=lm.loc[line, "assembly_id"], has_label=has_label,
                         patched=bool(sm.patched[g]) if "patched" in sm else False,
                         feasibility_tol=float(sm.feasibility_tol[g]) if "feasibility_tol" in sm else float("nan"),
                         excluded_reason=EXC_SET.get((line, g), "")))
    print(f"{line}: {len(genes_opt)} KO fluxomes, |v/mu| max {np.abs(V).max():.1f}, label {has_label}, patched rows {int(sm.patched.sum()) if 'patched' in sm else 0}, excluded {n_exc}, {time.time() - t0:.0f}s", flush=True)
smp = pd.DataFrame(rows)
cond = dict(script="script_python/64d_gnn_build_dataset_tolpatch.py", date=time.ctime(), moma_root=M, line_map=args.line_map, overrides=args.overrides,
            dependency_source=dep_src, scaffold_from=args.scaffold_from, scaffold_md5=scaf_md5,
            label_rule="DepMap CRISPRGeneDependency >= 0.5; label -1 and dependency NaN where the cell line has no DepMap row",
            lines=lines, lines_without_label=no_label, n_samples=int(len(smp)), n_in_gnn=int(smp.in_gnn.sum()), n_patched_rows=int(smp.patched.sum()),
            exclude_file=args.exclude, n_excluded_from_gnn=int((smp.excluded_reason != "").sum()), excluded=[dict(line=l, gene=g, reason=r) for (l, g), r in EXC_SET.items()],
            tolpatch_counts_per_line=patch_counts,
            note="in_gnn = optimal KO with a mapped DepMap gene; the fluxome-change filter (|v_KO - v_WT|_inf > 1e-6/mu_ref) is applied by the training, prediction and evaluation scripts, as in 81b")
json.dump(cond, open(f"{OUT}/build_condition.json", "w"), indent=1)
with open(f"{OUT}/samples.tsv", "w") as fh:
    fh.write(f"# {len(lines)} cell lines; label dependency >= 0.5 (label -1 = no DepMap row: {no_label}); script 64d; moma_root {M}; dependency {dep_src.split(' (')[0]}; {time.ctime()}\n")
    fh.write(f"# scaffold copied from {args.scaffold_from} md5 {scaf_md5}; medium map {args.line_map}; caps {args.overrides}; patched rows (script 104, FeasibilityTol 1e-8) {int(smp.patched.sum())} of {len(smp)}; "
             f"excluded from the GNN input (in_gnn False, excluded_reason): {int((smp.excluded_reason != '').sum())} from {args.exclude or 'none'}\n")
    smp.to_csv(fh, sep="\t", index=False)
lab = smp[smp.has_label]
print(f"samples {len(smp)}, in GNN {int(smp.in_gnn.sum())}, patched {int(smp.patched.sum())}, positive rate (labeled) {lab.label.mean():.3f}; done {time.time() - t0:.0f}s -> {OUT}")
