"""Within-model percentile of each drug-target gene among the control genes of the same model (2026-11-04, user ruling via coordinator
07:3x). Readouts: d_all (141 definition, sqrt(moma_objective)/||v_ref||, recomputed here in float64 from the 171 kept-solution objective
and the reference, then checked against the 141 overall_distance.tsv d_all at its printed precision) and d_sym per KEGG map (group_kind
kegg_map) from the 141 pathway_distance.tsv.gz (both routes; printed at %.6g by 141, so d_sym ties include ties at 6 significant digits).
Percentile: values = [target] + controls with a finite value in that model and reference; scipy.stats.rankdata(method='average');
percentile = 100 * (rank_target - 1) / n_controls (= 100 * (#controls below + 0.5 * #controls equal) / n_controls).
Tumor: one reference. Normal: per OptGP sample, then the median over samples (min/max also given).
Ties: n_controls_tied = controls with a value equal to the target (exact equality on the values used); tied control genes listed (tumor),
for normals the number of samples with >= 1 tie and the union of tied controls.
Gene roles: controls = genes/controls.tsv in_cut10 True; drug targets = the other genes of genes/poc_genes_cut10.txt (symbols from
data/scaffold/humangem_genes.tsv). Usage: 174_percentile_gated.py --moma-tumor T --moma-normal N --dist-tumor DT --dist-normal DN --out F
"""
import argparse, glob, os
from datetime import datetime
import numpy as np, pandas as pd, scipy
from scipy.stats import rankdata

R = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"; P = f"{R}/result/patient_poc_20261103"
ap = argparse.ArgumentParser()
for k in ("moma-tumor", "moma-normal", "dist-tumor", "dist-normal", "out"):
    ap.add_argument(f"--{k}", required=True)
a = ap.parse_args()
genes = [l.strip() for l in open(f"{P}/genes/poc_genes_cut10.txt") if l.strip() and not l.startswith("#")]
C = pd.read_csv(f"{P}/genes/controls.tsv", sep="\t", comment="#"); ctrl = list(C[C.in_cut10.astype(str) == "True"].ensg)
assert set(ctrl) <= set(genes); targets = [g for g in genes if g not in ctrl]
hg = pd.read_csv(f"{R}/data/scaffold/humangem_genes.tsv", sep="\t")
sym = dict(zip(hg.iloc[:, 0], hg["geneSymbols"])) if "geneSymbols" in hg.columns else {}


def pct(x, cv):
    cv = cv[np.isfinite(cv)]
    if not np.isfinite(x) or len(cv) == 0:
        return np.nan, len(cv), 0
    r = rankdata(np.r_[x, cv], method="average")[0]
    return 100.0 * (r - 1) / len(cv), len(cv), int((cv == x).sum())


rows = []
for side, mroot, droot in (("tumor", a.moma_tumor, a.dist_tumor), ("normal", a.moma_normal, a.dist_normal)):
    for dd in sorted(glob.glob(f"{droot}/*/")):
        model = os.path.basename(dd.rstrip("/"))
        summ = pd.read_csv(f"{mroot}/{model}/summary.tsv", sep="\t", comment="#")
        if "reference_id" not in summ.columns:
            summ["reference_id"] = "reference"
        ref = pd.read_parquet(f"{mroot}/{model}/reference.parquet")
        nr = {c: float(np.linalg.norm(ref[c].to_numpy(np.float64))) for c in ref.columns}
        obj = summ.moma_objective.to_numpy(np.float64)
        obj_c = np.where(obj >= 0, obj, np.where(obj >= -1e-9, 0.0, np.nan))   # 141 rule for slightly negative QP objectives
        summ["d_all"] = np.where(summ.status == "optimal", np.sqrt(obj_c) / summ.reference_id.map(nr).to_numpy(), np.nan)
        ov = pd.read_csv(f"{dd}/overall_distance.tsv", sep="\t", comment="#")
        chk = summ.merge(ov[["reference_id", "gene", "d_all"]], on=["reference_id", "gene"], suffixes=("", "_141"))
        assert len(chk) == len(summ)
        ok = chk.d_all.notna()
        assert np.allclose(chk.d_all[ok], chk.d_all_141[ok], rtol=1e-5, atol=0), f"{model}: d_all != 141 output"
        assert (chk.d_all.isna() == chk.d_all_141.isna()).all()
        pw = pd.read_csv(f"{dd}/pathway_distance.tsv.gz", sep="\t", comment="#",
                         usecols=["reference_id", "gene", "route", "group_kind", "group_id", "group_name", "d_sym"])
        pw = pw[pw.group_kind == "kegg_map"]
        readouts = [("d_all", "", "", "", summ.pivot(index="reference_id", columns="gene", values="d_all"))]
        for (route, gid, gname), sub in pw.groupby(["route", "group_id", "group_name"]):
            readouts.append(("d_sym", route, gid, gname, sub.pivot(index="reference_id", columns="gene", values="d_sym")))
        for rd, route, gid, gname, W in readouts:
            W = W.reindex(index=sorted(summ.reference_id.unique()), columns=genes)   # genes absent / non-optimal -> NaN
            for t in targets:
                res = []
                for rid in W.index:
                    p, nc, nt = pct(W.at[rid, t], W.loc[rid, ctrl].to_numpy(np.float64))
                    tied = [c for c in ctrl if np.isfinite(W.at[rid, t]) and W.at[rid, c] == W.at[rid, t]]
                    res.append((p, nc, nt, tied, W.at[rid, t]))
                ps = np.array([r[0] for r in res], dtype=float); fin = np.isfinite(ps)
                rows.append(dict(model=model, side=side, gene=t, symbol=sym.get(t, ""), readout=rd, route=route, kegg_map=gid,
                                 kegg_map_name=gname, n_references=len(res), n_references_with_percentile=int(fin.sum()),
                                 percentile=float(np.median(ps[fin])) if fin.any() else np.nan,
                                 percentile_min=float(ps[fin].min()) if fin.any() else np.nan,
                                 percentile_max=float(ps[fin].max()) if fin.any() else np.nan,
                                 value_target_median=float(np.nanmedian([r[4] for r in res])) if fin.any() else np.nan,
                                 n_controls_used_median=float(np.median([r[1] for r in res])),
                                 n_references_with_tie=int(sum(r[2] > 0 for r in res)),
                                 n_controls_tied_max=int(max(r[2] for r in res)),
                                 tied_controls=";".join(sorted({c for r in res for c in r[3]})) or "none"))
O = pd.DataFrame(rows)
out = a.out
if os.path.exists(out):
    out = out.replace(".tsv", f"__{datetime.now():%Y%m%d_%H%M%S}.tsv")
with open(out, "w") as fh:
    fh.write(f"# source: 171 kept solutions laid out by 173: {a.moma_tumor}, {a.moma_normal}; 141 outputs {a.dist_tumor}, {a.dist_normal}\n"
             "# rule: within-model percentile of each drug-target gene among the control genes (genes/controls.tsv in_cut10) of the same model "
             "and reference; scipy.stats.rankdata method 'average'; percentile = 100*(rank_target-1)/n_controls; controls with NaN "
             "(not in model, no reaction disabled, non-optimal; d_sym NA when no active reaction in the map) excluded per reference\n"
             "# d_all = sqrt(moma_objective)/||v_ref|| in float64 from 171 objectives (checked vs 141 d_all, rtol 1e-5); d_sym from 141 "
             "pathway_distance (printed %.6g; ties at that precision count as ties); kegg_map groups only, both routes (main route by ruling: "
             "kegg_reaction_or_ec)\n"
             "# normal: per OptGP sample, then median (min/max) over samples; tumor: one pFBA reference; tie columns: n_references_with_tie, "
             "n_controls_tied_max, tied_controls (union over references)\n"
             f"# controls (n={len(ctrl)}): {';'.join(ctrl)}; drug targets (n={len(targets)}): {';'.join(targets)}\n"
             f"# script_python/174_percentile_gated.py; numpy {np.__version__}, pandas {pd.__version__}, scipy {scipy.__version__}; "
             f"date {datetime.now().isoformat(timespec='seconds')} KST; n rows {len(O)}\n")
    O.to_csv(fh, sep="\t", index=False, float_format="%.6g")
print(out, len(O)); print(O[O.readout == "d_all"].to_string())
