"""MOMA under a common medium (sensitivity analysis; documentation/terminology.md row "MOMA under a common medium") compared with MOMA in
each cell line's own medium, both after the solver-tolerance patch (script 104) and the non-optimal exclusion rule (dataset_v2 rule).

Pre-specified readout (result/gnn_20261028/discussion_20261029/review_gem.md section 9.2.4 and 9.2.6, user approval status.md 2026-10-29
13:40 / 15:00 "both media in the paper, no selection by test performance"): report the common-medium run (a) on the INTERSECTION of its GNN
sample set with the own-medium sample set (paired comparison) and (b) on its OWN sample set, never mixed; the readouts are the change of
the MOMA-lethal list (Jaccard), the MOMA AUROC against the DepMap label, and the stratum sizes; the common-medium run is a control of the
generator, not a better model (the DepMap label was measured in each cell line's own medium).
Descriptive comparisons written here (per cell line and overall; no model, no selection):
  sample sets         GNN samples (81b fluxome-change filter) own / common, intersection, own-only, common-only
  MOMA-lethal calls   on the intersection: both lethal, own-only lethal, common-only lethal, neither; concordance and Jaccard of the lethal sets;
                      on each own set: MOMA-lethal count and rate
  MOMA AUROC          score 1 - relative growth rate (clipped to [0, 1]) vs DepMap label (>= 0.5), per cell line (>= 20 samples, both classes)
                      and pooled, on the intersection and on each own set; paired per-cell-line difference common - own on the intersection
  fluxome change      per cell line: wild-type fluxome |v_WT,common - v_WT,own|_inf (after the 1e-6/mu threshold), number of reactions
                      whose WT flux changes by > 1e-6/mu; per intersection sample: |v_KO,common - v_KO,own|_inf and the knockout's own
                      |v_KO - v_WT|_inf under each medium (median, 90th percentile per cell line)
  medium              own assembly per cell line, caps raised and script-62 gap-fill per medium (meta.json), WT growth per medium
Output: <out>/{sample_sets_per_cell_line, lethal_calls_per_cell_line, auroc_per_cell_line, fluxome_change_per_cell_line, overall,
raw/intersection_samples}.tsv (+ .csv) with '#' header lines. Nothing is overwritten.
Usage: python 111_compare_common_medium.py --own result/gnn_20261030_tolpatch/dataset_v2 --common result/gnn_fixed_DMEMF12_Gibco11320_20261030_tolpatch/dataset
       --common-tag DMEMF12_Gibco11320 [--common2 ... --common2-tag RPMI1640_Gibco11875] --out result/common_medium_20261030
Env: crcgem_gnn (pandas, numpy, scikit-learn).
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
ap = argparse.ArgumentParser()
ap.add_argument("--own", default=f"{ROOT}/result/gnn_20261030_tolpatch/dataset_v2")
ap.add_argument("--own-moma", default=f"{ROOT}/result/moma_20261030_tolpatch")
ap.add_argument("--common", nargs="+", required=True, help="common-medium dataset folders")
ap.add_argument("--common-tag", nargs="+", required=True)
ap.add_argument("--common-moma", nargs="+", required=True, help="patched common-medium MOMA roots (meta.json)")
ap.add_argument("--out", default=f"{ROOT}/result/common_medium_20261030")
args = ap.parse_args()
assert len(args.common) == len(args.common_tag) == len(args.common_moma)
os.makedirs(f"{args.out}/raw", exist_ok=True); t0 = time.time()
assert not os.path.exists(f"{args.out}/overall.tsv"), f"{args.out} exists (no overwrite)"
HDR = (f"# script_python/111_compare_common_medium.py; own medium {args.own}; common medium {dict(zip(args.common_tag, args.common))}; sample set = 81b fluxome-change filter; "
       f"MOMA-lethal = relative growth rate < 0.05; label = DepMap CRISPRGeneDependency >= 0.5; AUROC score = 1 - clipped relative growth rate; per-cell-line AUROC needs >= 20 samples with both classes; "
       f"plan review_gem.md 9.2.4 / 9.2.6 (intersection and own sets never mixed); date {time.strftime('%Y-%m-%d %H:%M %Z')}\n")


def write(df, name, note):
    for ext, sep in (("tsv", "\t"), ("csv", ",")):
        with open(f"{args.out}/{name}.{ext}", "w") as fh:
            fh.write(HDR + f"# {note}; n rows {len(df)}\n"); df.to_csv(fh, sep=sep, index=False)


def load(Dd):
    lines = sorted(f[:-4] for f in os.listdir(f"{Dd}/lines")); L = {}
    for ln in lines:
        z = np.load(f"{Dd}/lines/{ln}.npz"); d = {k: z[k] for k in z.files}; th = 1e-6 / float(d["mu_ref"]); d["th"] = th
        d["vwt"] = np.where(np.abs(d["vwt"]) > th, d["vwt"], 0).astype(np.float32); d["V"] = np.where(np.abs(d["V"]) > th, d["V"], 0).astype(np.float32); L[ln] = d
    smp = pd.read_csv(f"{Dd}/samples.tsv", sep="\t", comment="#")
    rows = []
    for ln in lines:
        d = L[ln]; dv = np.abs(d["V"] - d["vwt"]).max(1); keep = dv > d["th"]
        for i in np.where(keep)[0]:
            rows.append((ln, str(d["genes"][i]), i, float(dv[i])))
    S = pd.DataFrame(rows, columns=["line", "gene", "i", "dv_inf"]).merge(smp[["line", "gene", "symbol", "dependency", "label", "growth_ratio", "status"]], on=["line", "gene"], how="left")
    S["lethal"] = S.growth_ratio < 0.05; S["score"] = 1 - S.growth_ratio.clip(0, 1)
    return lines, L, smp, S


def auc(y, s):
    y = np.asarray(y) == 1; return float(roc_auc_score(y, s)) if len(y) >= 20 and 0 < y.sum() < len(y) else np.nan


lines, Lo, smp_o, So = load(args.own)
meta_o = {ln: json.load(open(f"{args.own_moma}/{ln}/meta.json")) for ln in lines}
sets, calls, aucs, flux, overall, inter_rows = [], [], [], [], [], []
for Dc, tag, Mc in zip(args.common, args.common_tag, args.common_moma):
    lines_c, Lc, smp_c, Sc = load(Dc); assert lines_c == lines, "cell-line sets differ"
    meta_c = {ln: json.load(open(f"{Mc}/{ln}/meta.json")) for ln in lines}
    ko, kc = set(zip(So.line, So.gene)), set(zip(Sc.line, Sc.gene)); inter = ko & kc
    M = So.merge(Sc, on=["line", "gene"], suffixes=("_own", "_common")); M["medium"] = tag
    assert len(M) == len(inter)
    M["dv_ko_common_vs_own_inf"] = [float(np.abs(Lc[l]["V"][ic] - Lo[l]["V"][io]).max()) for l, io, ic in zip(M.line, M.i_own, M.i_common)]
    inter_rows.append(M)
    for ln in lines:
        o = So[So.line == ln]; c = Sc[Sc.line == ln]; m = M[M.line == ln]
        so, sc = set(o.gene), set(c.gene)
        sets.append(dict(medium=tag, cell_line=ln, own_assembly=meta_o[ln]["assembly"], screened=meta_o[ln]["n_screened"], own_optimal=meta_o[ln]["n_optimal"], common_optimal=meta_c[ln]["n_optimal"],
                         own_gnn_samples=len(o), common_gnn_samples=len(c), intersection=len(so & sc), own_only=len(so - sc), common_only=len(sc - so),
                         own_wt_growth=meta_o[ln]["wt_growth"], common_wt_growth=meta_c[ln]["wt_growth"], own_caps_raised=len(meta_o[ln].get("caps_raised", {})), common_caps_raised=len(meta_c[ln].get("caps_raised", {})),
                         own_gapfill62=len(meta_o[ln].get("gapfill_script62", [])), common_gapfill62=len(meta_c[ln].get("gapfill_script62", [])),
                         own_patched_trace=meta_o[ln]["tolpatch"]["counts"]["n_trace"], common_patched_trace=meta_c[ln]["tolpatch"]["counts"]["n_trace"]))
        lo, lc = m.lethal_own.to_numpy(bool), m.lethal_common.to_numpy(bool)
        calls.append(dict(medium=tag, cell_line=ln, intersection=len(m), both_lethal=int((lo & lc).sum()), own_only_lethal=int((lo & ~lc).sum()), common_only_lethal=int((~lo & lc).sum()), neither=int((~lo & ~lc).sum()),
                          concordance=float((lo == lc).mean()) if len(m) else np.nan, jaccard_lethal=float((lo & lc).sum() / (lo | lc).sum()) if (lo | lc).sum() else np.nan,
                          own_set_lethal=int(o.lethal.sum()), own_set_lethal_rate=float(o.lethal.mean()), common_set_lethal=int(c.lethal.sum()), common_set_lethal_rate=float(c.lethal.mean()),
                          own_set_positive_rate=float((o.label == 1).mean()), common_set_positive_rate=float((c.label == 1).mean())))
        a_io, a_ic = auc(m.label_own, m.score_own), auc(m.label_own, m.score_common)
        aucs.append(dict(medium=tag, cell_line=ln, n_intersection=len(m), positives_intersection=int((m.label_own == 1).sum()), auroc_own_on_intersection=a_io, auroc_common_on_intersection=a_ic,
                         diff_common_minus_own=a_ic - a_io, n_own_set=len(o), auroc_own_set=auc(o.label, o.score), n_common_set=len(c), auroc_common_set=auc(c.label, c.score)))
        vwo, vwc = Lo[ln]["vwt"], Lc[ln]["vwt"]; th = max(Lo[ln]["th"], Lc[ln]["th"])
        flux.append(dict(medium=tag, cell_line=ln, mu_ref_own=float(Lo[ln]["mu_ref"]), mu_ref_common=float(Lc[ln]["mu_ref"]), wt_dv_inf=float(np.abs(vwc - vwo).max()), wt_n_reactions_changed=int((np.abs(vwc - vwo) > th).sum()),
                         wt_n_active_own=int((vwo != 0).sum()), wt_n_active_common=int((vwc != 0).sum()),
                         ko_dv_common_vs_own_median=float(m.dv_ko_common_vs_own_inf.median()) if len(m) else np.nan, ko_dv_common_vs_own_p90=float(m.dv_ko_common_vs_own_inf.quantile(0.9)) if len(m) else np.nan,
                         ko_own_dv_inf_median=float(m.dv_inf_own.median()) if len(m) else np.nan, ko_common_dv_inf_median=float(m.dv_inf_common.median()) if len(m) else np.nan,
                         medium_lb_n_changed=int((Lc[ln]["medium_lb"] != Lo[ln]["medium_lb"]).sum())))
    lo, lc = M.lethal_own.to_numpy(bool), M.lethal_common.to_numpy(bool)
    A = pd.DataFrame([a for a in aucs if a["medium"] == tag])
    overall.append(dict(medium=tag, cell_lines=len(lines), own_gnn_samples=len(ko), common_gnn_samples=len(kc), intersection=len(inter), own_only=len(ko - kc), common_only=len(kc - ko),
                        intersection_both_lethal=int((lo & lc).sum()), intersection_own_only_lethal=int((lo & ~lc).sum()), intersection_common_only_lethal=int((~lo & lc).sum()),
                        intersection_concordance=float((lo == lc).mean()), intersection_jaccard_lethal=float((lo & lc).sum() / (lo | lc).sum()),
                        own_set_lethal_rate=float(So.lethal.mean()), common_set_lethal_rate=float(Sc.lethal.mean()),
                        pooled_auroc_own_on_intersection=auc(M.label_own, M.score_own), pooled_auroc_common_on_intersection=auc(M.label_own, M.score_common),
                        pooled_auroc_own_set=auc(So.label, So.score), pooled_auroc_common_set=auc(Sc.label, Sc.score),
                        per_cell_line_auroc_diff_median=float(A.diff_common_minus_own.median()), per_cell_line_auroc_diff_min=float(A.diff_common_minus_own.min()), per_cell_line_auroc_diff_max=float(A.diff_common_minus_own.max()),
                        cell_lines_common_auroc_higher=int((A.diff_common_minus_own > 0).sum()), cell_lines_common_auroc_lower=int((A.diff_common_minus_own < 0).sum()), cell_lines_auroc_defined=int(A.diff_common_minus_own.notna().sum()),
                        cell_lines_in_own_medium_already=int(sum(meta_o[ln]["assembly"] == tag for ln in lines)),
                        wt_dv_inf_median=float(np.median([f["wt_dv_inf"] for f in flux if f["medium"] == tag])), ko_dv_common_vs_own_median=float(M.dv_ko_common_vs_own_inf.median())))
    print(f"{tag}: own {len(ko)} common {len(kc)} intersection {len(inter)}; concordance {overall[-1]['intersection_concordance']:.3f} Jaccard {overall[-1]['intersection_jaccard_lethal']:.3f}; "
          f"pooled AUROC own {overall[-1]['pooled_auroc_own_on_intersection']:.3f} common {overall[-1]['pooled_auroc_common_on_intersection']:.3f}; {time.time()-t0:.0f}s", flush=True)
write(pd.DataFrame(sets), "sample_sets_per_cell_line", "GNN sample sets under the own medium and the common medium per cell line, intersection and differences, WT growth, caps and gap-fill per medium")
write(pd.DataFrame(calls), "lethal_calls_per_cell_line", "MOMA-lethal calls on the intersection (paired) and on each own set")
write(pd.DataFrame(aucs), "auroc_per_cell_line", "MOMA AUROC vs DepMap label per cell line on the intersection (paired) and on each own set; NaN when < 20 samples or one class")
write(pd.DataFrame(flux), "fluxome_change_per_cell_line", "WT fluxome and knockout fluxome differences between media per cell line (fluxes divided by the WT biomass flux, 1e-6/mu threshold)")
write(pd.DataFrame(overall), "overall", "pooled and per-cell-line summary per common medium")
write(pd.concat(inter_rows, ignore_index=True).drop(columns=["i_own", "i_common"]), "raw/intersection_samples", "every intersection sample with growth ratio, lethal call, score and fluxome distance under both media")
print(f"done {time.time()-t0:.0f}s -> {args.out}")
