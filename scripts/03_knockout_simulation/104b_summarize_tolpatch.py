"""Aggregate the per-cell-line solver-tolerance patch (script 104) of one root into two tables with '#' condition lines:
  <root>/patch_summary_per_line.tsv   one row per cell line: trace / control counts, call changes, FBA consistency, reference check, timings
  <root>/patch_summary_total.tsv      totals over the cell lines
Usage: python 104b_summarize_tolpatch.py [--root result/moma_20261030_tolpatch] [--expect 59]
Env: crcgem_gnn or crcgem (pandas only).
"""
import argparse, json, os, time
import pandas as pd

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=f"{ROOT}/result/moma_20261030_tolpatch")
ap.add_argument("--expect", type=int, default=59)
args = ap.parse_args()
R = args.root if args.root.startswith("/") else f"{ROOT}/{args.root}"
rows = []
for ln in sorted(x for x in os.listdir(R) if x.startswith("ACH-") and os.path.exists(f"{R}/{x}/summary.tsv")):
    m = json.load(open(f"{R}/{ln}/meta.json")); tp = m["tolpatch"]; c = dict(tp["counts"]); rc = tp.get("reference_check", {})
    c.pop("extra_params", None); c["trace_kept_shipped"] = json.dumps(c.get("trace_kept_shipped", {}))
    rows.append(dict(line=ln, n_screened=m["n_screened"], n_lethal_shipped=json.load(open(f"{tp['source']}/meta.json"))["n_lethal_0p05"], n_lethal_patched=m["n_lethal_0p05"],
                     **c, ref_max_abs_Sv=rc.get("shipped_ref_max_abs_Sv"), ref_max_bound_violation=rc.get("shipped_ref_max_bound_violation"),
                     pfba_feas_status=rc.get("pfba_feas_status"), pfba_feas_growth_minus_shipped=rc.get("pfba_feas_growth_minus_shipped"),
                     pfba_feas_n_rxn_abs_dv_gt_1e6=rc.get("pfba_feas_n_rxn_abs_dv_gt_1e6"), pfba_feas_band_n_changed_gt_1e8=rc.get("pfba_feas_band_n_changed_gt_1e8"),
                     feasibility_tol=tp["feasibility_tol"], extra_params=json.dumps(tp["counts"].get("extra_params", {})), seconds=tp["seconds"]))
df = pd.DataFrame(rows)
tp0 = json.load(open(f"{R}/{df.line[0]}/meta.json"))["tolpatch"]
hdr = (f"# solver-tolerance patch summary (script_python/104b_summarize_tolpatch.py over script_python/104_moma_tolpatch_line.py outputs); root {R}; {len(df)}/{args.expect} cell lines; "
       f"FeasibilityTol {tp0['feasibility_tol']} with extra params {tp0['counts'].get('extra_params')}, production-path screen params {tp0['production_path_params_screen']}; "
       f"trace = 0 < max|v_ref| over GPR-disabled reactions <= {tp0['trace_max']}; lethal = growth_ratio < 0.05; control = {tp0['n_control']} non-trace knockouts per cell line (seed {tp0['seed']}), not patched; "
       f"versions {tp0['versions']}; date {time.strftime('%Y-%m-%d %H:%M %Z')}\n")
with open(f"{R}/patch_summary_per_line.tsv", "w") as fh:
    fh.write(hdr); df.to_csv(fh, sep="\t", index=False)
num = df.select_dtypes("number")
tot = dict(n_lines=len(df), **{k: int(num[k].sum()) for k in num if k.startswith("n_")},
           control_max_abs_d_growth_ratio=float(df.control_max_abs_d_growth_ratio.max()), control_median_of_line_medians=float(df.control_median_abs_d_growth_ratio.median()),
           control_max_abs_dv=float(df.control_max_abs_dv.max()), ref_max_abs_Sv_max=float(df.ref_max_abs_Sv.max()), ref_max_bound_violation_max=float(df.ref_max_bound_violation.max()),
           pfba_feas_growth_minus_shipped_maxabs=float(df.pfba_feas_growth_minus_shipped.abs().max()), pfba_feas_nonoptimal=int((df.pfba_feas_status != "optimal").sum()),
           mean_seconds_per_line=round(float(df.seconds.mean())), lines_missing=args.expect - len(df))
with open(f"{R}/patch_summary_total.tsv", "w") as fh:
    fh.write(hdr); pd.DataFrame([tot]).to_csv(fh, sep="\t", index=False)
print(json.dumps(tot, indent=1))
