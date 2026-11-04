"""Checker for script 171 outputs (2026-11-04): row/sample invariants and per-model gate summary G0-G2. Computation checks only.
Reads <root>/<side>/<line>/{summary_s*.tsv, gates_ref_s*.tsv, flux_*.parquet, meta_s*.json}; only ranges with a meta file (finished)
are used, unfinished ranges are listed. Asserts: each sample index at most once per model; summary rows = samples x genes; flux rows =
optimal genes per sample; flux columns identical across samples. Writes <root>/gates_per_model.tsv (one row per model) and
<root>/gates_ref_all.tsv (all reference gate rows). Usage: 172_check_moma_gated.py <root> [--expect-normal 100]
"""
import argparse, glob, json, os, re
from datetime import datetime
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--expect-normal", type=int, default=100)
ap.add_argument("--tag", default="", help="suffix of the output file names (e.g. a time stamp for an interim check)")
a = ap.parse_args()
rows, gall, notes = [], [], []
for side in ("tumor", "normal"):
    for d in sorted(glob.glob(f"{a.root}/{side}/*/")):
        line = os.path.basename(d.rstrip("/"))
        metas = sorted(glob.glob(f"{d}/meta_s*.json"))
        tags = [re.sub(r"^meta_(.*)\.json$", r"\1", os.path.basename(f)) for f in metas]
        unfinished = sorted(set(re.sub(r"^summary_(.*)\.tsv$", r"\1", os.path.basename(f)) for f in glob.glob(f"{d}/summary_s*.tsv")) - set(tags))
        assert not any("__" in t for t in tags), f"{line}: no-overwrite duplicates {tags}"
        if not tags:
            notes.append(f"{line}: no finished range (unfinished {unfinished})"); continue
        S = pd.concat([pd.read_csv(f"{d}/summary_{t}.tsv", sep="\t", comment="#") for t in tags])
        G = pd.concat([pd.read_csv(f"{d}/gates_ref_{t}.tsv", sep="\t", comment="#") for t in tags])
        idx = sorted(G.sample_idx.unique()); ng = S.gene.nunique()
        assert len(G) == len(idx), f"{line}: a sample index appears twice"
        assert len(S) == len(idx) * ng and not S.duplicated(["sample_idx", "gene"]).any(), f"{line}: rows != samples x genes"
        cols = None
        for s in idx:
            opt = set(S[(S.sample_idx == s) & (S.status == "optimal")].gene)
            f = pd.read_parquet(f"{d}/flux_{s}.parquet")
            assert set(f.index) == opt, f"{line} s{s}: flux rows != optimal genes"
            assert cols is None or list(f.columns) == cols or len(f) == 0; cols = cols or (list(f.columns) if len(f) else None)
            assert (f.dtypes == np.float64).all(), f"{line} s{s}: flux dtype not float64"
        ko = S[~S.status.isin(["not_in_model", "no_reaction_disabled", "reference_gate_failed", "reference_pfba_failed"])]
        o = ko[ko.status == "optimal"]
        G.insert(0, "model", line); G.insert(1, "side", side); gall.append(G)
        pf = json.load(open(metas[0])).get("pfba", {})
        rows.append(dict(model=line, side=side, n_references=len(idx), n_references_expected=1 if side == "tumor" else a.expect_normal,
                         unfinished_ranges=";".join(unfinished) or "none", pfba_status=pf.get("pfba_status", "na"),
                         max_abs_diff_vs_58_reference=pf.get("max_abs_diff_vs_58_reference", np.nan),
                         G0_pass=int(G.G0_pass.astype(str).eq("True").sum()), G0_max_bound_violation=G.max_bound_violation.max(),
                         G0_max_abs_Sv=G.max_abs_Sv.max(), G1_pass=int(G.G1_pass.astype(str).eq("True").sum()),
                         G1_both_pass=int(G.G1_both_pass.astype(str).eq("True").sum()), G1_max_objective_kept=G.G1_objective_kept.max(),
                         n_genes=ng, n_not_in_model=int((S.status == "not_in_model").sum()),
                         n_no_reaction_disabled=int((S.status == "no_reaction_disabled").sum()),
                         n_reference_gate_failed=int(S.status.isin(["reference_gate_failed", "reference_pfba_failed"]).sum()),
                         n_knockout_solves=len(ko), n_optimal_kept=len(o), n_no_optimal=len(ko) - len(o),
                         n_kept_m0=int((o.kept_method == 0).sum()), n_kept_m1=int((o.kept_method == 1).sum()),
                         n_status_m0_not_optimal=int((ko.status_m0 != "optimal").sum()), n_status_m1_not_optimal=int((ko.status_m1 != "optimal").sum()),
                         max_abs_obj_diff=o.abs_obj_diff.max(), n_abs_obj_diff_gt_1e12=int((o.abs_obj_diff > 1e-12).sum()),
                         G2_max_abs_Sv_kept=o.max_abs_Sv_kept.max(), G2_n_Sv_kept_gt_1e9=int((o.max_abs_Sv_kept > 1e-9).sum()),
                         G2_max_bound_violation_kept=o.max_bound_violation_kept.max(),
                         G2_n_bound_violation_kept_gt_1e9=int((o.max_bound_violation_kept > 1e-9).sum())))
R = pd.DataFrame(rows)
hdr = (f"# source: {a.root} (script_python/171_moma_gated.py outputs; finished ranges = those with meta_s*.json)\n"
       f"# checker: script_python/172_check_moma_gated.py; date {datetime.now().isoformat(timespec='seconds')} KST; n models {len(R)}\n"
       "# G0: max bound violation and max|S v| of the reference <= 1e-9; G1: no-knockout MOMA objective (kept) <= 1e-12, both = both methods;"
       " G2: both methods per knockout, kept = lower objective; counts of kept solutions with max|S v_KO| or bound violation > 1e-9\n"
       f"# notes: {' | '.join(notes) or 'none'}\n")
for nm, df in ((f"gates_per_model{a.tag}.tsv", R), (f"gates_ref_all{a.tag}.tsv", pd.concat(gall) if gall else pd.DataFrame())):
    p = f"{a.root}/{nm}"
    if os.path.exists(p):
        p = p.replace(".tsv", f"__{datetime.now():%Y%m%d_%H%M%S}.tsv")
    with open(p, "w") as fh:
        fh.write(hdr); df.to_csv(fh, sep="\t", index=False, float_format="%.6g")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print(R.T.to_string()); print("\n".join(notes))
