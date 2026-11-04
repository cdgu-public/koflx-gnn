"""Expression input for the patient POC (tumor / matched adjacent normal), 2026-11-03.
Spec: result/external_plan_20261103/design_poc_addendum_20261103.md section 2 (user rulings 2026-11-03: Q2 proxy medium
RPMI1640_Gibco11875 for tumor and normal; Q3 K = 3 by the barcode rule).
Patient rule (pre-specified, no expression used): patients (12-character barcode) with both a 01 (primary tumor) and an 11 (solid
tissue normal) sample; drop patients with more than one 01 or more than one 11 sample; sort by barcode; take the first K.
Expression: Xena GDC star_tpm (log2(TPM+1)); strip ENSG version; drop _PAR_Y rows; 2^x - 1; restrict to the ENSG index of the
cell-line matrix (--gene-space); rescale each column to sum 1e6 (TPM re-closed on the cell-line gene set); log2(x + 1).
Check P3 (unit, reported either way): median over shared genes of each tissue column vs the cell-line column medians.
Outputs (--out): expression.ENSG.parquet (ENSG x sample barcode), lines.tsv (76b column layout, read by 77b `tail -n +2` and by 62
without comment="#", so it carries no # lines), selection.tsv (# header: source, rule, n), p3_unit_check.tsv (# header).
Env: crcgem. Never overwrites (refuses if lines.tsv exists).
"""
import argparse, os
from datetime import datetime
import numpy as np, pandas as pd

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
ap = argparse.ArgumentParser()
ap.add_argument("--xena", required=True); ap.add_argument("--k", type=int, required=True)
ap.add_argument("--assembly", required=True); ap.add_argument("--gene-space", required=True); ap.add_argument("--out", required=True)
# unit rule, design_poc_unit_cut_20261103.md Decision 1 (user ruling 2026-11-03 ~08:00: target sum 1e6, not 946,401)
ap.add_argument("--unit", choices=["pc_universe", "geneset_1e6"], default="pc_universe",
                help="pc_universe: re-close over U (Xena genes 1:1 by symbol to the DepMap protein-coding columns), then subset; "
                     "geneset_1e6: the withdrawn 2026-11-03 05:0x rule (re-close on the 2,830 genes)")
ap.add_argument("--depmap-pc", default="data/depmap/crc_trainset/crc_expression.parquet",
                help="59 lines x 19,193 DepMap protein-coding symbols (log2 TPM+1); symbols define U, sums feed C3")
ap.add_argument("--probemap", default="data/tcga_xena_20261103/gencode.v36.annotation.gtf.gene.probemap")
ap.add_argument("--pc-target", type=float, default=1e6)
args = ap.parse_args()
ab = lambda p: p if p.startswith("/") else f"{ROOT}/{p}"
XD, OUT = ab(args.xena), ab(args.out)
if os.path.exists(f"{OUT}/lines.tsv"):
    raise SystemExit(f"exists, not overwritten: {OUT}/lines.tsv")
os.makedirs(OUT, exist_ok=True)

# ---- patient selection from the column names only ----
cols = {}
for proj in ("COAD", "READ"):
    h = pd.read_csv(f"{XD}/TCGA-{proj}.star_tpm.tsv.gz", sep="\t", nrows=0).columns[1:]
    for c in h:
        cols.setdefault(c, proj)
smp = pd.DataFrame({"sample": list(cols), "project": list(cols.values())})
smp["patient"] = smp["sample"].str[:12]; smp["type"] = smp["sample"].str[13:15]
cnt = smp.groupby(["patient", "type"]).size().unstack(fill_value=0)
paired = cnt[(cnt.get("01", 0) >= 1) & (cnt.get("11", 0) >= 1)]
single = paired[(paired["01"] == 1) & (paired["11"] == 1)]
chosen = sorted(single.index)[: args.k]
sel = smp[smp.patient.isin(chosen) & smp.type.isin(["01", "11"])].sort_values(["patient", "type"]).reset_index(drop=True)
assert len(sel) == 2 * args.k, sel
print(f"samples {len(smp)}, paired patients {len(paired)}, single-sample paired {len(single)}, chosen {chosen}")

# ---- expression ----
gs = pd.read_parquet(ab(args.gene_space))
X = []
for proj in ("COAD", "READ"):
    want = sel[sel.project == proj]["sample"].tolist()
    if not want:
        continue
    d = pd.read_csv(f"{XD}/TCGA-{proj}.star_tpm.tsv.gz", sep="\t", usecols=["Ensembl_ID"] + want, index_col=0)
    X.append(d)
X = pd.concat(X, axis=1)[sel["sample"].tolist()]
n_rows = len(X); X = X[~X.index.str.endswith("_PAR_Y")]; n_par = n_rows - len(X)
X.index = X.index.str.replace(r"\.\d+$", "", regex=True)
assert X.index.is_unique, "ENSG not unique after version strip"
missing = gs.index.difference(X.index)
# 2026-11-03 run: ENSG00000281500 is absent from the Xena GENCODE annotation. Kept as NaN (not measured): 22 gpr_score drops NaN
# in AND/OR exactly as for a gene absent from the matrix; the re-closure sums over the measured genes only.
print(f"cell-line genes absent from Xena (kept as NaN): {list(missing)}")
Tall = 2.0 ** X - 1.0
raw_sum = Tall.reindex(gs.index).sum()                       # tissue TPM sum on the 2,830 genes (delivered unit)
E_geneset = np.log2(Tall.reindex(gs.index) / raw_sum * 1e6 + 1.0)   # withdrawn rule (kept for C4 only)
# ---- universe U (Decision 1a): Xena ENSG whose GENCODE v36 symbol matches exactly one DepMap protein-coding symbol, one-to-one ----
pc = pd.read_parquet(ab(args.depmap_pc))                    # 59 lines x 19,193 symbols, log2(TPM+1)
pm = pd.read_csv(ab(args.probemap), sep="\t", usecols=["id", "gene"])
pm = pm[~pm.id.str.endswith("_PAR_Y")].assign(ensg=lambda d: d.id.str.replace(r"\.\d+$", "", regex=True))
assert pm.ensg.is_unique and set(X.index) <= set(pm.ensg)
nsym = pm.groupby("gene").ensg.nunique()
ambiguous = sorted(set(nsym[nsym > 1].index) & set(pc.columns))
pmu = pm[pm.gene.isin(set(pc.columns) - set(ambiguous))]
# coordinator ruling 2026-11-03: an ambiguous symbol whose ENSG is fixed by the 2,830-gene space enters U through that exact ENSG
# (PDE11A -> ENSG00000128655); the other ambiguous symbols stay out of U
amb_exact = pm[pm.gene.isin(ambiguous) & pm.ensg.isin(gs.index)]
pmu = pd.concat([pmu, amb_exact])
ambiguous_out = sorted(set(ambiguous) - set(amb_exact.gene))
U = pmu.ensg.tolist(); u_sym = dict(zip(pmu.ensg, pmu.gene))
print("C2 note: symbol ambiguous in probemap, matched by exact ENSG:", dict(zip(amb_exact.gene, amb_exact.ensg)))
unmatched = sorted(set(pc.columns) - set(pmu.gene) - set(ambiguous))
assert len(set(u_sym.values())) == len(U), "U not one-to-one"
outside = [e for e in gs.index if e not in set(U) and e not in set(missing)]
assert not outside, f"C2: 2,830-gene ENSG outside U (stop, do not drop): {outside}"
if args.unit == "pc_universe":
    TU = Tall.loc[U]; frac_in_U = TU.sum() / Tall.sum()
    TU = TU / TU.sum() * args.pc_target
    E = np.log2(TU.reindex(gs.index) + 1.0)
    # C1: sum over U of (2**E_U - 1) == pc_target
    assert np.allclose((2.0 ** np.log2(TU + 1.0) - 1.0).sum(), args.pc_target), "C1 failed"
else:
    E = E_geneset; frac_in_U = Tall.loc[U].sum() / Tall.sum()
# C2: 2,830 x 6, the only all-NaN row is the absent gene(s)
assert E.shape == (len(gs), 2 * args.k) and list(E.index[E.isna().all(axis=1)]) == list(missing) and not E.drop(index=missing).isna().any().any(), "C2 failed"
E.to_parquet(f"{OUT}/expression.ENSG.parquet")
# C3 (reported, no threshold): cell-line share of the protein-coding TPM sum carried by U symbols; per-line log2 scalar
Tpc = 2.0 ** pc - 1.0; usyms = sorted(set(u_sym.values()))
c3 = pd.DataFrame({"pc_sum": Tpc.sum(1), "frac_in_U": Tpc[usyms].sum(1) / Tpc.sum(1)})
c3["log2_scalar_to_target"] = np.log2(args.pc_target / Tpc[usyms].sum(1))
gsym = [u_sym[e] for e in gs.index if e in u_sym]
rec = np.log2(Tpc[gsym].div(Tpc[usyms].sum(1), axis=0) * args.pc_target + 1.0)      # step (b) applied to the cell lines
c3["max_abs_dev_from_scalar_shift"] = (rec - pc[gsym]).sub((rec - pc[gsym]).median(1), axis=0).where(pc[gsym] > 1).abs().max(1)

# ---- lines.tsv (76b column layout) ----
L = pd.DataFrame({"ModelID": sel["sample"], "CellLineName": sel["sample"], "OncotreeLineage": "Bowel",
                  "OncotreePrimaryDisease": "TCGA-" + sel["project"],
                  "OncotreeSubtype": sel["type"].map({"01": "Primary Tumor", "11": "Solid Tissue Normal"}),
                  "OnboardedMedia": "", "FormulationID": "tissue (proxy medium, user ruling 2026-11-03 Q2)",
                  "assembly_id": args.assembly, "has_dependency": False}).set_index("ModelID")
L.to_csv(f"{OUT}/lines.tsv", sep="\t")
assert list(pd.read_csv(f"{OUT}/lines.tsv", sep="\t").columns)[7] == "assembly_id"   # 77b reads column 8

now = datetime.now().isoformat(timespec="seconds")
HDR = [f"# source: {XD}/TCGA-{{COAD,READ}}.star_tpm.tsv.gz (UCSC Xena GDC hub, log2(TPM+1), see SOURCE.txt); script_python/76c_tcga_expression.py",
       f"# rule: patients with >=1 sample type 01 and >=1 type 11; drop >1 01 or >1 11; sort by 12-character barcode; first K={args.k}",
       f"# counts: samples {len(smp)}, paired patients {len(paired)}, paired with one 01 and one 11 {len(single)}",
       f"# expression: version stripped, {n_par} _PAR_Y rows dropped, restricted to {len(gs)} ENSG of {args.gene_space}, "
       + (f"re-closed over U to sum {args.pc_target:.0f} then subset (design_poc_unit_cut_20261103.md Decision 1; target 1e6 by user ruling "
        f"2026-11-03 ~08:00), log2(x+1)" if args.unit == "pc_universe" else "re-closed to sum 1e6 on the gene set (withdrawn rule), log2(x+1)")
       + f"; assembly {args.assembly} (proxy)",
       f"# universe U: {len(U)} Xena ENSG matched 1:1 by GENCODE v36 symbol ({args.probemap}) to {pc.shape[1]} DepMap protein-coding symbols "
       f"({args.depmap_pc}); unmatched DepMap symbols {len(unmatched)}; ambiguous symbols (>1 ENSG) {len(ambiguous)}: {ambiguous[:30]}; "
       f"of these, symbol ambiguous in probemap, matched by exact ENSG (in U): {dict(zip(amb_exact.gene, amb_exact.ensg))}; kept out of U: {ambiguous_out}",
       f"# per-sample fraction of Xena TPM inside U: {frac_in_U.round(4).to_dict()}; per-sample TPM sum on the {len(gs)} genes before re-closure: {raw_sum.round(0).to_dict()}",
       f"# exclusions: absent from Xena, kept as NaN (not measured): {list(missing)}", f"# date: {now} KST, n = {len(sel)} samples"]
with open(f"{OUT}/selection.tsv", "w") as f:
    f.write("\n".join(HDR) + "\n")
    sel.assign(xena_tpm_sum_on_gene_set=raw_sum.values).to_csv(f, sep="\t", index=False)

# ---- P3 unit check (reported either way) ----
cl_med = gs.median()
p3 = pd.DataFrame({"sample": E.columns, "median_delivered_xena": X.reindex(gs.index).median().values,
                   "median_geneset_reclosure_withdrawn": E_geneset.median().values, "median_used": E.median().values,
                   "cellline_median_of_column_medians": float(cl_med.median())})
for c in ("median_delivered_xena", "median_geneset_reclosure_withdrawn", "median_used"):
    p3[f"diff_{c}"] = p3[c] - p3.cellline_median_of_column_medians
    p3[f"within_0p5_{c}"] = p3[f"diff_{c}"].abs() <= 0.5
with open(f"{OUT}/c3_cellline_coverage.tsv", "w") as f:
    f.write("\n".join(HDR + ["# C3: per cell line, share of the DepMap protein-coding TPM sum carried by U symbols; log2 scalar of step (b); "
                             "max |deviation| from a single per-line shift over genes with log2(TPM+1) > 1 (reported, no threshold)"]) + "\n")
    c3.to_csv(f, sep="\t")
print("C3 frac_in_U min/median/max", c3.frac_in_U.min().round(4), c3.frac_in_U.median().round(4), c3.frac_in_U.max().round(4),
      "| log2 scalar", c3.log2_scalar_to_target.min().round(3), c3.log2_scalar_to_target.median().round(3), c3.log2_scalar_to_target.max().round(3),
      "| max dev", float(c3.max_abs_dev_from_scalar_shift.max()))
with open(f"{OUT}/p3_unit_check.tsv", "w") as f:
    f.write("\n".join(HDR + [f"# P3: |median over {len(gs)} shared genes of tissue column - median of the 59 cell-line column medians| <= 0.5 log2 "
                             f"(design section 2 falsifier); cell-line column medians min {cl_med.min():.3f} max {cl_med.max():.3f}; "
                             f"cell-line TPM sum on this gene set median {float((2.0**gs-1).sum().median()):.0f}"]) + "\n")
    p3.to_csv(f, sep="\t", index=False)
print(sel.to_string()); print(p3.to_string()); print("C1 C2 passed (asserted); C4/P3 used unit:", "within 0.5" if p3.within_0p5_median_used.all() else "exceeds 0.5 (reported)")
