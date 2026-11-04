"""Expression input for further external cell lines (2026-11-03; design result/external_plan_20261103/design_external_gi_normal_20261103.md
section 6(a); user rulings 2026-11-03: GI-tract cancer cell lines as the main external comparison, a few representative lines; normal lines as
the indirect test). Copy of script 76 with --lines / --out; extraction rule unchanged (CRC gene space, log2(TPM+1), asserted to reproduce the
CRC tINIT matrix). Medium: DepMap BROAD FormulationID -> existing assembly by the design's rule (scratchpad map_count.py, copied here).
Env: crcgem. Usage: 76b_external_expression.py --lines ACH-... --out data/depmap/<dir>
"""
import os, re, argparse
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--lines", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--assembly-override", nargs="*", default=[], help="ModelID=assembly_id proxy (user ruling 2026-11-03: BPH-1 on RPMI)"); args = ap.parse_args()
ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
OUT = args.out if args.out.startswith("/") else f"{ROOT}/{args.out}"; LINES = args.lines
if os.path.exists(f"{OUT}/lines.tsv"):
    raise SystemExit(f"exists, not overwritten: {OUT}/lines.tsv")
os.makedirs(OUT, exist_ok=True)
a = pd.read_parquet(f"{ROOT}/data/depmap/crc_trainset/crc_expression.ENSG.parquet")
hg = pd.read_csv(f"{ROOT}/data/scaffold/humangem_genes.tsv", sep="\t", dtype=str).set_index("genes")
key = pd.read_csv(f"{ROOT}/data/depmap/crc_trainset/crc_expression.genekey.tsv", sep="\t", dtype=str).set_index("symbol")
full = pd.read_csv(f"{ROOT}/data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
mp = pd.Series({e: hg.geneSymbols[e] for e in a.index}, name="symbol"); mp.index.name = "ENSG"
col = {e: f"{s} ({key.entrez[s]})" for e, s in mp.items()}
assert len(set(col.values())) == len(col), "ENSG -> column not one-to-one"
assert np.allclose(full.loc[list(a.columns), [col[e] for e in a.index]].T.to_numpy(), a.to_numpy(), atol=1e-6)
X = full.loc[LINES, [col[e] for e in a.index]].T; X.index = a.index
X.to_parquet(f"{OUT}/expression.ENSG.parquet")
mp.to_frame().assign(entrez=[key.entrez[x] for x in mp]).to_csv(f"{OUT}/ensg_map.tsv", sep="\t")
m = pd.read_csv(f"{ROOT}/data/depmap/Model.csv").set_index("ModelID").loc[LINES, ["CellLineName", "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype", "OnboardedMedia"]]
c = pd.read_csv(f"{ROOT}/data/medium/sources/depmap_24Q4/ModelCondition.csv")
c = c[(c.DataSource == "BROAD") & c.ModelID.isin(LINES)].drop_duplicates("ModelID").set_index("ModelID")
m["FormulationID"] = c.FormulationID
def rule(f):
    f = str(f)
    if re.match(r"^RPMI \+ 10% FBS \+ 2mM Glutamine \+ 25mM HEPES \+ 25mM Sodium bicarbonate$", f):
        return "RPMI1640_HEPES_Gibco22400+SUPP_L-glutamine_2mM+SUPP_HEPES_25mM+SUPP_NaHCO3_25mM"
    if re.match(r"^RPMI \+ \d+% FBS( \+ 2mM Glutamine)?$", f): return "RPMI1640_Gibco11875"
    if re.match(r"^DMEM \+ \d+% FBS( \+ 2mM [Gg]lutamine| \+ 1ug/ml Insulin)?$", f): return "DMEM_highglc_Gibco11965"
    if re.match(r"^DMEM:F12 \+ \d+% FBS$", f): return "DMEMF12_Gibco11320"
    return None
m["assembly_id"] = m.FormulationID.map(rule); m["assembly_proxy"] = False
for kv in args.assembly_override:
    k, v = kv.split("="); m.loc[k, "assembly_id"] = v; m.loc[k, "assembly_proxy"] = True
dep = pd.read_csv(f"{ROOT}/data/depmap/CRISPRGeneDependency.csv", usecols=[0]).iloc[:, 0]
m["has_dependency"] = m.index.isin(set(dep))
assert m.assembly_id.notna().all(), f"unmapped medium: {m[m.assembly_id.isna()].FormulationID.to_dict()}"
m.to_csv(f"{OUT}/lines.tsv", sep="\t")
print(m.to_string()); print(X.shape)
