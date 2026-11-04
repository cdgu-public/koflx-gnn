"""Expression input for the out-of-bag / POC lines (user 2026-10-28): DepMap 24Q4 lines with a Replogle 2022
Perturb-seq counterpart [ref: doi: 10.1016/j.cell.2022.05.013]: K-562 (ACH-000551) and the hTERT RPE-1-derived
DepMap lines RPE1-ss* (ACH-002462..002467). Same gene space and units as the CRC tINIT input
(data/depmap/crc_trainset/crc_expression.ENSG.parquet: 2,830 Human-GEM ENSG x lines, log2(TPM+1), from
OmicsExpressionProteinCodingGenesTPMLogp1). ENSG -> DepMap column by the Human-GEM symbol and the CRC gene key; asserted to
reproduce the CRC matrix exactly, so the new lines use the CRC gene definitions.
tINIT thresholds stay those of the 59 CRC lines (script 22 --threshold-expression), so the extraction rule is
the trained pipeline unchanged.
Env: crcgem. Output: data/depmap/oob_20261028/{oob_expression.ENSG.parquet, oob_lines.tsv, ensg_map.tsv}
"""
import os
import numpy as np, pandas as pd
ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
OUT = f"{ROOT}/data/depmap/oob_20261028"; os.makedirs(OUT, exist_ok=True)
LINES = ["ACH-000551", "ACH-002462", "ACH-002463", "ACH-002464", "ACH-002465", "ACH-002466", "ACH-002467"]
a = pd.read_parquet(f"{ROOT}/data/depmap/crc_trainset/crc_expression.ENSG.parquet")
hg = pd.read_csv(f"{ROOT}/data/scaffold/humangem_genes.tsv", sep="\t", dtype=str).set_index("genes")
key = pd.read_csv(f"{ROOT}/data/depmap/crc_trainset/crc_expression.genekey.tsv", sep="\t", dtype=str).set_index("symbol")
full = pd.read_csv(f"{ROOT}/data/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
# ENSG -> Human-GEM symbol -> DepMap column "SYMBOL (Entrez)" of the CRC gene key (one-to-one asserted). The first
# version matched value vectors over the CRC lines, which collides for identical (all-zero) vectors (review 2026-10-28)
mp = pd.Series({e: hg.geneSymbols[e] for e in a.index}, name="symbol"); mp.index.name = "ENSG"
col = {e: f"{s} ({key.entrez[s]})" for e, s in mp.items()}
assert len(set(col.values())) == len(col), "ENSG -> column not one-to-one"
# check: the same extraction reproduces the CRC tINIT matrix for all genes and lines
assert np.allclose(full.loc[list(a.columns), [col[e] for e in a.index]].T.to_numpy(), a.to_numpy(), atol=1e-6)
X = full.loc[LINES, [col[e] for e in a.index]].T; X.index = a.index
X.to_parquet(f"{OUT}/oob_expression.ENSG.parquet")
mp.to_frame().assign(entrez=[key.entrez[x] for x in mp]).to_csv(f"{OUT}/ensg_map.tsv", sep="\t")
m = pd.read_csv(f"{ROOT}/data/depmap/Model.csv").set_index("ModelID").loc[LINES, ["CellLineName", "OncotreeLineage", "OncotreePrimaryDisease", "OnboardedMedia"]]
c = pd.read_csv(f"{ROOT}/data/medium/sources/depmap_24Q4/ModelCondition.csv")
c = c[(c.DataSource == "BROAD") & c.ModelID.isin(LINES)].set_index("ModelID")
m["FormulationID"] = c.FormulationID
# assemblies already built for the CRC lines: IMDM + 10% FBS -> the IMDM assembly (the +2 mM glutamine
# supplement does not change the exchange set, and the glutamine cap is Jain-measured, not concentration-scaled);
# DMEM + 10% FBS -> DMEM high glucose (Gibco 11965), the DepMap default for DMEM.
m["assembly_id"] = m.FormulationID.map({"IMDM + 10% FBS": "IMDM_Gibco12440+SUPP_L-glutamine_2mM",
                                        "DMEM + 10% FBS": "DMEM_highglc_Gibco11965"})
dep = pd.read_csv(f"{ROOT}/data/depmap/CRISPRGeneDependency.csv", usecols=[0]).iloc[:, 0]
m["has_dependency"] = m.index.isin(set(dep))
m.to_csv(f"{OUT}/oob_lines.tsv", sep="\t")
print(m.to_string()); print(X.shape, X.describe().loc[["mean", "50%"]].round(2).to_string())
