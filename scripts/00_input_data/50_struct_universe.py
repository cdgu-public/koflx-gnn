"""Gene universe for the KO screen evaluation (rebuild of the lost 10-19 struct_universe.tsv).

Logic copied from result/ko_screen_20261026/recovered_1019_scripts/ceiling.py (universe part only):
model genes in the expression panel whose symbol is a DepMap column with no NaN over the 59 lines;
struct_candidate = single-gene GPR knockout disables >= 1 reaction. No LP.
Env: crcgem (cobra). Output: result/ko_screen_20261026/eval_20261027/struct_universe.tsv
"""
import os, warnings
warnings.filterwarnings("ignore")
import pandas as pd, cobra

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
OUT = f"{ROOT}/result/ko_screen_20261026/eval_20261027"
os.makedirs(OUT, exist_ok=True)
m = cobra.io.read_sbml_model(f"{ROOT}/data/scaffold/Human-GEM.xml")
expr = pd.read_parquet(f"{ROOT}/data/depmap/crc_trainset/crc_expression.ENSG.parquet")
dep = pd.read_parquet(f"{ROOT}/data/depmap/crc_trainset/crc_dependency.parquet")
hg = pd.read_csv(f"{ROOT}/data/scaffold/humangem_genes.tsv", sep="\t")
hg.columns = [c.strip('"').strip() for c in hg.columns]
for c in hg.columns:
    hg[c] = hg[c].astype(str).str.strip('"')
ens_col = [c for c in hg.columns if "genes" == c or "ENSG" in c.upper() or c.lower() in ("genes", "gene")][0]
sym_col = [c for c in hg.columns if "symbol" in c.lower() or "Name" in c][0]
print("using", ens_col, sym_col)
e2s = dict(zip(hg[ens_col], hg[sym_col]))

rows = []
for g in m.genes:
    hit = 0
    for r in g.reactions:
        try:
            if r.gpr.eval({g.id}) is False:
                hit += 1
        except Exception:
            pass
    rows.append((g.id, len(g.reactions), hit))
st = pd.DataFrame(rows, columns=["ensg", "n_rxn", "n_disabled"]).set_index("ensg")
st["struct_candidate"] = st.n_disabled > 0
st["symbol"] = [e2s.get(i, None) for i in st.index]
u = st.loc[st.index.intersection(expr.index)].copy()
u = u[u.symbol.isin(dep.columns)]
D = dep[u.symbol.tolist()]
u["n_pos"] = (D >= 0.5).sum(axis=0).values
u["complete"] = (~D.isna().any(axis=0)).values
uc = u[u.complete].copy()
print("model genes", len(st), "struct candidates", int(st.struct_candidate.sum()))
print("universe (expr n DepMap)", len(u), "complete", len(uc), "unique symbols", uc.symbol.nunique(),
      "struct candidates in universe", int(uc.struct_candidate.sum()))
uc.to_csv(f"{OUT}/struct_universe.tsv", sep="\t")
print("wrote", f"{OUT}/struct_universe.tsv")
