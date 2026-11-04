"""Copy of 64_gnn_build_dataset.py (unchanged) with env GNN_MOMA_ROOT / GNN_DATASET_OUT path overrides and a no-overwrite guard (2026-10-29, fixed-medium MOMA runs).
GNN dataset from the MOMA knockout fluxomes (script 58, result/moma_20261027/<line>/).

Per line (done lines only; re-runnable on a partial set):
  flux of WT (reference.parquet) and of every optimal KO (flux_matrix.parquet), both divided by the
  line's WT biomass flux (reference growth; scope section 5), placed on the scaffold reaction index
  (reactions absent from the context model = 0, with a presence mask);
  labels: DepMap CRISPRGeneDependency of the gene in that line (>= 0.5 positive), genes mapped
  ENSG -> symbol through struct_universe.tsv (script 50); unmapped genes dropped;
  MOMA growth ratio of every screened gene (incl. non-optimal KOs, ratio NaN) for the baseline;
  medium vector = the line's exchange lower bounds after the per-line requirement caps (script 62).
Shared: scaffold stoichiometry (sparse), reaction / metabolite ids, exchange ids, currency metabolite
ids (data/graph/currency_metabolite_ids.txt, Stelzer 2011 Table A1), gene -> disabled reactions
(scaffold GPR).
Env: crcgem. Output: result/gnn_20261028/dataset/{scaffold.npz, scaffold_meta.json, lines/<line>.npz, samples.tsv}
"""
import json, os, sys, time
import numpy as np, pandas as pd, cobra, scipy.sparse as sp
from cobra.util.array import create_stoichiometric_matrix

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
M = os.environ.get("GNN_MOMA_ROOT", f"{ROOT}/result/moma_20261027")   # 64b (2026-10-29): path override for the fixed-medium runs; default = script 64
OUT = os.environ.get("GNN_DATASET_OUT", f"{ROOT}/result/gnn_20261028/dataset")
assert not os.path.exists(f"{OUT}/samples.tsv"), f"{OUT} already holds a dataset; choose another GNN_DATASET_OUT (no overwrite)"
os.makedirs(f"{OUT}/lines", exist_ok=True)
t0 = time.time()

scaf = cobra.io.read_sbml_model(f"{ROOT}/data/scaffold/Human-GEM.xml")
RID = [r.id for r in scaf.reactions]; MID = [m.id for m in scaf.metabolites]
ridx = {r: i for i, r in enumerate(RID)}
S = sp.csr_matrix(create_stoichiometric_matrix(scaf, array_type="dok").tocsr())
EX = sorted(r.id for r in scaf.exchanges)
CUR = [x.strip() for x in open(f"{ROOT}/data/graph/currency_metabolite_ids.txt") if x.strip()]
gene_rxns = {g.id: [ridx[r.id] for r in g.reactions if r.gpr.eval({g.id}) is False] for g in scaf.genes}
sp.save_npz(f"{OUT}/scaffold_S.npz", S)
json.dump(dict(reactions=RID, metabolites=MID, exchanges=EX, currency=CUR, gene_rxns=gene_rxns,
               biomass="MAR13082"), open(f"{OUT}/scaffold_meta.json", "w"))
print(f"scaffold {S.shape}, {len(EX)} exchanges, {len(CUR)} currency ids, {time.time() - t0:.0f}s", flush=True)

lm = pd.read_csv(f"{ROOT}/data/medium/line_medium_map.tsv", sep="\t").set_index("ModelID")
LB = pd.read_csv(f"{ROOT}/data/medium/uptake_caps/exchange_lower_bounds_by_assembly.tsv", sep="\t", comment="#")
OV = pd.read_csv(f"{ROOT}/data/medium/uptake_caps/line_requirement_caps/lower_bound_overrides.tsv", sep="\t", comment="#")
st = pd.read_csv(f"{ROOT}/result/ko_screen_20261026/eval_20261027/struct_universe.tsv", sep="\t", index_col=0)
e2s = st.symbol.dropna().to_dict()
dep = pd.read_parquet(f"{ROOT}/data/depmap/crc_trainset/crc_dependency.parquet")

lines = sys.argv[1:] or sorted(x for x in os.listdir(M) if x.startswith("ACH-")
                               and os.path.exists(f"{M}/{x}/meta.json")
                               and json.load(open(f"{M}/{x}/meta.json")).get("status") == "done")
rows = []
for line in lines:
    meta = json.load(open(f"{M}/{line}/meta.json"))
    mu_ref = meta["reference_growth"]
    ref = pd.read_parquet(f"{M}/{line}/reference.parquet")["reference"]
    fx = pd.read_parquet(f"{M}/{line}/flux_matrix.parquet")
    sm = pd.read_csv(f"{M}/{line}/summary.tsv", sep="\t", index_col=0)
    cols = np.array([ridx[r] for r in ref.index])
    vwt = np.zeros(len(RID), np.float32); vwt[cols] = ref.to_numpy() / mu_ref
    present = np.zeros(len(RID), bool); present[cols] = True
    fcols = np.array([ridx[r] for r in fx.columns])
    genes_opt = [g for g in fx.index if e2s.get(g) in dep.columns]
    V = np.zeros((len(genes_opt), len(RID)), np.float32)
    V[:, fcols] = fx.loc[genes_opt].to_numpy(np.float32) / mu_ref
    lb = LB[LB.assembly_id == lm.loc[line, "assembly_id"]].set_index("exchange_rxn_id").lower_bound.to_dict()
    lb.update(dict(zip(OV.exchange_rxn_id[OV.line == line], OV.lower_bound_new[OV.line == line])))
    med = np.array([lb.get(r, 0.0) for r in EX], np.float32)
    dsym = [e2s[g] for g in genes_opt]
    np.savez_compressed(f"{OUT}/lines/{line}.npz", vwt=vwt, present=present, V=V, genes=np.array(genes_opt),
                        symbols=np.array(dsym), dependency=dep.loc[line, dsym].to_numpy(np.float32),
                        medium_lb=med, mu_ref=mu_ref)
    for g in sm.index:
        s_ = e2s.get(g)
        if s_ not in dep.columns:
            continue
        rows.append(dict(line=line, gene=g, symbol=s_, status=sm.status[g], growth_ratio=sm.growth_ratio[g],
                         in_gnn=g in set(genes_opt), dependency=float(dep.loc[line, s_]),
                         label=int(dep.loc[line, s_] >= 0.5), assembly=lm.loc[line, "assembly_id"]))
    print(f"{line}: {len(genes_opt)} KO fluxomes, |v/mu| max {np.abs(V).max():.1f}, {time.time() - t0:.0f}s", flush=True)
smp = pd.DataFrame(rows)
with open(f"{OUT}/samples.tsv", "w") as fh:
    fh.write(f"# {len(lines)} lines; label dependency >= 0.5; script 64; {time.ctime()}\n")
    smp.to_csv(fh, sep="\t", index=False)
print(f"samples {len(smp)}, in GNN {int(smp.in_gnn.sum())}, positive rate {smp.label.mean():.3f}")
