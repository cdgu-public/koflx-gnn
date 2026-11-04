"""QC of the MOMA knockout-screen outputs (result/moma_20261027/<line>/, script 58) before they are
used as training data. Re-runnable on a partial set; every table states n.

Per line:
 1. meta: status, model size, WT growth, reference growth, gap-fill, reference fallback.
 2. summary.tsv: screened genes, solver status counts, growth / growth ratio range, lethal (<0.05)
    and no-effect (ratio 1) counts, negative MOMA objective (should be >= 0 up to tolerance).
 3. flux_matrix.parquet: rows = optimal KOs, columns = context reactions; NaN/inf; |v| max; fluxes
    at |v| >= 999 (bound-hitting, loop signal).
 4. reference.parquet: biomass equals reference growth; steady state S.v = 0 with the scaffold S
    (context and gap-filled reactions are scaffold reactions); exchange lower bounds of the line's
    assembly, with the per-line requirement caps of script 62 (meta.json caps_raised), respected.
 5. KO rows: steady state on every row; every reaction the knockout disables (scaffold GPR, the
    rules tINIT keeps) carries |v| <= 1e-6.
 6. labels: screened genes mapped to DepMap symbols through struct_universe.tsv (script 50);
    coverage and positive rate (dependency >= 0.5).
Env: crcgem. Output: result/moma_20261027/qc_<date>/{per_line.tsv, summary.txt}
"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd, cobra
from cobra.util.array import create_stoichiometric_matrix

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
R = f"{ROOT}/result/moma_20261027"
OUT = f"{R}/qc_{time.strftime('%Y%m%d_%H%M')}"
os.makedirs(OUT, exist_ok=True)
TOL_SS, TOL_KO, TOL_LB = 1e-6, 1e-6, 1e-6

scaf = cobra.io.read_sbml_model(f"{ROOT}/data/scaffold/Human-GEM.xml")
S = create_stoichiometric_matrix(scaf, array_type="DataFrame")  # mets x rxns
S = S.astype(float)
RID = list(S.columns)
EX = {r.id for r in scaf.exchanges}
GENE_RXNS = {g.id: [r.id for r in g.reactions if r.gpr.eval({g.id}) is False] for g in scaf.genes}
lm = pd.read_csv(f"{ROOT}/data/medium/line_medium_map.tsv", sep="\t").set_index("ModelID")
LB = pd.read_csv(f"{ROOT}/data/medium/uptake_caps/exchange_lower_bounds_by_assembly.tsv", sep="\t", comment="#")
st = pd.read_csv(f"{ROOT}/result/ko_screen_20261026/eval_20261027/struct_universe.tsv", sep="\t", index_col=0)
e2s = st.symbol.dropna().to_dict()
dep = pd.read_parquet(f"{ROOT}/data/depmap/crc_trainset/crc_dependency.parquet")


def ss_residual(v):
    """max |S.v| over metabolites, v a Series on context reactions."""
    s = S[v.index.intersection(RID)]
    return float(np.abs(s.to_numpy() @ v[s.columns].to_numpy()).max())


def qc(line):
    d = f"{R}/{line}"
    meta = json.load(open(f"{d}/meta.json"))
    row = dict(line=line, status=meta.get("status"), assembly=meta.get("assembly"),
               n_reactions=meta.get("n_reactions"), wt_growth=meta.get("wt_growth"),
               reference_growth=meta.get("reference_growth"),
               n_gapfill=len(meta.get("gapfill_added", [])), reference_fallback=meta.get("reference_fallback", ""))
    if meta.get("status") != "done":
        return row
    sm = pd.read_csv(f"{d}/summary.tsv", sep="\t", index_col=0)
    fx = pd.read_parquet(f"{d}/flux_matrix.parquet")
    ref = pd.read_parquet(f"{d}/reference.parquet")["reference"]
    opt = sm[sm.status == "optimal"]
    row.update(n_screened=len(sm), n_optimal=len(opt), n_nonoptimal=int((sm.status != "optimal").sum()),
               nonoptimal_status=";".join(f"{k}:{v}" for k, v in sm.status[sm.status != "optimal"].value_counts().items()),
               ratio_min=opt.growth_ratio.min(), ratio_max=opt.growth_ratio.max(),
               n_ratio_gt1=int((opt.growth_ratio > 1 + 1e-6).sum()),
               n_lethal_0p05=int((opt.growth_ratio < 0.05).sum()),
               n_ratio_partial=int(((opt.growth_ratio >= 0.05) & (opt.growth_ratio < 1 - 1e-6)).sum()),
               n_no_effect=int((opt.growth_ratio >= 1 - 1e-6).sum()),
               moma_obj_min=opt.moma_objective.min(), n_moma_obj_neg=int((opt.moma_objective < -1e-6).sum()))
    # flux matrix
    row.update(fx_rows=fx.shape[0], fx_cols=fx.shape[1], fx_rows_eq_optimal=set(fx.index) == set(opt.index),
               fx_cols_eq_ref=set(fx.columns) == set(ref.index),
               fx_n_nan=int(fx.isna().sum().sum()), fx_n_inf=int(np.isinf(fx.to_numpy()).sum()),
               fx_absmax=float(np.nanmax(np.abs(fx.to_numpy()))),
               fx_n_at_bound=int((np.abs(fx.to_numpy()) >= 999).sum()),
               fx_rows_with_bound=int((np.abs(fx.to_numpy()) >= 999).any(1).sum()),
               fx_not_in_scaffold=len(set(fx.columns) - set(RID)))
    # reference
    lb = LB[LB.assembly_id == lm.loc[line, "assembly_id"]].set_index("exchange_rxn_id").lower_bound
    ex = [r for r in ref.index if r in EX]
    lbv = pd.Series({r: lb.get(r, 0.0) for r in ex})
    # per-line requirement caps (script 62) as applied by script 58 and recorded in meta.json
    lbv.update(pd.Series(meta.get("caps_raised", {}), dtype=float).reindex(lbv.index).dropna())
    row.update(ref_n_nan=int(ref.isna().sum()), ref_biomass=float(ref.get("MAR13082", np.nan)),
               ref_biomass_rel_err=abs(ref.get("MAR13082", np.nan) - meta["reference_growth"]) / meta["reference_growth"],
               ref_absmax=float(ref.abs().max()), ref_n_at_bound=int((ref.abs() >= 999).sum()),
               ref_n_active=int((ref.abs() > 1e-9).sum()),
               ref_ss_resid=ss_residual(ref), ref_lb_viol=int((ref[ex] < lbv - TOL_LB).sum()),
               ref_n_uptake=int((ref[ex] < -1e-9).sum()))
    # KO rows: steady state and disabled-reaction flux
    s = S[fx.columns.intersection(RID)].to_numpy()
    F = fx[fx.columns.intersection(RID)].to_numpy(dtype=float)
    res = np.abs(F @ s.T).max(1)
    row.update(ko_ss_resid_max=float(res.max()), ko_n_rows_ss_viol=int((res > TOL_SS).sum()))
    colpos = {c: i for i, c in enumerate(fx.columns)}
    nviol, nrow_viol, nmissing = 0, 0, 0
    for i, g in enumerate(fx.index):
        rx = [colpos[r] for r in GENE_RXNS.get(g, []) if r in colpos]
        if not rx:
            nmissing += 1; continue
        v = np.abs(fx.iloc[i, rx].to_numpy(dtype=float))
        k = int((v > TOL_KO).sum()); nviol += k; nrow_viol += int(k > 0)
    row.update(ko_disabled_flux_viol=nviol, ko_rows_disabled_viol=nrow_viol, ko_rows_no_disabled_rxn=nmissing)
    # biomass-normalized scale (input to the GNN: v / WT biomass flux)
    row.update(norm_ref_absmax=float(ref.abs().max() / meta["reference_growth"]))
    # labels
    syms = [e2s.get(g) for g in sm.index]
    ok = [s_ for s_ in syms if s_ is not None and s_ in dep.columns]
    row.update(label_line_in_depmap=line in dep.index, n_genes_mapped=len(ok),
               n_genes_unmapped=len(syms) - len(ok),
               n_pos_0p5=int((dep.loc[line, ok] >= 0.5).sum()) if line in dep.index else np.nan)
    return row


if __name__ == "__main__":
    lines = sys.argv[2:] or sorted(x for x in os.listdir(R) if x.startswith("ACH-") and os.path.exists(f"{R}/{x}/meta.json"))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p:
        rows = p.map(qc, lines)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/per_line.tsv", sep="\t", index=False)
    done = df[df.status == "done"]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80)
    with open(f"{OUT}/summary.txt", "w") as fh:
        fh.write(f"# condition: {len(df)} lines with meta.json, {len(done)} done; tolerances steady state "
                 f"{TOL_SS}, disabled flux {TOL_KO}, lower bound {TOL_LB}; run {time.ctime()}\n")
        fh.write(done.describe().T.to_string() + "\n\n")
        fh.write(df.to_string(index=False) + "\n")
    print(open(f"{OUT}/summary.txt").read()[:6000])
