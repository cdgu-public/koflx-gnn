"""Serum-lipid sensitivity analysis (scope section 5, D12 option D): baseline MOMA (result/moma_20261027/<line>/)
vs serum-lipid variant (result/moma_serum_variant_20261029/own/<line>/; script_bash/92, cholesterol + palmitate +
oleate + stearate + arachidonate opened at the assembly's linolenate requirement cap, -0.000305 mmol/gDW/h each).
Five lines, same context model, same caps; the only varied factor is the five extra exchange lower bounds.
CPU only, pandas/scipy. Output: result/moma_serum_variant_20261029/analysis_20261030/ (new folder, no overwrite).
"""
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata, pearsonr

R = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
B = f"{R}/result/moma_20261027"; V = f"{R}/result/moma_serum_variant_20261029/own"
OUT = f"{R}/result/moma_serum_variant_20261029/analysis_20261030"
assert not os.path.exists(OUT), f"{OUT} exists; no overwrite"
os.makedirs(OUT)
LINES = ["ACH-000007", "ACH-000249", "ACH-000467", "ACH-000651", "ACH-001454"]
OPEN = ["MAR09285", "MAR00611", "MAR00650", "MAR00639", "MAR00568"]
LIPID = ["FASN", "ACACA", "ACACB", "SCD", "SCD5", "ELOVL6", "ACLY", "HMGCR", "HMGCS1", "MVK", "PMVK", "MVD", "IDI1", "FDPS",
         "FDFT1", "SQLE", "LSS", "CYP51A1", "DHCR7", "DHCR24", "SC5D", "NSDHL", "MSMO1", "HSD17B7", "EBP", "LDLR", "ACSL1",
         "ACSL3", "ACSL4", "FADS1", "FADS2", "GGPS1", "ACAT2", "SREBF1", "SREBF2", "SCAP", "MBTPS1", "MBTPS2"]
LETHAL = 0.05; PASS_J = 0.95; PASS_RHO = 0.95; REL_TH = 0.2
st = pd.read_csv(f"{R}/result/ko_screen_20261026/eval_20261027/struct_universe.tsv", sep="\t", index_col=0)
e2s = st.symbol.dropna().to_dict()
dep = pd.read_parquet(f"{R}/data/depmap/crc_trainset/crc_dependency.parquet")
meta_sc = json.load(open(f"{R}/result/gnn_20261028/dataset/scaffold_meta.json"))
EXCH = set(meta_sc["exchanges"])
MASKR = set(json.load(open(f"{R}/result/gnn_20261028/growth_mask_v2_20261029.json"))["mask_reactions"])
lm = pd.read_csv(f"{R}/data/medium/line_medium_map.tsv", sep="\t").set_index("ModelID")

def auroc(score, label):
    """Mann-Whitney AUROC with average ranks (ties count 0.5). score: higher = predicted essential."""
    label = np.asarray(label, bool); n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def hdr(fh, what, n, extra=""):
    fh.write(f"# {what}\n# baseline: {B}/<line>/ (script 58, per-line requirement caps script 62); variant: {V}/<line>/ "
             f"(script 58b via script_bash/92_run_moma_serum_variant.sh, open_exchanges {','.join(OPEN)} at the assembly "
             f"linolenate EFA_requirement_x_mu cap -0.000305 each)\n# params: MOMA-lethal = relative growth rate < {LETHAL}; "
             f"essentiality label = DepMap CRISPRGeneDependency >= 0.5 (data/depmap/crc_trainset/crc_dependency.parquet); "
             f"gene symbol map result/ko_screen_20261026/eval_20261027/struct_universe.tsv; AUROC = Mann-Whitney with average ranks, "
             f"score = -relative growth rate; pass criteria scope section 5: Jaccard >= {PASS_J}, Spearman rho >= {PASS_RHO}; "
             f"GNN normalization as script 81 (v/mu_ref, zero below 1e-6/mu_ref, 13 growth-mask reactions removed); {extra}\n"
             f"# tool: python {sys.version.split()[0]} pandas {pd.__version__} scipy; env crcgem; date {time.strftime('%Y-%m-%d %H:%M')} KST; n = {n}\n")

rows, disc, lipid_rows, prop, exch_rows, perko_rows = [], [], [], [], [], []
for ln in LINES:
    mb = json.load(open(f"{B}/{ln}/meta.json")); mv = json.load(open(f"{V}/{ln}/meta.json"))
    sb = pd.read_csv(f"{B}/{ln}/summary.tsv", sep="\t", index_col=0); sv = pd.read_csv(f"{V}/{ln}/summary.tsv", sep="\t", index_col=0)
    rb = pd.read_parquet(f"{B}/{ln}/reference.parquet")["reference"]; rv = pd.read_parquet(f"{V}/{ln}/reference.parquet")["reference"]
    fb = pd.read_parquet(f"{B}/{ln}/flux_matrix.parquet"); fv = pd.read_parquet(f"{V}/{ln}/flux_matrix.parquet")
    assert (rb.index == rv.index).all() and set(sb.index) == set(sv.index) and list(fb.columns) == list(fv.columns)
    mub, muv = mb["reference_growth"], mv["reference_growth"]
    # (a) WT exchanges: control check that only the five opened exchanges differ in uptake
    ex = [r for r in rb.index if r in EXCH]
    eb, ev = rb[ex], rv[ex]
    for r in ex:
        if abs(eb[r]) > 1e-9 or abs(ev[r]) > 1e-9:
            exch_rows.append(dict(line=ln, exchange_rxn_id=r, wt_flux_baseline=eb[r], wt_flux_variant=ev[r],
                                  opened_in_variant=r in OPEN, abs_diff=abs(eb[r] - ev[r])))
    new_uptake = [r for r in ex if ev[r] < -1e-9 and eb[r] >= -1e-9]
    # (b) lethal sets, growth-ratio Spearman
    both_opt = sb.index[(sb.status == "optimal") & (sv.status.reindex(sb.index) == "optimal")]
    lb_set = set(sb.index[(sb.status == "optimal") & (sb.growth_ratio < LETHAL)]); lv_set = set(sv.index[(sv.status == "optimal") & (sv.growth_ratio < LETHAL)])
    lb_o, lv_o = lb_set & set(both_opt), lv_set & set(both_opt)
    jac_all = len(lb_set & lv_set) / len(lb_set | lv_set); jac_opt = len(lb_o & lv_o) / len(lb_o | lv_o)
    rho_gr = spearmanr(sb.growth_ratio[both_opt], sv.growth_ratio[both_opt]).statistic
    flips = sorted((lb_set ^ lv_set))
    # (c) whole-fluxome agreement, WT (raw mmol/gDW/h, reactions nonzero in either)
    nz = (rb.abs() > 1e-9) | (rv.abs() > 1e-9)
    rho_wt = spearmanr(rb[nz], rv[nz]).statistic; pr_wt = pearsonr(rb[nz], rv[nz])[0]
    wt_linf = (rb - rv).abs().max(); wt_nchanged = int(((rb - rv).abs() > 1e-6).sum())
    # per-KO fluxome Spearman (raw), genes optimal in both
    genes_c = [g for g in both_opt if g in fb.index and g in fv.index]
    Fb = fb.loc[genes_c].to_numpy(); Fv = fv.loc[genes_c].to_numpy()
    rho_ko = np.full(len(genes_c), np.nan)
    for i in range(len(genes_c)):
        m = (np.abs(Fb[i]) > 1e-9) | (np.abs(Fv[i]) > 1e-9)
        rho_ko[i] = spearmanr(Fb[i, m], Fv[i, m]).statistic
    # (d) DepMap agreement
    gd = [g for g in both_opt if e2s.get(g) in dep.columns]
    lab = np.array([dep.loc[ln, e2s[g]] >= 0.5 for g in gd])
    au_b = auroc(-sb.growth_ratio[gd].to_numpy(), lab); au_v = auroc(-sv.growth_ratio[gd].to_numpy(), lab)
    # lethal-call accuracy vs label
    cb = np.array([g in lb_set for g in gd]); cv = np.array([g in lv_set for g in gd])
    def prf(c):
        tp = (c & lab).sum(); fp = (c & ~lab).sum(); fn = (~c & lab).sum()
        return tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    pb, rcb = prf(cb); pv, rcv = prf(cv)
    for g in sorted(set(flips) | set(both_opt[(sb.growth_ratio[both_opt] - sv.growth_ratio[both_opt]).abs() > 0.05])):
        s_ = e2s.get(g); d_ = float(dep.loc[ln, s_]) if s_ in dep.columns else np.nan
        disc.append(dict(line=ln, assembly=mb["assembly"], gene=g, symbol=s_ if s_ else "", growth_ratio_baseline=sb.growth_ratio.get(g, np.nan),
                         growth_ratio_variant=sv.growth_ratio.get(g, np.nan), status_baseline=sb.status[g], status_variant=sv.status[g],
                         lethal_baseline=g in lb_set, lethal_variant=g in lv_set, dependency=d_,
                         label=("" if np.isnan(d_) else int(d_ >= 0.5)), n_disabled=int(sb.n_disabled[g])))
    for g in sb.index:
        s_ = e2s.get(g)
        if s_ in LIPID:
            d_ = float(dep.loc[ln, s_]) if s_ in dep.columns else np.nan
            lipid_rows.append(dict(line=ln, gene=g, symbol=s_, growth_ratio_baseline=sb.growth_ratio[g], growth_ratio_variant=sv.growth_ratio[g],
                                   lethal_baseline=g in lb_set, lethal_variant=g in lv_set, dependency=d_, label=("" if np.isnan(d_) else int(d_ >= 0.5))))
    # (e) propagation to GNN inputs, script-81 normalization
    keep = np.array([r not in MASKR for r in fb.columns]); keep_noopen = keep & np.array([r not in OPEN for r in fb.columns])
    thb, thv = 1e-6 / mub, 1e-6 / muv
    vwb = rb.to_numpy() / mub; vwv = rv.to_numpy() / muv
    vwb = np.where(np.abs(vwb) > thb, vwb, 0); vwv = np.where(np.abs(vwv) > thv, vwv, 0)
    gg = [g for g in gd]  # GNN gene set: optimal in both, symbol in DepMap
    Vb = fb.loc[gg].to_numpy() / mub; Vv = fv.loc[gg].to_numpy() / muv
    Vb = np.where(np.abs(Vb) > thb, Vb, 0); Vv = np.where(np.abs(Vv) > thv, Vv, 0)
    wt_shift = np.abs(vwv - vwb)[keep].max(); wt_shift_noopen = np.abs(vwv - vwb)[keep_noopen].max()
    scale = np.abs(Vb - vwb)[:, keep].max(1)                      # the knockout's own |v_KO - v_WT|_inf (baseline), the rel_nn_dist denominator
    insample = scale > thb                                          # script 81 sample filter (knockouts that change the fluxome)
    d_ko = np.abs(Vv - Vb)[:, keep].max(1)                          # knockout fluxome, baseline -> variant
    d_ko_noopen = np.abs(Vv - Vb)[:, keep_noopen].max(1)
    d_shift = np.abs((Vv - vwv) - (Vb - vwb))[:, keep].max(1)       # the slog(v_KO - v_WT) feature
    rel = d_ko / scale; rel_no = d_ko_noopen / scale; rel_sh = d_shift / scale
    gr_b = sb.growth_ratio[gg].to_numpy()
    for i, g in enumerate(gg):
        perko_rows.append(dict(line=ln, gene=g, symbol=e2s[g], in_gnn_sample=bool(insample[i]), own_scale_Linf=scale[i], d_ko_Linf=d_ko[i],
                               d_ko_Linf_excl_opened=d_ko_noopen[i], d_shift_Linf=d_shift[i], rel_change=rel[i], rel_change_excl_opened=rel_no[i],
                               rel_change_shift=rel_sh[i], rho_ko_raw=rho_ko[genes_c.index(g)] if g in genes_c else np.nan,
                               growth_ratio_baseline=gr_b[i], growth_ratio_variant=sv.growth_ratio[g], dependency=float(dep.loc[ln, e2s[g]])))
    ins = insample
    prop.append(dict(line=ln, assembly=mb["assembly"], n_gnn_genes=len(gg), n_in_sample=int(ins.sum()),
                     wt_shift_Linf_norm=wt_shift, wt_shift_Linf_norm_excl_opened=wt_shift_noopen,
                     median_own_scale=np.median(scale[ins]), median_rel_change=np.median(rel[ins]),
                     frac_rel_ge_0p2=float((rel[ins] >= REL_TH).mean()), frac_rel_ge_0p5=float((rel[ins] >= 0.5).mean()), frac_rel_ge_1=float((rel[ins] >= 1).mean()),
                     frac_rel_excl_opened_ge_0p2=float((rel_no[ins] >= REL_TH).mean()), frac_rel_shift_ge_0p2=float((rel_sh[ins] >= REL_TH).mean()),
                     frac_rel_shift_ge_1=float((rel_sh[ins] >= 1).mean()),
                     frac_ko_identical=float((d_ko[ins] <= thb).mean()), n_rxn_wt_changed_norm=int((np.abs(vwv - vwb)[keep] > thb).sum()),
                     frac_rel_ge_0p2_lethal_b=float((rel[ins & (gr_b < LETHAL)] >= REL_TH).mean()),
                     frac_rel_ge_0p2_nonlethal_b=float((rel[ins & (gr_b >= LETHAL)] >= REL_TH).mean())))
    rows.append(dict(line=ln, assembly=mb["assembly"], wt_growth_baseline=mub, wt_growth_variant=muv, wt_growth_ratio_var_over_base=muv / mub,
                     n_screened=len(sb), n_optimal_baseline=int((sb.status == "optimal").sum()), n_optimal_variant=int((sv.status == "optimal").sum()),
                     n_optimal_both=len(both_opt), n_lethal_baseline=len(lb_set), n_lethal_variant=len(lv_set), n_lethal_flips=len(flips),
                     n_gained_lethal_variant=len(lv_set - lb_set), n_lost_lethal_variant=len(lb_set - lv_set),
                     jaccard_lethal_all=jac_all, jaccard_lethal_optimal_both=jac_opt, pass_jaccard=jac_opt >= PASS_J,
                     spearman_growth_ratio=rho_gr, spearman_wt_fluxome=rho_wt, pearson_wt_fluxome=pr_wt, pass_rho_wt=rho_wt >= PASS_RHO,
                     wt_fluxome_Linf_diff=wt_linf, n_wt_rxn_changed_1e6=wt_nchanged, new_wt_uptakes_variant=";".join(new_uptake),
                     ko_fluxome_spearman_median=np.nanmedian(rho_ko), ko_fluxome_spearman_min=np.nanmin(rho_ko),
                     frac_ko_spearman_ge_0p95=float(np.nanmean(rho_ko >= PASS_RHO)), n_depmap_genes=len(gd), n_depmap_pos=int(lab.sum()),
                     auroc_baseline=au_b, auroc_variant=au_v, auroc_delta=au_v - au_b,
                     lethal_precision_baseline=pb, lethal_recall_baseline=rcb, lethal_precision_variant=pv, lethal_recall_variant=rcv))
    print(ln, f"J={jac_opt:.3f} rho_gr={rho_gr:.3f} rho_wt={rho_wt:.3f} AUROC {au_b:.3f}->{au_v:.3f} flips={len(flips)} "
          f"rel>=0.2 {prop[-1]['frac_rel_ge_0p2']:.3f} wt_shift {wt_shift:.3g}", flush=True)

T = pd.DataFrame(rows); D = pd.DataFrame(disc); Lp = pd.DataFrame(lipid_rows); P = pd.DataFrame(prop); E = pd.DataFrame(exch_rows); K = pd.DataFrame(perko_rows)
# pooled AUROC over the 5 lines
pool_b, pool_v, pool_lab = [], [], []
for ln in LINES:
    sb = pd.read_csv(f"{B}/{ln}/summary.tsv", sep="\t", index_col=0); sv = pd.read_csv(f"{V}/{ln}/summary.tsv", sep="\t", index_col=0)
    gd = [g for g in sb.index if sb.status[g] == "optimal" and sv.status[g] == "optimal" and e2s.get(g) in dep.columns]
    pool_b += list(-sb.growth_ratio[gd]); pool_v += list(-sv.growth_ratio[gd]); pool_lab += [dep.loc[ln, e2s[g]] >= 0.5 for g in gd]
pooled = dict(line="POOLED_5", n_depmap_genes=len(pool_lab), n_depmap_pos=int(sum(pool_lab)), auroc_baseline=auroc(np.array(pool_b), pool_lab),
              auroc_variant=auroc(np.array(pool_v), pool_lab))
pooled["auroc_delta"] = pooled["auroc_variant"] - pooled["auroc_baseline"]
T = pd.concat([T, pd.DataFrame([pooled])], ignore_index=True)
with open(f"{OUT}/table1_per_line_comparison.tsv", "w") as fh:
    hdr(fh, "Table 1. Per-line baseline vs serum-lipid variant: WT growth, MOMA-lethal set, growth-ratio Spearman, WT and per-knockout fluxome agreement, DepMap AUROC; last row pooled over the 5 lines", len(T),
        "jaccard_lethal_optimal_both and spearman_growth_ratio use genes optimal in both runs; new_wt_uptakes_variant = exchange reactions with WT uptake in the variant and none in the baseline")
    T.to_csv(fh, sep="\t", index=False, float_format="%.6g")
with open(f"{OUT}/table2_discordant_genes.tsv", "w") as fh:
    hdr(fh, "Table 2. Genes whose MOMA-lethal call flips or whose relative growth rate changes by > 0.05 between baseline and variant", len(D))
    D.to_csv(fh, sep="\t", index=False, float_format="%.6g")
with open(f"{OUT}/table3_lipid_gene_panel.tsv", "w") as fh:
    hdr(fh, "Table 3. Pre-named lipid-synthesis genes (fatty-acid and cholesterol synthesis, SREBP axis) in every line, changed or not", len(Lp),
        f"panel = {','.join(LIPID)}; genes absent from a line's screen are absent from the table (not in that context model)")
    Lp.to_csv(fh, sep="\t", index=False, float_format="%.6g")
with open(f"{OUT}/table4_gnn_input_propagation.tsv", "w") as fh:
    hdr(fh, "Table 4. Propagation to the GNN inputs: per knockout, Chebyshev distance between the variant and baseline knockout fluxomes (v/mu_ref, growth mask applied) divided by the knockout's own baseline |v_KO - v_WT|_inf (the relative nearest-neighbor distance denominator); fractions over knockouts passing the script-81 sample filter", len(P),
        "rel_change_shift uses the (v_KO - v_WT) feature; excl_opened removes the 5 opened exchange reactions from the distance; frac_ko_identical = knockout fluxome unchanged within 1e-6/mu")
    P.to_csv(fh, sep="\t", index=False, float_format="%.6g")
with open(f"{OUT}/table5_wt_exchange_fluxes.tsv", "w") as fh:
    hdr(fh, "Table 5. WT (pFBA reference) exchange fluxes nonzero in either run, control check that only the opened exchanges add uptake", len(E))
    E.to_csv(fh, sep="\t", index=False, float_format="%.6g")
with open(f"{OUT}/table6_per_knockout_propagation.tsv", "w") as fh:
    hdr(fh, "Table 6. Per-knockout propagation values behind Table 4 (all GNN genes, in_gnn_sample marks the script-81 sample filter)", len(K))
    K.to_csv(fh, sep="\t", index=False, float_format="%.6g")
print(T[["line", "wt_growth_ratio_var_over_base", "jaccard_lethal_optimal_both", "spearman_growth_ratio", "spearman_wt_fluxome", "auroc_baseline", "auroc_variant", "n_lethal_flips"]].to_string())
print(P[["line", "n_in_sample", "wt_shift_Linf_norm", "median_rel_change", "frac_rel_ge_0p2", "frac_rel_excl_opened_ge_0p2", "frac_rel_shift_ge_0p2", "frac_rel_ge_1", "frac_ko_identical"]].to_string())
print(f"done {OUT}")
