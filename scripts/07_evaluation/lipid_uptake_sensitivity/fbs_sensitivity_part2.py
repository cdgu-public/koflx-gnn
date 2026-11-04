"""Part 2 of the serum-lipid sensitivity analysis: (i) control check that the opened exchanges exist in each context
model, (ii) tie-robust growth-ratio agreement, (iii) reactions whose WT flux changes (Human-GEM names, subsystems),
(iv) reference-point test for every flipped gene: flux through its disabled reactions in the baseline vs variant WT
reference, (v) paired bootstrap of the pooled AUROC difference, (vi) full lipid-gene panel wide view.
Writes new files into result/moma_serum_variant_20261029/analysis_20261030/ (no existing file overwritten)."""
import json, os, sys, time
import numpy as np, pandas as pd, cobra
from scipy.stats import spearmanr, rankdata, pearsonr

R = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
B = f"{R}/result/moma_20261027"; V = f"{R}/result/moma_serum_variant_20261029/own"
OUT = f"{R}/result/moma_serum_variant_20261029/analysis_20261030"
LINES = ["ACH-000007", "ACH-000249", "ACH-000467", "ACH-000651", "ACH-001454"]
OPEN = ["MAR09285", "MAR00611", "MAR00650", "MAR00639", "MAR00568"]
LETHAL = 0.05
for f in ["table7_reactions_changed_wt.tsv", "table8_flipped_genes_reference_test.tsv", "table9_robust_growth_agreement.tsv", "table10_lipid_panel_wide.tsv"]:
    assert not os.path.exists(f"{OUT}/{f}"), f"{f} exists; no overwrite"
st = pd.read_csv(f"{R}/result/ko_screen_20261026/eval_20261027/struct_universe.tsv", sep="\t", index_col=0); e2s = st.symbol.dropna().to_dict()
dep = pd.read_parquet(f"{R}/data/depmap/crc_trainset/crc_dependency.parquet")
ms = json.load(open(f"{R}/result/gnn_20261028/dataset/scaffold_meta.json")); RID = ms["reactions"]; GR = ms["gene_rxns"]
t0 = time.time(); scaf = cobra.io.read_sbml_model(f"{R}/data/scaffold/Human-GEM.xml"); print(f"scaffold loaded {time.time()-t0:.0f}s", flush=True)
rname = {r.id: r.name for r in scaf.reactions}; rsub = {r.id: ";".join(r.subsystem) if isinstance(r.subsystem, list) else str(r.subsystem) for r in scaf.reactions}
rgenes = {r.id: ";".join(sorted(e2s.get(g.id, g.id) for g in r.genes)) for r in scaf.reactions}

def auroc(score, label):
    label = np.asarray(label, bool); n1, n0 = label.sum(), (~label).sum(); r = rankdata(score)
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def hdr(fh, what, n, extra=""):
    fh.write(f"# {what}\n# baseline: {B}/<line>/; variant: {V}/<line>/ (script_bash/92, open_exchanges {','.join(OPEN)} at -0.000305 each); "
             f"Human-GEM names/subsystems from data/scaffold/Human-GEM.xml; MOMA-lethal = relative growth rate < {LETHAL}; {extra}\n"
             f"# tool: python {sys.version.split()[0]} pandas {pd.__version__} cobra {cobra.__version__}; env crcgem; date {time.strftime('%Y-%m-%d %H:%M')} KST; n = {n}\n")

rx_rows, flip_rows, rob_rows, ctl = [], [], [], []
pool_b, pool_v, pool_lab, pool_line = [], [], [], []
for ln in LINES:
    mb = json.load(open(f"{B}/{ln}/meta.json")); mv = json.load(open(f"{V}/{ln}/meta.json"))
    sb = pd.read_csv(f"{B}/{ln}/summary.tsv", sep="\t", index_col=0); sv = pd.read_csv(f"{V}/{ln}/summary.tsv", sep="\t", index_col=0)
    rb = pd.read_parquet(f"{B}/{ln}/reference.parquet")["reference"]; rv = pd.read_parquet(f"{V}/{ln}/reference.parquet")["reference"]
    mub, muv = mb["reference_growth"], mv["reference_growth"]
    present = {r: (r in rb.index) for r in OPEN}
    ctl.append(dict(line=ln, assembly=mb["assembly"], **{f"in_model_{r}": present[r] for r in OPEN},
                    **{f"wt_uptake_variant_{r}": (float(rv[r]) if r in rv.index else np.nan) for r in OPEN}))
    # (ii) tie-robust growth agreement
    both = sb.index[(sb.status == "optimal") & (sv.status == "optimal")]
    gb, gv = sb.growth_ratio[both].clip(lower=0), sv.growth_ratio[both].clip(lower=0)
    sel = (gb < 0.99) | (gv < 0.99)
    rob_rows.append(dict(line=ln, n_optimal_both=len(both), n_either_below_0p99=int(sel.sum()),
                         spearman_all_raw=spearmanr(gb, gv).statistic, spearman_rounded_1e3=spearmanr(gb.round(3), gv.round(3)).statistic,
                         spearman_either_below_0p99=spearmanr(gb[sel], gv[sel]).statistic, pearson_all=pearsonr(gb, gv)[0],
                         kendall_na="", frac_abs_diff_le_0p01=float(((gb - gv).abs() <= 0.01).mean()), frac_abs_diff_gt_0p05=float(((gb - gv).abs() > 0.05).mean()),
                         n_lethal_flips=int(((gb < LETHAL) != (gv < LETHAL)).sum())))
    # (iii) reactions whose WT flux changes (raw and normalized), top 40 by |delta v/mu|
    d = (rv / muv - rb / mub); dn = d.abs().sort_values(ascending=False)
    for r in dn.index[:40]:
        if dn[r] < 1e-6 / mub:
            break
        rx_rows.append(dict(line=ln, reaction=r, name=rname.get(r, ""), subsystem=rsub.get(r, ""), genes=rgenes.get(r, ""),
                            wt_flux_baseline=rb[r], wt_flux_variant=rv[r], delta_v_over_mu=d[r], opened_exchange=r in OPEN))
    # (iv) flipped genes: WT reference flux through the disabled reactions in each run
    lb_set = set(sb.index[(sb.status == "optimal") & (sb.growth_ratio < LETHAL)]); lv_set = set(sv.index[(sv.status == "optimal") & (sv.growth_ratio < LETHAL)])
    for g in sorted(lb_set ^ lv_set):
        rx = [RID[i] for i in GR.get(g, [])]; rx_in = [r for r in rx if r in rb.index]
        fb_ = rb[rx_in].abs().max() if rx_in else np.nan; fv_ = rv[rx_in].abs().max() if rx_in else np.nan
        s_ = e2s.get(g, "")
        flip_rows.append(dict(line=ln, gene=g, symbol=s_, lethal_baseline=g in lb_set, lethal_variant=g in lv_set,
                              growth_ratio_baseline=sb.growth_ratio[g], growth_ratio_variant=sv.growth_ratio[g],
                              dependency=(float(dep.loc[ln, s_]) if s_ in dep.columns else np.nan),
                              n_disabled=len(rx_in), disabled_reactions=";".join(rx_in), subsystems=";".join(sorted({rsub.get(r, "") for r in rx_in})),
                              wt_flux_disabled_max_baseline=fb_, wt_flux_disabled_max_variant=fv_,
                              wt_carries_flux_baseline=bool(fb_ > 1e-9), wt_carries_flux_variant=bool(fv_ > 1e-9),
                              lipid_subsystem=any(("holesterol" in rsub.get(r, "")) or ("atty acid" in rsub.get(r, "")) or ("teroid" in rsub.get(r, "")) for r in rx_in)))
    gd = [g for g in both if e2s.get(g) in dep.columns]
    pool_b += list(-sb.growth_ratio[gd]); pool_v += list(-sv.growth_ratio[gd]); pool_lab += [dep.loc[ln, e2s[g]] >= 0.5 for g in gd]; pool_line += [ln] * len(gd)
    print(ln, "ctl", present, flush=True)
# (v) paired bootstrap of pooled delta AUROC (resample genes with replacement, 2000 draws, seed 0)
pb, pv, pl = np.array(pool_b), np.array(pool_v), np.array(pool_lab, bool); rng = np.random.default_rng(0); n = len(pb); deltas = []
for _ in range(2000):
    i = rng.integers(0, n, n)
    if pl[i].sum() == 0 or (~pl[i]).sum() == 0:
        continue
    deltas.append(auroc(pv[i], pl[i]) - auroc(pb[i], pl[i]))
deltas = np.array(deltas); d0 = auroc(pv, pl) - auroc(pb, pl)
boot = dict(n_genes=n, n_pos=int(pl.sum()), auroc_baseline=auroc(pb, pl), auroc_variant=auroc(pv, pl), delta=d0,
            boot_ci95_low=float(np.percentile(deltas, 2.5)), boot_ci95_high=float(np.percentile(deltas, 97.5)), n_boot=len(deltas), seed=0)
with open(f"{OUT}/table9_robust_growth_agreement.tsv", "w") as fh:
    hdr(fh, "Table 9. Tie-robust agreement of relative growth rate (genes optimal in both runs); last row = control check of opened exchanges per context model; then paired gene bootstrap of the pooled AUROC difference", len(rob_rows),
        "spearman_either_below_0p99 drops knockouts whose relative growth rate is >= 0.99 in both runs (ties at 1 differing only by solver noise)")
    pd.DataFrame(rob_rows).to_csv(fh, sep="\t", index=False, float_format="%.6g")
    fh.write("# control check: opened exchange present in the context model and its WT uptake in the variant\n"); pd.DataFrame(ctl).to_csv(fh, sep="\t", index=False, float_format="%.6g")
    fh.write("# pooled AUROC, paired bootstrap over genes\n"); pd.DataFrame([boot]).to_csv(fh, sep="\t", index=False, float_format="%.6g")
with open(f"{OUT}/table7_reactions_changed_wt.tsv", "w") as fh:
    hdr(fh, "Table 7. Reactions whose WT (pFBA reference) flux changes most between baseline and variant, per line, top 40 by |delta (v/mu)|", len(rx_rows))
    pd.DataFrame(rx_rows).to_csv(fh, sep="\t", index=False, float_format="%.6g")
with open(f"{OUT}/table8_flipped_genes_reference_test.tsv", "w") as fh:
    hdr(fh, "Table 8. Every gene whose MOMA-lethal call flips, with the maximal |WT reference flux| through its disabled reactions in each run (reference-point test) and whether the disabled reactions are in a cholesterol/fatty-acid/steroid subsystem", len(flip_rows))
    pd.DataFrame(flip_rows).to_csv(fh, sep="\t", index=False, float_format="%.6g")
# (vi) lipid panel wide
L = pd.read_csv(f"{OUT}/table3_lipid_gene_panel.tsv", sep="\t", comment="#")
W = L.pivot_table(index="symbol", columns="line", values=["growth_ratio_baseline", "growth_ratio_variant"], aggfunc="first")
W.columns = [f"{a[-8:]}_{b}" for a, b in W.columns]
with open(f"{OUT}/table10_lipid_panel_wide.tsv", "w") as fh:
    hdr(fh, "Table 10. Lipid-gene panel, relative growth rate baseline vs variant by line (wide view of Table 3)", len(W))
    W.round(4).to_csv(fh, sep="\t")
print(pd.DataFrame(rob_rows).to_string()); print(pd.DataFrame(ctl).to_string()); print(boot)
F = pd.DataFrame(flip_rows); print(F[["line", "symbol", "lethal_baseline", "lethal_variant", "wt_carries_flux_baseline", "wt_carries_flux_variant", "lipid_subsystem", "subsystems"]].to_string())
print(W.round(3).to_string())
