"""Patient POC, KEGG pathway-map distance restricted to central-metabolism maps (user 2026-11-04 ~23:5x: the purpose of the per-map
readout is the metabolic shift in pathways central to growth and maintenance, e.g. glycolysis, pentose phosphate pathway,
beta-oxidation). Maps: map00010 Glycolysis / Gluconeogenesis, map00030 Pentose phosphate pathway, map00071 Fatty acid degradation
(beta-oxidation), map00020 Citrate cycle (TCA cycle), map00190 Oxidative phosphorylation.
Input (read-only): distance/poc_percentile_gated_20261104.tsv (tumor pFBA, normal OptGP) and
distance/poc_percentile_tumor_optgp_gated_20261104.tsv (rows side == 'normal' = tumor model with OptGP references).
Readout d_sym, route kegg_reaction_or_ec (main route by ruling); within-model percentile of the drug target among the control
genes of the same model and reference (rule of script 174), median over references; here median [min, max] over the 3 patients.
Output: distance/poc_central_pathways_20261104.tsv with '#' header; printed pivot."""
import datetime
import pandas as pd
D = "/data/workspace/cdgu/Winery/seeding/CRC_GEM/result/patient_poc_20261103/distance/"
MAPS = ["map00010", "map00030", "map00071", "map00020", "map00190"]
a = pd.read_csv(D + "poc_percentile_gated_20261104.tsv", sep="\t", comment="#")
a["reference"] = a.side.map({"tumor": "tumor_pFBA", "normal": "normal_OptGP"})
b = pd.read_csv(D + "poc_percentile_tumor_optgp_gated_20261104.tsv", sep="\t", comment="#")
b = b[b.side == "normal"].assign(reference="tumor_OptGP")
k = pd.concat([a, b])
k = k[(k.readout == "d_sym") & (k.route == "kegg_reaction_or_ec") & k.kegg_map.isin(MAPS) & (k.symbol != "GLS")]
f = lambda s: f"{s.median():.0f} [{s.min():.0f}, {s.max():.0f}] n={s.notna().sum()}"
o = k.groupby(["kegg_map", "kegg_map_name", "symbol", "reference"]).agg(
    percentile=("percentile", f), d_sym_median=("value_target_median", "median")).reset_index()
hdr = [f"# source: {D}poc_percentile_gated_20261104.tsv; {D}poc_percentile_tumor_optgp_gated_20261104.tsv (side normal rows only)",
       "# readout d_sym, route kegg_reaction_or_ec; percentile among 10 control genes of the same model and reference (script 174 rule), median over references",
       "# per row: median [min, max] over patients, n = patients with a percentile; d_sym_median = median over patients of the per-model median d_sym",
       f"# maps {MAPS}; GLS excluded (absent from 4/6 models); script_python/205_poc_central_pathways.py; {datetime.datetime.now().isoformat(timespec='seconds')}; n rows {len(o)}"]
with open(D + "poc_central_pathways_20261104.tsv", "w") as fh:
    fh.write("\n".join(hdr) + "\n"); o.to_csv(fh, sep="\t", index=False)
pd.set_option("display.width", 250)
print(o.pivot_table(index=["kegg_map_name", "symbol"], columns="reference", values="percentile", aggfunc="first").to_string())
