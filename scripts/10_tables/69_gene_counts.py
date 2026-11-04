"""Gene counts before the knockout results (user, 2026-10-28): how many Human-GEM genes enter each stage and
how many of them are DepMap-lethal (CRISPRGeneDependency >= 0.5), 59 DepMap CRC lines.

Stages (gene level; per line where the stage is line-specific):
 G0  DepMap CRISPR genes with a dependency value in all 59 lines
 G1  Human-GEM v2.0.0 genes (scaffold)
 G2  Human-GEM genes in the expression panel whose symbol is a DepMap gene with no NaN over the 59 lines
     (struct_universe.tsv, script 50) = the evaluation universe
 G3  G2 genes whose single-gene knockout disables >= 1 scaffold reaction (scaffold GPR)
 L1  per line: G2 genes in the line's context model (tINIT + task repair + script-62 gap-fill + script-58
     gap-fill, i.e. the model the screen ran on)
 L2  per line: knockouts screened by MOMA (disable >= 1 reaction of the context model) with a DepMap label
     (samples.tsv, script 64)
 L3  L2 with a MOMA solution (status optimal)
 L4  L3 knockouts that change the line's fluxome (|v_KO - v_WT| > 1e-6 raw; GNN set, script 65 rule)
Per stage: genes (or line-gene pairs), DepMap-lethal among them; per line median (min-max); genes lethal in
>= 1 line and in all 59 lines. The lethal share of all DepMap-lethal genes of the line (G0) is also given.
Env: crcgem (cobra). Output: <out>/{t1_stages.tsv, t2_per_line.tsv, summary.txt}
CLI (added 2026-11-01, user ruling "Table 1 다시"; defaults reproduce the 2026-10-28 run):
  --dataset  GNN dataset dir with samples.tsv and lines/*.npz   (default result/gnn_20261028/dataset;
             tolerance patch: result/gnn_20261030_tolpatch/dataset_v2)
  --moma-root MOMA root with <line>/meta.json (gap-fill lists)   (default result/moma_20261027)
  --out      output dir, must not exist unless --overwrite        (default result/gene_counts_20261028)
L3 honours the samples.tsv column excluded_reason when present (dataset_v2: 14 knockouts excluded by user ruling
2026-10-30, status optimal but no optimum under the patch condition); L4 is L3 and the fluxome-change rule.
"""
import os, glob, json, warnings, argparse, time
warnings.filterwarnings("ignore")
from multiprocessing import Pool
import numpy as np, pandas as pd, cobra

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=f"{ROOT}/result/gnn_20261028/dataset")
ap.add_argument("--moma-root", default=f"{ROOT}/result/moma_20261027")
ap.add_argument("--out", default=f"{ROOT}/result/gene_counts_20261028")
ap.add_argument("--overwrite", action="store_true")
args = ap.parse_args()
OUT, DS, MR = args.out, args.dataset, args.moma_root
if os.path.exists(OUT) and not args.overwrite:
    raise SystemExit(f"{OUT} exists; choose a new --out (never overwrite results) or pass --overwrite")
os.makedirs(OUT, exist_ok=True)
DATE = time.strftime("%Y-%m-%d %H:%M %Z")
log = open(f"{OUT}/summary.txt", "w")
def say(*a):
    s = " ".join(map(str, a)); print(s, flush=True); log.write(s + "\n")

dep = pd.read_parquet(f"{ROOT}/data/depmap/crc_trainset/crc_dependency.parquet")
dep = dep.loc[:, ~dep.isna().any()]
LINES = sorted(dep.index)
st = pd.read_csv(f"{ROOT}/result/ko_screen_20261026/eval_20261027/struct_universe.tsv", sep="\t", index_col=0)
hg = pd.read_csv(f"{ROOT}/data/scaffold/humangem_genes.tsv", sep="\t")
smp = pd.read_csv(f"{DS}/samples.tsv", sep="\t", comment="#")
if "excluded_reason" not in smp: smp["excluded_reason"] = ""
smp["excluded_reason"] = smp.excluded_reason.fillna("").astype(str)
smp["kept"] = (smp.status == "optimal") & (smp.excluded_reason == "")
eff = {}
for f in glob.glob(f"{DS}/lines/*.npz"):
    z = np.load(f); ln = os.path.basename(f)[:-4]; th = 1e-6 / float(z["mu_ref"])
    d = np.abs(np.where(np.abs(z["V"]) > th, z["V"], 0) - np.where(np.abs(z["vwt"]) > th, z["vwt"], 0)).max(1)
    eff.update({(ln, g): bool(x > th) for g, x in zip(z["genes"], d)})
smp["effect"] = [eff.get((l, g), False) for l, g in zip(smp.line, smp.gene)]


def context_genes(line):
    m = cobra.io.read_sbml_model(f"{ROOT}/result/tinit_localT2_20261027/models/{line}/{line}.repaired.xml")
    meta = json.load(open(f"{MR}/{line}/meta.json"))
    genes = {g.id for g in m.genes}
    add = set(meta.get("gapfill_script62", [])) | set(meta.get("gapfill_added", []))
    if add:
        scaf = cobra.io.read_sbml_model(f"{ROOT}/data/scaffold/Human-GEM.xml")
        genes |= {g.id for r in add if r in scaf.reactions for g in scaf.reactions.get_by_id(r).genes}
    return line, sorted(genes), meta.get("n_genes")


with Pool(6) as p:
    CG = {ln: (g, n) for ln, g, n in p.map(context_genes, LINES)}

def _hdr(desc, n):
    return (f"# script_python/69_gene_counts.py; dataset {DS} (samples.tsv, lines/*.npz); moma_root {MR} (meta.json gap-fill lists); "
            f"context models {ROOT}/result/tinit_localT2_20261027/models/<line>/<line>.repaired.xml; scaffold Human-GEM v2.0.0; "
            f"DepMap 24Q4 CRISPRGeneDependency {ROOT}/data/depmap/crc_trainset/crc_dependency.parquet, essential = dependency >= 0.5, "
            f"genes with a value in all {len(LINES)} cell lines; universe {ROOT}/result/ko_screen_20261026/eval_20261027/struct_universe.tsv; "
            f"L3 = status optimal and excluded_reason empty (excluded knockouts in samples.tsv: {int((smp.excluded_reason != '').sum())}); "
            f"L4 = L3 and max|v_KO - v_WT| > 1e-6/mu_ref after zeroing |v| <= 1e-6/mu_ref (81b rule); env crcgem; date {DATE}\n"
            f"# {desc}; n rows {n}\n")
def save_tsv(df, path, desc):
    with open(path, "w") as fh:
        fh.write(_hdr(desc, len(df))); df.to_csv(fh, sep="\t", index=False)
e2s = st.symbol.to_dict()
lethal = dep >= 0.5
def gene_stage(name, syms):
    syms = sorted(set(syms)); L = lethal[syms]
    per = L.sum(axis=1)
    return dict(stage=name, level="gene", n=len(syms), lethal_per_line_median=per.median(),
                lethal_per_line_min=per.min(), lethal_per_line_max=per.max(),
                genes_lethal_ge1_line=int(L.any(axis=0).sum()), genes_lethal_all_lines=int(L.all(axis=0).sum()))
rows = [gene_stage("G0 DepMap genes, no NaN over 59 lines", dep.columns),
        dict(stage="G1 Human-GEM v2.0.0 genes (scaffold)", level="gene", n=len(hg)),
        gene_stage("G2 Human-GEM genes with DepMap dependency (evaluation universe)", st.symbol),
        gene_stage("G3 G2 genes whose knockout disables >= 1 scaffold reaction", st.symbol[st.struct_candidate])]

per_line = []
for ln in LINES:
    cg = [g for g in CG[ln][0] if g in e2s]
    s = smp[smp.line == ln]
    r = dict(line=ln, assembly=s.assembly.iloc[0], context_model_genes=CG[ln][1], depmap_lethal_all=int(lethal.loc[ln].sum()),
             L1_n=len(cg), L1_lethal=int(lethal.loc[ln, [e2s[g] for g in cg]].sum()))
    for k, d in (("L2", s), ("L3", s[s.kept]), ("L4", s[s.kept & s.effect])):
        r[f"{k}_n"], r[f"{k}_lethal"] = len(d), int((d.dependency >= 0.5).sum())
    per_line.append(r)
pl = pd.DataFrame(per_line); save_tsv(pl, f"{OUT}/t2_per_line.tsv", "per cell line counts per stage (L1-L4) and essential genes among them (lethal_* columns = DepMap essential, code label)")

names = {"L1": "L1 G2 genes in the line's context model",
         "L2": "L2 knockouts screened by MOMA (disable >= 1 context reaction), DepMap-labelled",
         "L3": "L3 L2 with a MOMA solution",
         "L4": "L4 L3 knockouts that change the fluxome (GNN set)"}
for k, nm in names.items():
    if k == "L1":
        pairs = [(ln, e2s[g]) for ln in LINES for g in CG[ln][0] if g in e2s]
    else:
        d = smp if k == "L2" else smp[smp.kept] if k == "L3" else smp[smp.kept & smp.effect]
        pairs = list(zip(d.line, d.symbol))
    pdf = pd.DataFrame(pairs, columns=["line", "symbol"])
    pdf["lethal"] = [bool(lethal.at[l, s]) for l, s in pairs]
    gl = pdf.groupby("symbol").lethal
    rows.append(dict(stage=nm, level="line-gene pairs", n=len(pdf), n_per_line_median=pl[f"{k}_n"].median(),
                     n_per_line_min=pl[f"{k}_n"].min(), n_per_line_max=pl[f"{k}_n"].max(),
                     n_unique_genes=pdf.symbol.nunique(), lethal_pairs=int(pdf.lethal.sum()),
                     lethal_per_line_median=pl[f"{k}_lethal"].median(), lethal_per_line_min=pl[f"{k}_lethal"].min(),
                     lethal_per_line_max=pl[f"{k}_lethal"].max(),
                     lethal_share_of_line_depmap_lethal_median=(pl[f"{k}_lethal"] / pl.depmap_lethal_all).median(),
                     genes_lethal_ge1_line=int(gl.any().sum())))
t1 = pd.DataFrame(rows); save_tsv(t1, f"{OUT}/t1_stages.tsv", "stage counts G0-G3 (gene level) and L1-L4 (cell line-gene pairs); lethal_* columns = DepMap essential (code label)")
say(f"# gene counts; DepMap 24Q4 CRISPRGeneDependency >= 0.5 = essential (code label lethal); {len(LINES)} lines; Human-GEM v2.0.0")
say(f"# dataset {DS}; moma_root {MR}; out {OUT}; {DATE}")
say(f"# samples.tsv rows {len(smp)}; status optimal {int((smp.status == 'optimal').sum())}; excluded by ruling {int((smp.excluded_reason != '').sum())}; "
    f"kept (L3) {int(smp.kept.sum())}; L4 {int((smp.kept & smp.effect).sum())}; L4 genes {smp[smp.kept & smp.effect].symbol.nunique()}")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 80)
say(t1.T.to_string())
say("\nper line context_model_genes (all genes incl. those outside G2): median "
    f"{pl.context_model_genes.median():.0f} ({pl.context_model_genes.min()}-{pl.context_model_genes.max()})")
