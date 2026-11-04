"""Gene lists of the patient POC: drug-target genes after the structural filter, and seeded control genes (2026-11-03).
Spec: result/external_plan_20261103/design_poc_addendum_20261103.md section 3 (user ruling 2026-11-03 Q4: the 17 include_poc genes
of drug_targets_metabolic_20261103.tsv; TYMP, DPYD, UMPS stay out).
Structural filter: keep a gene only if its knockout disables >= 1 reaction (GPR evaluation, rule of 58:106-108:
r.gpr.eval({g}) is False) in EVERY patient context model (<models-root>/<sample>/<sample>.repaired.xml; scripts 22 + 56).
Control pool: Human-GEM genes not in any row of the drug table (any target_ensg, include_poc yes or no) that pass the same filter.
Bins: number of reactions the knockout disables in the Human-GEM scaffold (data/scaffold/Human-GEM.xml): 1, 2-3, 4-10, > 10.
Draw: n controls with numpy default_rng(seed), allocated across bins in proportion to the surviving drug-target genes (largest
remainder rounding); within a bin rng.choice over the pool sorted by ENSG, without replacement; bins drawn in the order 1, 2-3,
4-10, > 10, and controls.tsv keeps that drawn order (draw_order).
Outputs (--out, all with # header): drug_targets.tsv (17 genes, per-model disabled counts, kept, reason), controls.tsv,
control_pool.tsv, poc_genes.txt (kept drug-target genes then controls, one ENSG per line). Refuses to overwrite.
Env: crcgem.
"""
import argparse, os
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, cobra

ROOT = Path("/data/workspace/cdgu/Winery/seeding/CRC_GEM")
POC17 = ["TYMS", "DHFR", "GART", "RRM1", "RRM2", "ADA", "ODC1", "CYP17A1", "CYP19A1", "GLS", "DHODH", "FASN", "IDO1", "MAT2A",
         "HMGCR", "PTGS2", "PTGS1"]   # user ruling 2026-11-03 Q4 (= the table's "# POC drug-target gene set" line)
ap = argparse.ArgumentParser()
ap.add_argument("--drug-table", required=True); ap.add_argument("--models-root", required=True)
ap.add_argument("--seed", type=int, required=True); ap.add_argument("--n-controls", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--genes-file", help="drug-target symbols, one per line, replaces POC17 (user ruling 2026-11-03: 8 genes; a gene "
                "absent from a model, or disabling no reaction there, is skipped for that model only)")
ap.add_argument("--n-cut", type=int, default=0, help="cut rule of design_poc_unit_cut_20261103.md section 3 (10)")
args = ap.parse_args()
ab = lambda p: Path(p) if str(p).startswith("/") else ROOT / p
OUT = ab(args.out)
assert not (OUT / "poc_genes.txt").exists(), f"{OUT}/poc_genes.txt exists (no overwrite)"
OUT.mkdir(parents=True, exist_ok=True)

dt = pd.read_csv(ab(args.drug_table), sep="\t", comment="#", dtype=str)
pairs = {}
for gs, es in zip(dt.target_genes, dt.target_ensg):
    syms = [s.strip().split(" ")[0] for s in gs.split(";")]; ens = [e.strip() for e in es.split(";")]
    assert len(syms) == len(ens), (gs, es)
    for s, e in zip(syms, ens):
        assert pairs.get(s, e) == e, f"{s}: two ENSG"; pairs[s] = e
table_ensg = set(pairs.values())
hdr_line = [l for l in open(ab(args.drug_table)) if l.startswith("# POC drug-target gene set")][0]
assert all(s in hdr_line for s in POC17) and all(s in pairs for s in POC17)
if args.genes_file:
    POC17 = [l.strip() for l in open(ab(args.genes_file)) if l.strip() and not l.startswith("#")]
    assert all(s in pairs for s in POC17), POC17
drug = {pairs[s]: s for s in POC17}

mroot = ab(args.models_root)
samples = sorted(d for d in os.listdir(mroot) if (mroot / d / f"{d}.repaired.xml").exists())
assert len(samples) == 6, f"expected 6 patient models, found {samples}"


def disabled_counts(model):
    return {g.id: sum(1 for r in g.reactions if r.gpr.eval({g.id}) is False) for g in model.genes}


cnt = {}
for s in samples:
    cnt[s] = disabled_counts(cobra.io.read_sbml_model(str(mroot / s / f"{s}.repaired.xml")))
scaf = cobra.io.read_sbml_model(str(ROOT / "data/scaffold/Human-GEM.xml"))
scnt = disabled_counts(scaf)
passes = lambda g: all(cnt[s].get(g, 0) >= 1 for s in samples)
# --genes-file: a drug gene is kept if its knockout disables >= 1 reaction in >= 1 model (per-model skip); controls keep the all-model rule
passes_drug = (lambda g: any(cnt[s].get(g, 0) >= 1 for s in samples)) if args.genes_file else passes


def bin_of(n):
    return None if n < 1 else "1" if n == 1 else "2-3" if n <= 3 else "4-10" if n <= 10 else ">10"


BINS = ["1", "2-3", "4-10", ">10"]
rows = []
for e, sym in drug.items():
    per = {s: cnt[s].get(e, None) for s in samples}
    miss = [s for s in samples if e not in cnt[s]]; zero = [s for s in samples if cnt[s].get(e, 0) == 0 and e in cnt[s]]
    reason = "; ".join(x for x in (f"not in model: {miss}" if miss else "",
                                   f"disables no reaction (isozyme/OR) in: {zero}" if zero else "") if x)
    rows.append(dict(ensg=e, symbol=sym, kept=passes_drug(e), scaffold_n_disabled=scnt.get(e, 0), bin=bin_of(scnt.get(e, 0)),
                     skipped_in_models=reason, **{f"n_disabled_{s}": per[s] for s in samples}))
D = pd.DataFrame(rows)
kept = D[D.kept]
assert kept.bin.notna().all()

pool = sorted(g.id for g in scaf.genes if g.id not in table_ensg and passes(g.id))
P = pd.DataFrame(dict(ensg=pool, scaffold_n_disabled=[scnt[g] for g in pool]))
P["bin"] = P.scaffold_n_disabled.map(bin_of)
n_unbinned = int(P.bin.isna().sum()); P = P[P.bin.notna()]
def alloc_of(n):
    share = kept.bin.value_counts().reindex(BINS, fill_value=0) / len(kept) * n
    a = np.floor(share).astype(int)
    for b in (share - a).sort_values(ascending=False, kind="stable").index[: n - a.sum()]:
        a[b] += 1
    assert a.sum() == n
    return a


alloc = alloc_of(args.n_controls)
rng = np.random.default_rng(args.seed); C = []
for b in BINS:
    cand = P[P.bin == b].ensg.sort_values().tolist()
    assert alloc[b] <= len(cand), f"bin {b}: pool {len(cand)} < {alloc[b]}"
    for g in rng.choice(cand, size=alloc[b], replace=False):
        C.append(dict(draw_order=len(C) + 1, ensg=g, bin=b, scaffold_n_disabled=scnt[g],
                      **{f"n_disabled_{s}": cnt[s][g] for s in samples}))
C = pd.DataFrame(C)
assert C.ensg.is_unique and not set(C.ensg) & table_ensg and not set(C.ensg) & set(kept.ensg)
# cut rule (design_poc_unit_cut_20261103.md section 3): per bin keep the first alloc_cut[b] controls in draw order; no new draw
alloc_cut = alloc_of(args.n_cut) if args.n_cut else None
if args.n_cut:
    assert (alloc_cut <= alloc).all(), f"alloc{args.n_cut} {alloc_cut.to_dict()} > alloc{args.n_controls} {alloc.to_dict()}: ask the user"
    C[f"in_cut{args.n_cut}"] = C.groupby("bin").cumcount() < C.bin.map(alloc_cut)
    assert C[f"in_cut{args.n_cut}"].sum() == args.n_cut
    assert C[C[f"in_cut{args.n_cut}"]].bin.value_counts().reindex(BINS, fill_value=0).equals(alloc_cut)

HDR = [f"# source: script_python/143_poc_gene_lists.py; drug table {args.drug_table}; models {args.models_root}/<sample>/<sample>.repaired.xml "
       f"({', '.join(samples)}); scaffold data/scaffold/Human-GEM.xml",
       f"# rule: knockout disables >= 1 reaction (GPR eval, 58:106-108) in all {len(samples)} models; controls from Human-GEM genes in no "
       f"drug-table row ({len(table_ensg)} ENSG excluded), binned by scaffold disabled count {BINS}; default_rng({args.seed}); n_controls {args.n_controls}; "
       f"allocation {alloc.to_dict()} (largest remainder, proportional to kept drug targets)"
       + (f"; cut rule (design_poc_unit_cut_20261103.md section 3): alloc{args.n_cut} {alloc_cut.to_dict()}, first alloc{args.n_cut}[bin] "
          f"controls of each bin in draw order (column in_cut{args.n_cut})" if args.n_cut else ""),
       f"# counts: drug targets {len(POC17)} (genes file {args.genes_file}; drug-gene rule: {'>= 1 model, per-model skip' if args.genes_file else 'all models'}), kept {len(kept)}, dropped {len(POC17) - len(kept)}; control pool {len(P)} (excluded from pool for scaffold count 0: {n_unbinned})",
       f"# date: {datetime.now().isoformat(timespec='seconds')} KST"]
for name, df in (("drug_targets.tsv", D), ("controls.tsv", C), ("control_pool.tsv", P)):
    with open(OUT / name, "w") as f:
        f.write("\n".join(HDR) + "\n"); df.to_csv(f, sep="\t", index=False)
with open(OUT / "poc_genes.txt", "w") as f:
    f.write("\n".join(HDR + ["# order: kept drug-target genes (table order), then controls in draw order"]) + "\n")
    f.write("\n".join(kept.ensg.tolist() + C.ensg.tolist()) + "\n")
if args.n_cut:
    with open(OUT / f"poc_genes_cut{args.n_cut}.txt", "w") as f:
        f.write("\n".join(HDR + [f"# order: kept drug-target genes (table order), then in_cut{args.n_cut} controls in draw order"]) + "\n")
        f.write("\n".join(kept.ensg.tolist() + C[C[f"in_cut{args.n_cut}"]].ensg.tolist()) + "\n")
print(D[["symbol", "ensg", "kept", "bin", "skipped_in_models"]].to_string()); print(C.to_string()); print(HDR[1]); print(HDR[2])
