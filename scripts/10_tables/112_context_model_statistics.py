"""Context-specific model statistics for the Methods section (user request 2026-10-30).

Counts genes, reactions, metabolites, compartments and exchange reactions of every context-specific
model at each stage of its construction, for the 59 DepMap colorectal cell lines and, in a separate
block, the 7 cell lines of the external validation set of non-colorectal cell lines.

Stages (all measured, nothing estimated):
  S0  Human-GEM scaffold                      data/scaffold/Human-GEM.xml
  S1  tINIT model                             <models>/<line>/<line>.xml            (script 22)
  S2  after essential-task repair             <models>/<line>/<line>.repaired.xml   (script 56)
  S3  model as screened                       S2 + per-cell-line requirement-cap gap-fill (script 62,
                                              <caps>/gapfill.json) + medium gap-fill recorded in
                                              <moma>/<line>/meta.json gapfill_added (script 58)
  genes screened                              genes whose knockout disables >= 1 reaction of the S3
                                              model (GPR rule of script 58), vs meta.json n_screened
                                              and the rows of summary.tsv
Independent paths per count: (A) cobra SBML parse; (B) raw XML tag count (regex on the file, no cobra);
(C) pipeline records (extraction_summary.tsv, kept_reactions.txt, task_repair.json, meta.json,
summary.tsv). Disagreements are listed in raw/consistency_checks.tsv, never silently dropped.
Medium: exchange lower bounds of the cell line's assembly (script 57) plus the per-cell-line
requirement caps (script 62), exactly as applied in script 58; an exchange is "open for uptake" when
its lower bound < 0 after this step. Blocked reactions (optional, --blocked): cobra
find_blocked_reactions (FVA, zero_cutoff 1e-9) on the S3 model under its medium, Gurobi 1 thread.

Usage: ~/anaconda3/envs/crcgem/bin/python 112_context_model_statistics.py [--jobs 4] [--blocked]
Env: crcgem. At most --jobs processes (default 4; 24 MOMA jobs share the host).
Output (never overwrites: an existing file gets a __YYYYMMDD_HHMMSS suffix on the new one):
  submission_packages/20261029/Tables/Table_context_model_statistics/{*.tsv,*.csv,README.md,raw/}
  submission_packages/20261029/Figures/Supplementary/FigureS_context_model_statistics/{panel_*.{html,svg,png,jpeg},
                                                                        panel_*_data.tsv}
"""
import argparse, json, os, re, sys, time, platform
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/workspace/cdgu/Winery/seeding/CRC_GEM")
SCAFFOLD = ROOT / "data/scaffold/Human-GEM.xml"
BIOMASS = "MAR13082"
SETS = {
    "colorectal": dict(models=ROOT / "result/tinit_localT2_20261027/models", moma=ROOT / "result/moma_20261027",
                       caps=ROOT / "data/medium/uptake_caps/line_requirement_caps",
                       lines=[l.strip() for l in open(ROOT / "result/tinit_localT2_20261027/lines.txt") if l.strip()],
                       label="59 colorectal cell lines (training set)"),
    "external": dict(models=ROOT / "result/oob_20261028/models", moma=ROOT / "result/oob_20261028/moma",
                     caps=ROOT / "result/oob_20261028/line_requirement_caps",
                     lines=sorted(os.listdir(ROOT / "result/oob_20261028/moma")),
                     label="7 cell lines of the external validation set of non-colorectal cell lines"),
}
ASSEMBLY_LB = ROOT / "data/medium/uptake_caps/exchange_lower_bounds_by_assembly.tsv"
LINE_MEDIUM = ROOT / "data/medium/line_medium_map.tsv"
EXT_ASSEMBLY = ROOT / "result/oob_20261028/lines_assembly.tsv"
PK = ROOT / "submission_packages/20261029"
TAB = PK / "Tables/Table_context_model_statistics"
FIG = PK / "Figures/Supplementary/FigureS_context_model_statistics"
STAGES = [("S0", "Human-GEM scaffold"), ("S1", "tINIT model"), ("S2", "after essential-task repair"),
          ("S3", "model as screened (after gap-fill)")]
STAGE_NAME = dict(STAGES)

TAG = re.compile(rb"<(reaction|species|fbc:geneProduct|compartment) ")


def xml_tag_counts(path):
    """Path B: count SBML element tags in the raw file; independent of cobra."""
    c = {"reaction": 0, "species": 0, "fbc:geneProduct": 0, "compartment": 0}
    with open(path, "rb") as fh:
        for chunk_line in fh:
            for m in TAG.finditer(chunk_line):
                c[m.group(1).decode()] += 1
    return dict(n_reactions=c["reaction"], n_metabolites=c["species"], n_genes=c["fbc:geneProduct"],
                n_compartments=c["compartment"])


def cobra_counts(m):
    import cobra
    comp_used = sorted({x.compartment for x in m.metabolites})
    per_comp = pd.Series([x.compartment for x in m.metabolites]).value_counts().to_dict()
    return dict(n_reactions=len(m.reactions), n_metabolites=len(m.metabolites), n_genes=len(m.genes),
                n_genes_with_reaction=sum(1 for g in m.genes if len(g.reactions) > 0),
                n_metabolites_with_reaction=sum(1 for x in m.metabolites if len(x.reactions) > 0),
                n_reactions_with_gpr=sum(1 for r in m.reactions if r.gene_reaction_rule.strip()),
                n_compartments=len(comp_used), compartments=";".join(comp_used),
                n_exchange_reactions=len(m.exchanges),
                n_transport_reactions=sum(1 for r in m.reactions if len({x.compartment for x in r.metabolites}) > 1),
                n_reversible=sum(1 for r in m.reactions if r.reversibility),
                metabolites_per_compartment=json.dumps(per_comp, sort_keys=True))


def screened_genes(m):
    """Exactly the rule of script 58 (lines 97-99): genes whose knockout disables >= 1 reaction."""
    return [g.id for g in m.genes if any(r.gpr.eval({g.id}) is False for r in g.reactions)]


def medium_bounds(assembly, caps_dir, line):
    lb = pd.read_csv(ASSEMBLY_LB, sep="\t", comment="#")
    lb = lb[lb.assembly_id == assembly].set_index("exchange_rxn_id").lower_bound.to_dict()
    ov = pd.read_csv(caps_dir / "lower_bound_overrides.tsv", sep="\t", comment="#")
    ov = ov[ov.line == line]
    lb.update(dict(zip(ov.exchange_rxn_id, ov.lower_bound_new)))
    return lb


def worker(job):
    import cobra
    from cobra.flux_analysis import find_blocked_reactions
    set_name, line, assembly, do_blocked = job
    S = SETS[set_name]
    t0 = time.time()
    row = dict(set=set_name, cell_line=line, assembly=assembly)
    p1 = S["models"] / line / f"{line}.xml"
    p2 = S["models"] / line / f"{line}.repaired.xml"
    # ---- path B: raw XML tags
    for stage, p in (("S1", p1), ("S2", p2)):
        for k, v in xml_tag_counts(p).items():
            row[f"{stage}_xml_{k}"] = v
    # ---- path C: records
    es = pd.read_csv(S["models"] / line / "extraction_summary.tsv", sep="\t").iloc[0]
    row["S1_rec_n_reactions"] = int(es.n_kept); row["S1_rec_n_genes"] = int(es.n_genes)
    row["S1_rec_kept_reactions_txt"] = sum(1 for l in open(S["models"] / line / f"{line}.kept_reactions.txt") if l.strip())
    tr = json.load(open(S["models"] / line / "task_repair.json"))
    row["S2_rec_n_failed_tinit"] = tr["n_failed_tinit"]; row["S2_rec_n_added_reactions"] = tr["n_added_reactions"]
    row["S2_rec_n_failed_after_repair"] = tr["n_failed_after_repair"]; row["S2_rec_n_reactions"] = tr["n_reactions_repaired"]
    meta = json.load(open(S["moma"] / line / "meta.json"))
    row["S3_rec_n_reactions"] = meta["n_reactions"]; row["S3_rec_n_genes"] = meta["n_genes"]
    row["S3_rec_n_screened"] = meta["n_screened"]; row["S3_rec_n_optimal"] = meta["n_optimal"]
    row["S3_rec_status"] = meta["status"]; row["S3_rec_wt_growth"] = meta["wt_growth"]
    gf62 = json.load(open(S["caps"] / "gapfill.json")).get(line, [])
    gf58 = meta.get("gapfill_added", [])
    row["S3_gapfill_script62_n"] = len(gf62); row["S3_gapfill_script62_ids"] = ";".join(gf62)
    row["S3_gapfill_script58_medium_n"] = len(gf58); row["S3_gapfill_script58_medium_ids"] = ";".join(gf58)
    row["S3_caps_raised_n"] = len(meta.get("caps_raised", {}))
    row["S3_rec_summary_tsv_rows"] = len(pd.read_csv(S["moma"] / line / "summary.tsv", sep="\t"))
    # ---- path A: cobra
    m1 = cobra.io.read_sbml_model(str(p1))
    c1 = cobra_counts(m1)
    for k, v in c1.items():
        row[f"S1_{k}"] = v
    s1_rxn, s1_gene, s1_met = {r.id for r in m1.reactions}, {g.id for g in m1.genes}, {x.id for x in m1.metabolites}
    del m1
    m = cobra.io.read_sbml_model(str(p2))
    c2 = cobra_counts(m)
    for k, v in c2.items():
        row[f"S2_{k}"] = v
    s2_rxn = {r.id for r in m.reactions}
    row["S2_added_vs_S1_measured"] = len(s2_rxn - s1_rxn); row["S2_removed_vs_S1_measured"] = len(s1_rxn - s2_rxn)
    # ---- S3: exactly as script 58 builds the screened model
    to_add = [x for x in gf62 if x not in m.reactions] + [x for x in gf58 if x not in m.reactions]
    if to_add:
        scaf = cobra.io.read_sbml_model(str(SCAFFOLD))
        m.add_reactions([scaf.reactions.get_by_id(x).copy() for x in to_add])
        del scaf
    row["S3_added_vs_S2_measured"] = len(to_add)
    lb = medium_bounds(assembly, S["caps"], line)
    for r in m.exchanges:
        r.lower_bound = lb.get(r.id, 0.0)
    c3 = cobra_counts(m)
    for k, v in c3.items():
        row[f"S3_{k}"] = v
    ex_ids = {r.id for r in m.exchanges}
    open_up = [r.id for r in m.exchanges if r.lower_bound < 0]
    row["S3_n_exchange_open_uptake"] = len(open_up)
    row["S3_n_exchange_open_uptake_unbounded"] = sum(1 for r in m.exchanges if r.lower_bound <= -1000)
    row["S3_n_exchange_open_uptake_capped"] = sum(1 for r in m.exchanges if -1000 < r.lower_bound < 0)
    row["S3_n_medium_entries_assembly_plus_caps"] = sum(1 for v in lb.values() if v < 0)
    row["S3_n_medium_entries_absent_from_model"] = sum(1 for k, v in lb.items() if v < 0 and k not in ex_ids)
    row["S3_n_exchange_closed_for_uptake"] = len(ex_ids) - len(open_up)
    row["S3_exchange_open_uptake_ids"] = ";".join(sorted(open_up))
    sg = screened_genes(m)
    row["S3_n_screened_measured"] = len(sg)
    row["S3_n_reactions_disabled_by_any_screened_gene"] = len({r.id for g in sg for r in m.genes.get_by_id(g).reactions
                                                               if r.gpr.eval({g}) is False})
    # WT growth check in the medium (same objective as script 58)
    m.solver = "gurobi"; m.solver.configuration.verbosity = 0; m.solver.problem.Params.Threads = 1
    m.objective = m.reactions.get_by_id(BIOMASS)
    wt = m.slim_optimize(); row["S3_wt_growth_measured"] = float(wt) if wt == wt else float("nan")
    if do_blocked:
        tb = time.time()
        blocked = find_blocked_reactions(m, open_exchanges=False, processes=1, zero_cutoff=1e-9)
        row["S3_n_blocked_reactions_medium"] = len(blocked); row["S3_blocked_seconds"] = round(time.time() - tb, 1)
    sets = dict(S1_rxn=sorted(s1_rxn), S1_gene=sorted(s1_gene), S1_met=sorted(s1_met),
                S3_rxn=sorted(r.id for r in m.reactions), S3_gene=sorted(g.id for g in m.genes),
                S3_met=sorted(x.id for x in m.metabolites))
    row["seconds"] = round(time.time() - t0, 1)
    print(f"{set_name} {line} S1 {c1['n_reactions']}/{c1['n_genes']}/{c1['n_metabolites']} "
          f"S2 {c2['n_reactions']} S3 {c3['n_reactions']}/{c3['n_genes']}/{c3['n_metabolites']} "
          f"screened {len(sg)} (meta {meta['n_screened']}) open uptake {len(open_up)} {row['seconds']}s", flush=True)
    return row, sets


def safe_path(p: Path) -> Path:
    """Never overwrite: existing target -> new file carries a timestamp suffix."""
    if not p.exists():
        return p
    return p.with_name(f"{p.stem}__{datetime.now():%Y%m%d_%H%M%S}{p.suffix}")


def write_table(df, path: Path, header_lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext, sep in ((".tsv", "\t"), (".csv", ",")):
        p = safe_path(path.with_suffix(ext))
        with open(p, "w") as fh:
            for h in header_lines:
                fh.write(f"# {h}\n")
            df.to_csv(fh, sep=sep, index=False)
        print("wrote", p)


def consistency(df):
    checks = [
        ("S1 reactions: cobra = xml tags", "S1_n_reactions", "S1_xml_n_reactions"),
        ("S1 reactions: cobra = extraction_summary n_kept", "S1_n_reactions", "S1_rec_n_reactions"),
        ("S1 reactions: cobra = kept_reactions.txt", "S1_n_reactions", "S1_rec_kept_reactions_txt"),
        ("S1 genes: cobra = xml geneProducts", "S1_n_genes", "S1_xml_n_genes"),
        ("S1 genes: cobra = extraction_summary n_genes", "S1_n_genes", "S1_rec_n_genes"),
        ("S1 metabolites: cobra = xml species", "S1_n_metabolites", "S1_xml_n_metabolites"),
        ("S2 reactions: cobra = xml tags", "S2_n_reactions", "S2_xml_n_reactions"),
        ("S2 reactions: cobra = task_repair n_reactions_repaired", "S2_n_reactions", "S2_rec_n_reactions"),
        ("S2 genes: cobra = xml geneProducts", "S2_n_genes", "S2_xml_n_genes"),
        ("S2 metabolites: cobra = xml species", "S2_n_metabolites", "S2_xml_n_metabolites"),
        ("S2 added reactions: measured S2-S1 = task_repair n_added_reactions", "S2_added_vs_S1_measured", "S2_rec_n_added_reactions"),
        ("S3 reactions: cobra = meta.json n_reactions", "S3_n_reactions", "S3_rec_n_reactions"),
        ("S3 genes: cobra = meta.json n_genes", "S3_n_genes", "S3_rec_n_genes"),
        ("S3 added reactions: measured S3-S2 = gapfill.json + gapfill_added", "S3_added_vs_S2_measured", "S3_gapfill_total_n"),
        ("genes screened: GPR rule recomputed = meta.json n_screened", "S3_n_screened_measured", "S3_rec_n_screened"),
        ("genes screened: meta.json n_screened = summary.tsv rows", "S3_rec_n_screened", "S3_rec_summary_tsv_rows"),
    ]
    rows = []
    for name, a, b in checks:
        d = (df[a] - df[b])
        bad = df.loc[d != 0, "cell_line"].tolist()
        rows.append(dict(check=name, column_a=a, column_b=b, n_models=len(df), n_equal=int((d == 0).sum()),
                         n_different=len(bad), models_different=";".join(bad),
                         max_abs_difference=int(d.abs().max())))
    wt = (df["S3_wt_growth_measured"] - df["S3_rec_wt_growth"]).abs()
    rows.append(dict(check="S3 WT growth in medium: re-solved = meta.json wt_growth (|diff| <= 1e-6)",
                     column_a="S3_wt_growth_measured", column_b="S3_rec_wt_growth", n_models=len(df),
                     n_equal=int((wt <= 1e-6).sum()), n_different=int((wt > 1e-6).sum()),
                     models_different=";".join(df.loc[wt > 1e-6, "cell_line"]), max_abs_difference=float(wt.max())))
    return pd.DataFrame(rows)


def med_range(s):
    s = pd.Series(s).dropna()
    if s.empty:
        return "", np.nan, np.nan, np.nan
    return f"{s.median():,.0f} ({s.min():,.0f}-{s.max():,.0f})", s.median(), s.min(), s.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--blocked", action="store_true", help="also compute blocked reactions (FVA) per S3 model")
    ap.add_argument("--only", nargs="*", help="restrict to these cell lines (dry test)")
    ap.add_argument("--out-root", help="redirect Tables/ and Figures/ output (dry test); default = submission package")
    args = ap.parse_args()
    global TAB, FIG
    if args.out_root:
        TAB = Path(args.out_root) / "Tables/Table_context_model_statistics"; FIG = Path(args.out_root) / "Figures/Supplementary/FigureS_context_model_statistics"
    import cobra, plotly
    t0 = time.time()
    today = datetime.now().strftime("%Y-%m-%d")
    versions = (f"python {platform.python_version()}, cobra {cobra.__version__}, pandas {pd.__version__}, "
                f"plotly {plotly.__version__}, Gurobi " + ".".join(map(str, __import__('gurobipy').gurobi.version())))

    # ---- scaffold (S0)
    scaf = cobra.io.read_sbml_model(str(SCAFFOLD))
    s0 = cobra_counts(scaf); s0x = xml_tag_counts(SCAFFOLD)
    s0_sets = dict(rxn={r.id for r in scaf.reactions}, gene={g.id for g in scaf.genes}, met={x.id for x in scaf.metabolites})
    s0_screen = len(screened_genes(scaf))
    s0_comp = json.loads(s0["metabolites_per_compartment"])
    print("S0 scaffold", s0, s0x, "genes whose knockout disables >=1 scaffold reaction", s0_screen, flush=True)
    del scaf

    # ---- jobs
    lm = pd.read_csv(LINE_MEDIUM, sep="\t").set_index("ModelID")
    ext = pd.read_csv(EXT_ASSEMBLY, sep="\t", header=None, names=["line", "assembly"]).set_index("line").assembly
    jobs = []
    for set_name, S in SETS.items():
        for line in S["lines"]:
            if args.only and line not in args.only:
                continue
            assembly = lm.loc[line, "assembly_id"] if set_name == "colorectal" else ext.loc[line]
            jobs.append((set_name, line, assembly, args.blocked))
    with Pool(min(args.jobs, len(jobs))) as pool:
        results = pool.map(worker, jobs, chunksize=1)
    df = pd.DataFrame([r for r, _ in results])
    df["S3_gapfill_total_n"] = df["S3_gapfill_script62_n"] + df["S3_gapfill_script58_medium_n"]
    sets = {(r["set"], r["cell_line"]): s for r, s in results}

    TAB.mkdir(parents=True, exist_ok=True); (TAB / "raw").mkdir(exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    src = [f"source models: {SETS['colorectal']['models']} (scripts 22, 56), external {SETS['external']['models']}",
           f"source screens: {SETS['colorectal']['moma']} (script 58), external {SETS['external']['moma']}",
           f"gap-fill records: {SETS['colorectal']['caps']}/gapfill.json, {SETS['external']['caps']}/gapfill.json; medium: {ASSEMBLY_LB} + lower_bound_overrides.tsv",
           f"scaffold: {SCAFFOLD} (Human-GEM, model id HumanGEM, v2.0.0 per documentation/methods.md)",
           f"tool: script_python/112_context_model_statistics.py; {versions}; date {today}",
           f"n: {int((df.set == 'colorectal').sum())} colorectal cell lines + {int((df.set == 'external').sum())} external validation cell lines; exclusions: none; all MOMA meta.json status = done",
           "blocked reactions: " + ("cobra find_blocked_reactions, FVA, zero_cutoff 1e-9, S3 model in its medium, Gurobi 1 thread" if args.blocked else "not computed"),
           "note: result/moma_20261030_tolpatch re-solves trace-flux knockouts of the same models at a tighter tolerance; model composition and the screened gene set are those of result/moma_20261027"]

    # ---- per-model table (raw)
    write_table(df, TAB / "raw/per_model_statistics", src + ["one row per cell line; S1/S2/S3 prefixes = stage; _xml_ = raw tag count (path B); _rec_ = pipeline record (path C); unprefixed = cobra (path A)"])

    # ---- consistency
    cons = consistency(df)
    write_table(cons, TAB / "raw/consistency_checks", src + ["each row: one equality check between two independent paths over all 66 models"])
    print(cons[["check", "n_equal", "n_different", "models_different"]].to_string())

    # ---- summary table: stage x metric x set
    metrics = [("n_genes", "genes"), ("n_reactions", "reactions"), ("n_metabolites", "metabolites"),
               ("n_compartments", "compartments"), ("n_exchange_reactions", "exchange reactions"),
               ("n_transport_reactions", "transport reactions (>= 2 compartments)"),
               ("n_reactions_with_gpr", "reactions with a gene-protein-reaction rule")]
    rows = []
    for set_name, S in SETS.items():
        d = df[df.set == set_name]
        if d.empty:
            continue
        n = len(d)
        for stage, sname in STAGES:
            for col, mname in metrics:
                if stage == "S0":
                    v = s0[col]
                    rows.append(dict(set=S["label"], n_models=1, stage=stage, stage_name=sname, metric=mname,
                                     median_range=f"{v:,}", median=v, min=v, max=v, scaffold=v))
                else:
                    mr, med, mn, mx = med_range(d[f"{stage}_{col}"])
                    rows.append(dict(set=S["label"], n_models=n, stage=stage, stage_name=sname, metric=mname,
                                     median_range=mr, median=med, min=mn, max=mx, scaffold=s0[col]))
        extra = [("S2", "S2_rec_n_added_reactions", "reactions added by essential-task repair (script 56)"),
                 ("S2", "S2_rec_n_failed_tinit", "essential tasks failed by the tINIT model (of 57)"),
                 ("S2", "S2_rec_n_failed_after_repair", "essential tasks failed after repair"),
                 ("S3", "S3_gapfill_script62_n", "reactions added by the per-cell-line requirement-cap gap-fill (script 62)"),
                 ("S3", "S3_gapfill_script58_medium_n", "reactions added by the medium gap-fill of script 58"),
                 ("S3", "S3_n_exchange_open_uptake", "exchange reactions open for uptake in the cell line's medium"),
                 ("S3", "S3_n_exchange_open_uptake_capped", "of which capped (lower bound > -1000)"),
                 ("S3", "S3_n_exchange_open_uptake_unbounded", "of which unbounded (lower bound -1000)"),
                 ("S3", "S3_n_medium_entries_absent_from_model", "medium entries whose exchange reaction is absent from the model"),
                 ("S3", "S3_n_screened_measured", "genes screened by MOMA (knockout disables >= 1 reaction)"),
                 ("S3", "S3_n_reactions_disabled_by_any_screened_gene", "reactions disabled by >= 1 screened knockout"),
                 ("S3", "S3_rec_n_optimal", "knockouts with a MOMA solution (solver status optimal)")]
        if args.blocked:
            extra.append(("S3", "S3_n_blocked_reactions_medium", "blocked reactions in the cell line's medium (FVA)"))
        for stage, col, mname in extra:
            mr, med, mn, mx = med_range(d[col])
            rows.append(dict(set=S["label"], n_models=n, stage=stage, stage_name=STAGE_NAME[stage], metric=mname,
                             median_range=mr, median=med, min=mn, max=mx,
                             scaffold=(s0_screen if col == "S3_n_screened_measured" else np.nan)))
    summ = pd.DataFrame(rows)
    write_table(summ, TAB / "Table_context_model_statistics",
                src + ["median (min-max) over the cell lines of the set; scaffold column = the Human-GEM value for the same metric (genes screened: genes whose knockout disables >= 1 scaffold reaction)"])

    # ---- overlap (core = in all models of the set, union = in any)
    ov_rows, freq_rows = [], []
    for set_name, S in SETS.items():
        keys = [k for k in sets if k[0] == set_name]
        if not keys:
            continue
        for stage in ("S1", "S3"):
            for ent, ename, s0set in (("rxn", "reactions", s0_sets["rxn"]), ("gene", "genes", s0_sets["gene"]), ("met", "metabolites", s0_sets["met"])):
                lists = [set(sets[k][f"{stage}_{ent}"]) for k in keys]
                core = set.intersection(*lists); union = set.union(*lists)
                ov_rows.append(dict(set=S["label"], n_models=len(keys), stage=stage, stage_name=STAGE_NAME[stage], entity=ename,
                                    scaffold=len(s0set), core_in_all_models=len(core), union_any_model=len(union),
                                    scaffold_in_no_model=len(s0set - union),
                                    median_per_model=float(np.median([len(x) for x in lists])),
                                    core_share_of_median=round(len(core) / np.median([len(x) for x in lists]), 4)))
                if ent == "rxn" and stage == "S3":
                    cnt = pd.Series([r for x in lists for r in x]).value_counts()
                    for rid in sorted(s0set):
                        freq_rows.append(dict(set=set_name, reaction=rid, n_models_present=int(cnt.get(rid, 0)), n_models=len(keys)))
    # 59 + 7 together
    keys = list(sets)
    for ent, ename, s0set in (("rxn", "reactions", s0_sets["rxn"]), ("gene", "genes", s0_sets["gene"]), ("met", "metabolites", s0_sets["met"])):
        lists = [set(sets[k][f"S3_{ent}"]) for k in keys]
        core = set.intersection(*lists); union = set.union(*lists)
        ov_rows.append(dict(set="all 66 cell lines (59 + 7)", n_models=len(keys), stage="S3", stage_name=STAGE_NAME["S3"], entity=ename,
                            scaffold=len(s0set), core_in_all_models=len(core), union_any_model=len(union),
                            scaffold_in_no_model=len(s0set - union), median_per_model=float(np.median([len(x) for x in lists])),
                            core_share_of_median=round(len(core) / np.median([len(x) for x in lists]), 4)))
    ovdf = pd.DataFrame(ov_rows)
    write_table(ovdf, TAB / "Table_context_model_overlap", src + ["core = present in every model of the set; union = present in >= 1 model; scaffold_in_no_model = scaffold entities absent from all models"])
    freq = pd.DataFrame(freq_rows)
    write_table(freq, TAB / "raw/reaction_presence_frequency", src + ["per scaffold reaction: number of S3 models (model as screened) containing it, per set"])
    core59 = sorted(set.intersection(*[set(sets[k]["S3_rxn"]) for k in sets if k[0] == "colorectal"]))
    write_table(pd.DataFrame(dict(reaction=core59)), TAB / "raw/core_reactions_S3_colorectal59", src + ["reactions present in all 59 colorectal S3 models"])

    # ---- compartments
    comp_rows = []
    d59 = df[df.set == "colorectal"]
    per = [json.loads(x) for x in d59["S3_metabolites_per_compartment"]]
    for c in sorted(s0_comp):
        vals = [p.get(c, 0) for p in per]
        mr, med, mn, mx = med_range(vals)
        comp_rows.append(dict(compartment=c, scaffold_metabolites=s0_comp[c], S3_colorectal59_median_range=mr, median=med, min=mn, max=mx))
    write_table(pd.DataFrame(comp_rows), TAB / "raw/compartment_metabolites", src + ["metabolites per compartment: scaffold vs S3 models of the 59 colorectal cell lines (median, min-max)"])

    # ---- README
    readme = f"""# Table: context-specific model statistics by construction stage

{SETS['colorectal']['label']} and, as a separate block, {SETS['external']['label']}.
Stages: S0 Human-GEM scaffold; S1 tINIT model (script 22); S2 after essential-task repair (script 56; reactions of
failed essential tasks added back from the scaffold); S3 model as screened (S2 plus the per-cell-line requirement-cap
gap-fill of script 62 and the medium gap-fill of script 58, with the cell line's medium applied as in script 58).
Genes screened: genes whose knockout disables at least one reaction of the S3 model (script 58 rule).
Exchange reactions open for uptake: exchange reactions of the S3 model with lower bound < 0 after the medium step.
Counts are measured by cobra {cobra.__version__} (path A) and checked against raw SBML tag counts (path B) and the
pipeline records extraction_summary.tsv, kept_reactions.txt, task_repair.json, meta.json, summary.tsv (path C);
see raw/consistency_checks.tsv. Every value is median (min-max) over the cell lines of the set.
Files: Table_context_model_statistics.tsv/.csv (summary), Table_context_model_overlap.tsv/.csv (core and union
across cell lines), raw/per_model_statistics (one row per cell line, all paths), raw/consistency_checks,
raw/reaction_presence_frequency, raw/core_reactions_S3_colorectal59, raw/compartment_metabolites.
Blocked reactions: {'computed (FVA, see header)' if args.blocked else 'not computed'}.
Built by script_python/112_context_model_statistics.py on {today}; {versions}.
"""
    p = safe_path(TAB / "README.md"); open(p, "w").write(readme); print("wrote", p)

    # ---- figures
    make_figures(df, freq, s0, today, versions, src)
    print(f"done in {time.time() - t0:.0f}s")


def make_figures(df, freq, s0, today, versions, src):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # 2026-11-02: one plot per panel file (panels a..l), project palette; builders shared with script 131 in module/crcgem/figpanels.py.
    # The old facetted panels a..d are retired (backup _backup/20261102_figures/Figures/Figure_context_model_statistics). Output folder FIG
    # (default submission_packages/20261029/Figures/Supplementary/FigureS_context_model_statistics). Redraw without this script's heavy pass:
    # script_python/131_build_figures.py --only s2 (reads the frozen panel data).
    import sys
    sys.path.insert(0, str(ROOT / "module"))
    from crcgem import figpanels as fp
    df = df.copy()
    nn = df.set.value_counts().to_dict()
    labels = {"colorectal": f"colorectal cell line (n={nn.get('colorectal', 0)})",
              "external": f"external validation cell line (n={nn.get('external', 0)})"}
    df["set_label"] = df.set.map(labels)

    # panel data a: counts by stage (long table)
    long = []
    for _, r in df.iterrows():
        for stage, sname in STAGES[1:]:
            for col, ename in (("n_genes", "genes"), ("n_reactions", "reactions"), ("n_metabolites", "metabolites")):
                long.append(dict(set=r.set, set_label=r.set_label, cell_line=r.cell_line, stage=stage, stage_name=sname,
                                 entity=ename, count=r[f"{stage}_{col}"], scaffold=s0[col]))
    long = pd.DataFrame(long)
    f59 = freq[freq.set == "colorectal"].copy()
    f59 = f59[f59.n_models_present > 0].sort_values("n_models_present", ascending=False).reset_index(drop=True); f59["rank"] = np.arange(1, len(f59) + 1)
    fp.context_model_panels(long, df, f59, str(FIG), src + ["independent verification: NONE (script 112 output, model files of record)"])
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
