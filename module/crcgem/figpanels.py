"""Panel builders shared by script_python/110_data_statistics.py, 112_context_model_statistics.py and 131_build_figures.py.
One plot per panel file (user rule 2026-11-02). Style and saving: crcgem.figstyle. Labels: documentation/terminology.md terms only.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from . import figstyle as fs

DATASET_SHORT = {"colorectal": "colorectal cell lines<br>(training and<br>cross-validation)",
                 "external": "external validation set<br>of non-colorectal<br>cell lines"}


def _dataset_key(s):
    return "external" if str(s).startswith("external") else "colorectal"


def _strip(fig, x, y, color, symbol, name, text, jitter=0.25, showlegend=True, kind="point", seed=0):
    rng = np.random.default_rng(seed)
    xj = np.asarray(x, dtype=float) + rng.uniform(-jitter, jitter, len(y))
    fig.add_trace(go.Scatter(x=xj, y=y, mode="markers", name=name, legendgroup=name, showlegend=showlegend,
                             marker=fs.marker(color, symbol, kind), text=text, hovertemplate="%{text}<br>%{y}"))


# ------------------------------------------------------------------ data statistics (script 110 data frames)
def data_statistics_panels(PL, PG, F, PS, out_main, out_supp, hdr):
    """PL per cell line, PG per gene, F per split x fold x part, PS per sample (110). Main Figure 2 -> out_main (a, b);
    supplementary Figure S1 -> out_supp (A..L). hdr = list of '#' header lines (source, parameters, date)."""
    H = list(hdr)
    # ---- Figure S1 A: knockouts per cell line (strip, one point per cell line)
    PL = PL.copy(); PL["set"] = PL.dataset.map(_dataset_key)
    fig = go.Figure()
    for i, (k, lab) in enumerate(DATASET_SHORT.items()):
        d = PL[PL.set == k]
        _strip(fig, [i] * len(d), d.knockouts, fs.SET_COLOR[k], fs.SET_SYMBOL[k], lab, d.cell_line + "<br>" + d.assembly)
    fig.update_xaxes(tickvals=[0, 1], ticktext=list(DATASET_SHORT.values()), range=[-0.6, 1.6])
    fs.layout(fig, yaxis="knockouts (GNN samples) per cell line", legend="top", width=520)
    fs.save(fig, out_supp, "panel_A_knockouts_per_cell_line", PL, H + ["panel A: one point per cell line"])
    # ---- S1 B: cell lines per gene (violin with points)
    PG = PG.copy(); PG["set"] = PG.dataset.map(_dataset_key)
    fig = go.Figure()
    for i, (k, lab) in enumerate(DATASET_SHORT.items()):
        d = PG[PG.set == k]
        fig.add_trace(go.Violin(x=[lab] * len(d), y=d.cell_lines, name=lab, line_color=fs.SET_COLOR[k],
                                fillcolor="rgba(0,0,0,0)", points="all", pointpos=0, jitter=0.35, marker=fs.marker(fs.SET_COLOR[k], fs.SET_SYMBOL[k], "dense"),
                                text=d.symbol, hovertemplate="%{text}<br>%{y}", box_visible=True, meanline_visible=False, spanmode="hard", width=0.8))
    fs.layout(fig, yaxis="cell lines per gene among the GNN samples", legend="top", width=520)
    fs.save(fig, out_supp, "panel_B_cell_lines_per_gene", PG, H + ["panel B: one point per gene"])
    # ---- S1 C-J: fold composition, one quantity per panel, per split
    QL = {"cell_line_gene_pairs": "cell line-gene pairs (knockouts)", "cell_lines": "cell lines", "genes": "genes",
          "disabled_reaction_groups": "disabled-reaction groups"}
    letters = iter("CDEFGHIJ")
    for split in ("gene-disjoint", "cell-line-disjoint"):
        for q, ql in QL.items():
            L = next(letters); f = F[F.split == split]
            fig = go.Figure()
            for part in ("train", "validation", "test"):
                d = f[f.part == part]
                fig.add_trace(go.Scatter(x=d.fold, y=d[q], mode="markers", name=part, marker=fs.marker(fs.PART_COLOR[part], fs.PART_SYMBOL[part], "mean")))
            fig.update_xaxes(tickvals=list(range(5)), title_text=f"fold ({split} split)")
            fs.layout(fig, yaxis=ql, width=480, height=400)
            fs.save(fig, out_supp, f"panel_{L}_fold_{q}_{split.replace('-', '_')}", f[["fold", "part", q]],
                    H + [f"panel {L}: {ql} per fold and part, {split} split"])
    # ---- S1 K, L: label-positive rate and MOMA-lethal rate per fold (one quantity per panel; symbol = split)
    for L, q, ql in (("K", "label_positive_rate", "essential fraction (DepMap CRISPRGeneDependency ≥ 0.5)"), ("L", "moma_lethal_rate", "MOMA-lethal fraction (relative growth rate < 0.05)")):
        fig = go.Figure()
        for split in ("gene-disjoint", "cell-line-disjoint"):
            for part in ("train", "validation", "test"):
                d = F[(F.split == split) & (F.part == part)]
                fig.add_trace(go.Scatter(x=d.fold + (-0.12 if split == "gene-disjoint" else 0.12), y=d[q], mode="markers",
                                         name=f"{part}, {split}", marker=fs.marker(fs.PART_COLOR[part], fs.SPLIT_SYMBOL[split], "mean")))
        fig.update_xaxes(tickvals=list(range(5)), title_text="fold")
        fs.layout(fig, yaxis=fs.br(ql, 34), width=620, height=420)
        fs.save(fig, out_supp, f"panel_{L}_{q}_per_fold", F[["split", "fold", "part", q]], H + [f"panel {L}: {ql} per split, fold and part"])
    # ---- Figure 2 a: strata of the gene-disjoint test knockouts per fold
    SL = {"rel_ge_0p2": "not train-test near duplicates (relative nearest-neighbor distance ≥ 0.2)",
          "rel_lt_0p2": "train-test near duplicates (distance < 0.2, not exact)", "exact_duplicate_of_training": "exact duplicate of a training fluxome"}
    f = F[(F.split == "gene-disjoint") & (F.part == "test")]
    fig = go.Figure()
    for s, sl in SL.items():
        fig.add_trace(go.Scatter(x=f.fold, y=f[s], mode="markers", name=fs.br(sl, 30), marker=fs.marker(fs.STRATUM_COLOR[s], fs.STRATUM_SYMBOL[s], "mean")))
    fig.update_xaxes(tickvals=list(range(5)), title_text="fold (gene-disjoint split)")
    fs.layout(fig, yaxis="test knockouts (n)", width=640, height=420)
    fs.save(fig, out_main, "panel_a_strata_per_fold_gene_disjoint", f[["fold", "part"] + list(SL)], H + ["panel a: gene-disjoint test knockouts per fold by relative nearest-neighbor distance stratum"])
    # validation-part and cell-line-disjoint strata go to the supplement as one panel (S1 M)
    f2 = F[(F.part != "train")]
    fig = go.Figure()
    for s, sl in SL.items():
        for part in ("validation", "test"):
            d = f2[(f2.split == "gene-disjoint") & (f2.part == part)]
            fig.add_trace(go.Scatter(x=d.fold + (-0.12 if part == "validation" else 0.12), y=d[s], mode="markers", name=f"{fs.br(sl, 26)},<br>{part}",
                                     marker=fs.marker(fs.STRATUM_COLOR[s], fs.PART_SYMBOL[part], "mean")))
    fig.update_xaxes(tickvals=list(range(5)), title_text="fold (gene-disjoint split)")
    fs.layout(fig, yaxis="knockouts (n)", width=720, height=440)
    fs.save(fig, out_supp, "panel_M_strata_validation_and_test_gene_disjoint", f2[f2.split == "gene-disjoint"][["fold", "part"] + list(SL)],
            H + ["panel M: validation and test knockouts per gene-disjoint fold by stratum (cell-line-disjoint split: distance undefined, all zero, not drawn)"])
    # ---- Figure 2 b: relative nearest-neighbor distance of every gene-disjoint test knockout
    T = PS[(PS.part == "test") & (PS.split == "gene")].copy()
    Tpos = T[T.rel_nn_dist > 0]
    fig = go.Figure()
    for lab, nm, col, sym in ((1, "essential (DepMap)", fs.PALETTE[1], "circle"), (0, "not essential", fs.PALETTE[0], "circle-open")):
        d = Tpos[Tpos.label == lab]
        _strip(fig, d.fold, d.rel_nn_dist, col, sym, nm, d.symbol + " " + d.line, jitter=0.3, kind="dense", seed=lab)
    fs.hline(fig, 0.2, "0.2")
    fig.update_yaxes(type="log"); fig.update_xaxes(tickvals=list(range(5)), title_text="fold (gene-disjoint split)")
    nz = int((T.rel_nn_dist == 0).sum())
    fig.add_annotation(text=f"{nz:,} exact duplicates (distance 0) not drawn", x=0.99, y=0.01, xref="paper", yref="paper", showarrow=False, font=dict(size=10, color=fs.NEUTRAL), xanchor="right")
    fs.layout(fig, yaxis="relative nearest-neighbor distance (log)", width=720, height=460, legend="top")
    fs.save(fig, out_main, "panel_b_rel_nn_dist_test_gene_disjoint", T[["fold", "line", "gene", "symbol", "rel_nn_dist", "label", "growth_ratio"]],
            H + ["panel b: one point per gene-disjoint test knockout; log y; dashed line 0.2; zero distances (exact duplicates) are not drawn"])


# ------------------------------------------------------------------ context-specific model statistics (script 112 data frames)
def context_model_panels(long, df, f59, out_dir, hdr):
    """long = per cell line x stage x entity counts (panel a data), df = per cell line S3 table (panel b/c data), f59 = reaction presence (panel d data)."""
    H = list(hdr)
    long = long.copy(); long["set"] = long.set.astype(str); df = df.copy()
    STAGE = {"S1": "S1 tINIT", "S2": "S2 essential-task<br>repair", "S3": "S3 as screened"}
    SETLAB = {"colorectal": "colorectal cell line", "external": "external validation cell line"}
    # a-c: counts by stage, one entity per panel (violin + points)
    for L, ename in zip("abc", ("genes", "reactions", "metabolites")):
        fig = go.Figure()
        for k in ("colorectal", "external"):
            d = long[(long.entity == ename) & (long.set == k)]
            if d.empty:
                continue
            fig.add_trace(go.Violin(x=d.stage.map(STAGE), y=d["count"], name=SETLAB[k], legendgroup=k, line_color=fs.SET_COLOR[k], fillcolor="rgba(0,0,0,0)",
                                    points="all", pointpos=0, jitter=0.35, marker=fs.marker(fs.SET_COLOR[k], fs.SET_SYMBOL[k], "point"), text=d.cell_line,
                                    hovertemplate="%{text}<br>%{y}", scalemode="width", width=0.8, meanline_visible=False, box_visible=False, spanmode="hard"))
        fs.hline(fig, float(d.scaffold.iloc[0]), "Human-GEM scaffold", dash="dot")
        fig.update_layout(violinmode="group")
        fs.layout(fig, yaxis=f"{ename} per context-specific model", legend="top", width=560, height=440)
        fs.save(fig, out_dir, f"panel_{L}_{ename}_by_stage", long[long.entity == ename], H + [f"panel {L}: {ename} per cell line and construction stage (S1 tINIT, S2 after essential-task repair, S3 model as screened); scaffold = Human-GEM value"])
    # d-f: size relations (S3)
    pairs = [("d", "S3_n_genes", "S3_n_reactions", "genes in model (S3)", "reactions in model (S3)"),
             ("e", "S3_n_reactions", "S3_n_metabolites", "reactions in model (S3)", "metabolites in model (S3)"),
             ("f", "S3_n_genes", "S3_n_screened_measured", "genes in model (S3)", "genes screened by knockout")]
    for L, xc, yc, xl, yl in pairs:
        fig = go.Figure()
        for k in ("colorectal", "external"):
            d = df[df.set == k]
            fig.add_trace(go.Scatter(x=d[xc], y=d[yc], mode="markers", name=SETLAB[k], marker=fs.marker(fs.SET_COLOR[k], fs.SET_SYMBOL[k], "point"),
                                     text=d.cell_line, hovertemplate="%{text}<br>x=%{x}<br>y=%{y}"))
        fs.layout(fig, xaxis=xl, yaxis=yl, legend="top", width=480, height=440)
        fs.save(fig, out_dir, f"panel_{L}_{yc.replace('S3_n_', '').replace('_measured', '')}_vs_{xc.replace('S3_n_', '')}", df[["set", "cell_line", "assembly", xc, yc]],
                H + [f"panel {L}: {yl} vs {xl} per cell line (model as screened, S3)"])
    # g-j: additions and medium exchanges, one quantity per panel
    cols = [("g", "S2_rec_n_added_reactions", "reactions added by essential-task repair"), ("h", "S3_gapfill_script62_n", "reactions added by requirement-cap gap-fill"),
            ("i", "S3_n_exchange_open_uptake", "exchange reactions open for uptake (medium)"), ("j", "S3_n_exchange_open_uptake_capped", "exchange reactions open for uptake and capped")]
    for L, c, cl in cols:
        fig = go.Figure()
        for i, k in enumerate(("colorectal", "external")):
            d = df[df.set == k]
            _strip(fig, [i] * len(d), d[c], fs.SET_COLOR[k], fs.SET_SYMBOL[k], SETLAB[k], d.cell_line + "<br>" + d.assembly)
        fig.update_xaxes(tickvals=[0, 1], ticktext=[fs.br(SETLAB[k], 16) for k in ("colorectal", "external")], range=[-0.6, 1.6])
        fs.layout(fig, yaxis=fs.br(cl, 30), legend="top", width=420, height=420)
        fs.save(fig, out_dir, f"panel_{L}_{c.replace('S3_', '').replace('S2_rec_', '').replace('_script62', '')}", df[["set", "cell_line", "assembly", c]], H + [f"panel {L}: {cl} per cell line"])
    # k, l: reaction presence
    f59 = f59.sort_values("rank")
    fig = go.Figure(go.Scattergl(x=f59["rank"], y=f59.n_models_present, mode="markers", marker=fs.marker(fs.PALETTE[0], "circle", "dense"), name="reaction",
                                 text=f59.reaction, hovertemplate="%{text}<br>rank %{x}<br>in %{y} models"))
    fs.layout(fig, xaxis="reaction rank (union across the 59 colorectal models)", yaxis="models containing the reaction (of 59)", showlegend=False, width=560, height=420)
    fs.save(fig, out_dir, "panel_k_reaction_presence_rank", f59[["reaction", "n_models_present", "n_models", "rank"]], H + ["panel k: each scaffold reaction present in >= 1 of the 59 colorectal S3 models, ranked by the number of models containing it"])
    dist = f59.n_models_present.value_counts().sort_index().rename_axis("n_models_present").reset_index(name="n_reactions")
    fig = go.Figure(go.Scatter(x=dist.n_models_present, y=dist.n_reactions, mode="markers+lines", marker=fs.marker(fs.PALETTE[0], "circle", "mean"), line=dict(width=1, color=fs.PALETTE[0]), name="reactions"))
    fig.update_yaxes(type="log")
    fs.layout(fig, xaxis="models containing the reaction (of 59)", yaxis="reactions (n, log)", showlegend=False, width=560, height=420)
    fs.save(fig, out_dir, "panel_l_reactions_per_presence_count", dist, H + ["panel l: number of scaffold reactions per presence count across the 59 colorectal S3 models"])
