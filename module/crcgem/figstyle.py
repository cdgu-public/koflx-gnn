"""Project figure style for CRC_GEM (plotly only): one palette, role colors, marker specs, layout, label breaking, saving.

Palette origin (user request 2026-11-02): seeded from quantum_gem/script_python/update_suppl_figures.py:16 PALETTE
['#2E86C1','#E74C3C','#27AE60','#F39C12','#8E44AD','#16A085'] and its COND_COLOR rows. Greens (#27AE60, #16A085) dropped
(no green hues); #E74C3C / #F39C12 darkened to #C0392B / #D35400 (the same file's COND_COLOR values) for contrast on white;
gold #B7950B and pink #E84393 added as slots 4-5. No liver / HCC project directory exists under /data/workspace/cdgu or
/home/cdgu (searched 2026-11-02), so quantum_gem is the only earlier-project palette referenced.
Colorblind check (dataviz validate_palette.js, light surface, 2026-11-02): slots 1-6 adjacent pairs PASS (worst CVD dE 13.6,
normal dE 15.6); slots 1-5 all-pairs PASS normal vision (worst 15.4), CVD worst 6.6 (deutan, blue vs purple) = warn band, so
every multi-series scatter also encodes the series by marker symbol. Red (slot 6) is never drawn next to orange in one plot.
Gold is below 3:1 contrast on white (2.8): gold series always carry a legend entry and a marker outline.
Gray #7F8C8D is reserved for neutral references (random, scaffold line, zero line), never a categorical slot.
"""
import os
import pandas as pd
import plotly.graph_objects as go

PALETTE = ["#2E86C1", "#D35400", "#8E44AD", "#B7950B", "#E84393", "#C0392B"]  # blue, orange, purple, gold, pink, red
NEUTRAL = "#7F8C8D"
INK = "#2C3E50"
GRID = "#EAECEE"

# role -> color. Labels are terminology.md terms (code labels mapped here, never shown raw).
MODEL_COLOR = {
    "GNN, Mass Flow Graph (MFG)": PALETTE[0],
    "GNN, bipartite metabolite-reaction graph (BMRG)": PALETTE[2],
    "MOMA": PALETTE[1],
    "gradient-boosting baseline, without medium": PALETTE[3],
    "gradient-boosting baseline, with medium": PALETTE[3],
    "per-gene base rate": PALETTE[4],
    "per-cell-line base rate": PALETTE[4],
    "random": NEUTRAL,
}
MODEL_SYMBOL = {
    "GNN, Mass Flow Graph (MFG)": "circle",
    "GNN, bipartite metabolite-reaction graph (BMRG)": "diamond",
    "MOMA": "square",
    "gradient-boosting baseline, without medium": "triangle-up",
    "gradient-boosting baseline, with medium": "triangle-up-open",
    "per-gene base rate": "cross",
    "per-cell-line base rate": "x",
    "random": "line-ew-open",
}
PART_COLOR = {"train": PALETTE[0], "validation": PALETTE[3], "test": PALETTE[1]}
PART_SYMBOL = {"train": "circle", "validation": "diamond", "test": "square"}
SET_COLOR = {"colorectal": PALETTE[0], "external": PALETTE[4]}
SET_SYMBOL = {"colorectal": "circle", "external": "diamond"}
STRATUM_COLOR = {"rel_ge_0p2": PALETTE[0], "rel_lt_0p2": PALETTE[1], "exact_duplicate_of_training": NEUTRAL}
STRATUM_SYMBOL = {"rel_ge_0p2": "circle", "rel_lt_0p2": "square", "exact_duplicate_of_training": "x"}
MEDIUM_COLOR = {"own medium": PALETTE[0], "DMEM/F-12": PALETTE[1], "RPMI-1640": PALETTE[2]}
MEDIUM_SYMBOL = {"own medium": "circle", "DMEM/F-12": "square", "RPMI-1640": "diamond"}
SPLIT_SYMBOL = {"gene-disjoint": "circle", "cell-line-disjoint": "diamond"}

# code label -> dictionary term (documentation/terminology.md). Line breaks (<br>) only; words are never changed.
STRATUM_LABEL = {
    "all": "all test<br>knockouts",
    "exact-dup-free": "no exact match in<br>any training fluxome",
    "rel>=0.2": "not train-test<br>near duplicates<br>(relative nearest-neighbor<br>distance ≥ 0.2)",
    "rel>=0.5": "relative<br>nearest-neighbor<br>distance ≥ 0.5",
    "MOMA-lethal (G<0.05)": "MOMA-lethal<br>knockouts",
    "rel>=0.2 & MOMA-lethal": "distance ≥ 0.2 and<br>MOMA-lethal",
    "G>0.99": "relative growth<br>rate > 0.99",
}
STRATUM_ORDER = ["all", "exact-dup-free", "rel>=0.2", "rel>=0.5", "MOMA-lethal (G<0.05)", "rel>=0.2 & MOMA-lethal", "G>0.99"]
CONTROL_LABEL = {  # 85/118 variant tokens -> terminology.md terms
    "nomed_fixedcos30_tolpatch": "GNN, Mass Flow Graph (MFG)",
    "nomed_meanmax_fixedcos30_tolpatch": "GNN, bipartite metabolite-reaction graph (BMRG)",
    "nomed_shufrelabel_fixedcos30_tolpatch": "node-feature<br>shuffling control",
    "nomed_mlp81_fixedcos30_tolpatch": "multilayer perceptron baseline<br>(GNN training recipe)",  # PROVISIONAL label, user to ratify (open_questions 29)
    "nomed_mlphgb_fixedcos30_tolpatch": "multilayer perceptron baseline<br>(gradient-boosting-like<br>training recipe)",  # PROVISIONAL label, user to ratify (open_questions 29)
    "nomed_rid16_fixedcos30_tolpatch": "reaction identity<br>embedding",
    "nomed_wtonly_fixedcos30_tolpatch": "WT-only input",
    "nomed_disflag_fixedcos30_tolpatch": "baseline input plus a column<br>marking the reactions disabled<br>by the knockout",
}
MEDIUM_LABEL = {"DMEMF12_Gibco11320": "DMEM/F-12", "RPMI1640_Gibco11875": "RPMI-1640", "own": "own medium"}
EXTERNAL_LINE_LABEL = {"ACH-000551": "K-562", "ACH-002462": "RPE1-ss48", "ACH-002463": "RPE1-ss77", "ACH-002464": "RPE1-ss6",
                       "ACH-002465": "RPE1-ss119", "ACH-002466": "RPE1-ss111", "ACH-002467": "RPE1-ss51",
                       "ACH-001654": "SK-GT-4", "ACH-000216": "JH-EsoAd1", "ACH-000427": "NCI-N87", "ACH-000466": "SNU-216", "ACH-000307": "PK-1", "ACH-000976": "HuCCT1", "ACH-000133": "Hs 729", "ACH-001453": "BPH-1"}  # DepMap Model.csv names (external_eval_plan.md:21-26)

MARKER = dict(size=6, opacity=0.6, line=dict(width=0.5, color="white"))        # per-sample / per-fold points
MARKER_MEAN = dict(size=11, opacity=0.95, line=dict(width=1.0, color=INK))     # fold-mean / summary points
MARKER_DENSE = dict(size=4, opacity=0.5, line=dict(width=0))                   # thousands of points
LINE = dict(width=1.5)


def marker(color, symbol="circle", kind="point"):
    base = {"point": MARKER, "mean": MARKER_MEAN, "dense": MARKER_DENSE}[kind]
    m = dict(base); m["line"] = dict(base["line"]); m["color"] = color; m["symbol"] = symbol
    return m


def br(text, width=22):
    """Break a label at spaces into lines of at most `width` characters (words intact, never renamed)."""
    words, lines, cur = str(text).split(" "), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "<br>".join(lines)


def layout(fig, title=None, xaxis=None, yaxis=None, width=700, height=480, legend="right", showlegend=True, bottom=70):
    fig.update_layout(template="simple_white", font=dict(size=13, color=INK, family="Arial"), width=width, height=height,
                      title=dict(text=title, x=0.01, font=dict(size=14)) if title else None,
                      margin=dict(l=70, r=20, t=50 if title else 20, b=bottom), showlegend=showlegend,
                      plot_bgcolor="white", paper_bgcolor="white")
    if legend == "right":
        fig.update_layout(legend=dict(orientation="v", x=1.02, y=1, xanchor="left", yanchor="top", font=dict(size=11),
                                      bgcolor="rgba(255,255,255,0)", itemsizing="constant"))
    elif legend == "top":
        fig.update_layout(legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom", font=dict(size=11),
                                      bgcolor="rgba(255,255,255,0)", itemsizing="constant"))
    elif legend == "inside":
        fig.update_layout(legend=dict(orientation="v", x=0.02, y=0.98, xanchor="left", yanchor="top", font=dict(size=11),
                                      bgcolor="rgba(255,255,255,0.7)", itemsizing="constant"))
    fig.update_xaxes(showgrid=False, zeroline=False, ticks="outside", linecolor=INK, tickfont=dict(size=11), title_font=dict(size=12))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, ticks="outside", linecolor=INK, tickfont=dict(size=11), title_font=dict(size=12))
    if xaxis:
        fig.update_xaxes(title_text=xaxis)
    if yaxis:
        fig.update_yaxes(title_text=yaxis)
    return fig


def hline(fig, y, text=None, color=NEUTRAL, dash="dash"):
    if text:
        fig.add_hline(y=y, line=dict(color=color, width=1, dash=dash),
                      annotation_text=text, annotation_position="top right", annotation_font=dict(size=10, color=color))
    else:
        fig.add_hline(y=y, line=dict(color=color, width=1, dash=dash))


def write_table(df, path_noext, header_lines):
    """TSV + CSV with '#' header lines (source, parameters, date, n)."""
    for ext, sep in (("tsv", "\t"), ("csv", ",")):
        with open(f"{path_noext}.{ext}", "w") as fh:
            for h in header_lines:
                fh.write("# " + h.rstrip("\n") + "\n")
            fh.write(f"# n rows {len(df)}\n")
            df.to_csv(fh, sep=sep, index=False)


def save(fig, out_dir, name, data=None, header_lines=(), scale=2):
    """html + svg + png + jpeg, plus <name>_data.tsv/.csv (panel-wise data with '#' header lines)."""
    os.makedirs(out_dir, exist_ok=True)
    fig.write_html(f"{out_dir}/{name}.html", include_plotlyjs="cdn")
    for ext in ("svg", "png", "jpeg"):
        try:
            fig.write_image(f"{out_dir}/{name}.{ext}", scale=None if ext == "svg" else scale)
        except Exception as e:  # noqa: BLE001
            print(f"{name}.{ext} not written: {e}", flush=True)
    if data is not None:
        write_table(data, f"{out_dir}/{name}_data", list(header_lines))
    return f"{out_dir}/{name}"


def placeholder(out_dir, name, text, header_lines=()):
    """Empty panel with a centered note (for results that do not exist yet); same file set as save()."""
    fig = go.Figure()
    fig.add_annotation(text=text, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=13, color=NEUTRAL), align="center")
    layout(fig, showlegend=False)
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return save(fig, out_dir, name, pd.DataFrame({"placeholder": [text.replace("<br>", " ")]}), list(header_lines))
