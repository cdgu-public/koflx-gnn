#!/usr/bin/env python
"""Assemble the DepMap CRC cell-line training set for the mortality predictor.

Filters DepMap (24Q4) to colorectal models, intersects the lines that have BOTH
a CRISPR screen and expression, and writes training-ready tables:
  - crc_dependency.parquet  : lines x genes, 0-1 DepMap dependency probability (label)
  - crc_gene_effect.parquet : lines x genes, Chronos gene effect
  - crc_expression.parquet  : lines x genes, log2(TPM+1)
  - crc_trainset_models.tsv : the usable line list + metadata

DepMap gene headers are "SYMBOL (Entrez)"; this script strips them to symbol and
to Entrez, so downstream `genemap` can map to the scaffold namespace (ENSG for
Human-GEM). Nothing here imports scTrx code — this is a self-contained DepMap loader.

    python script_python/11_depmap_crc_trainset.py --depmap-dir data/depmap \
        --out data/depmap/crc_trainset
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("depmap_crc")

_HEADER = re.compile(r"^(?P<symbol>.+?)\s*\((?P<entrez>\d+)\)\s*$")


def strip_headers(cols) -> tuple[list, list]:
    """DepMap 'SYMBOL (Entrez)' -> (symbols, entrez_ids)."""
    syms, ents = [], []
    for c in cols:
        m = _HEADER.match(str(c))
        if m:
            syms.append(m.group("symbol")); ents.append(m.group("entrez"))
        else:
            syms.append(str(c)); ents.append("")
    return syms, ents


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depmap-dir", default="data/depmap")
    ap.add_argument("--out", default="data/depmap/crc_trainset")
    ap.add_argument("--disease-match", default="Colorectal|Colon|Rectal")
    args = ap.parse_args()

    d = Path(args.depmap_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    model = pd.read_csv(d / "Model.csv")
    crc = model[model["OncotreeLineage"] == "Bowel"]
    crc = crc[crc["OncotreePrimaryDisease"].str.contains(
        args.disease_match, case=False, na=False)]
    crc_ids = set(crc["ModelID"])
    log.info("colorectal models in Model.csv: %d", len(crc_ids))

    dep = pd.read_csv(d / "CRISPRGeneDependency.csv", index_col=0)
    eff = pd.read_csv(d / "CRISPRGeneEffect.csv", index_col=0)
    expr = pd.read_csv(d / "OmicsExpressionProteinCodingGenesTPMLogp1.csv", index_col=0)
    log.info("dependency %s | effect %s | expression %s",
             dep.shape, eff.shape, expr.shape)

    usable = sorted(crc_ids & set(dep.index) & set(eff.index) & set(expr.index))
    log.info("CRC lines with dependency AND effect AND expression: %d", len(usable))
    if not usable:
        raise SystemExit("no CRC line has all three modalities — check inputs")

    # subset to usable lines; write with a symbol/entrez key map per matrix
    for name, df in [("dependency", dep), ("gene_effect", eff), ("expression", expr)]:
        sub = df.loc[usable].copy()
        syms, ents = strip_headers(sub.columns)
        sub.columns = syms                     # symbol-keyed (genemap input)
        sub.to_parquet(out / f"crc_{name}.parquet")
        pd.DataFrame({"symbol": syms, "entrez": ents}).to_csv(
            out / f"crc_{name}.genekey.tsv", sep="\t", index=False)
        log.info("wrote crc_%s.parquet %s", name, sub.shape)

    crc[crc["ModelID"].isin(usable)].to_csv(out / "crc_trainset_models.tsv",
                                            sep="\t", index=False)
    manifest = {
        "release": "DepMap 24Q4 (figshare; 25Q2 portal is bot-walled)",
        "n_crc_models_total": len(crc_ids),
        "n_usable_all_three_modalities": len(usable),
        "gene_key": "DepMap 'SYMBOL (Entrez)' -> symbol columns + genekey.tsv; "
                    "map to scaffold via crcgem.genemap",
        "label": "crc_dependency.parquet = 0-1 DepMap dependency probability",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
