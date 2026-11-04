"""Harmonise incoming gene IDs to the scaffold's namespace.

Two scaffolds are in use (user 2026-07-27), with different gene namespaces:
  * Human-GEM (Human2)  — genes keyed on Ensembl ENSG
  * Recon3D              — genes keyed on NCBI Entrez

Incoming data may be ENSG (Census scRNA) or symbol (ARCHS4 bulk) or entrez, so a
mapping layer is unavoidable. The mapping is LOSSY for symbol/transcript
sources; this module reports mapped/dropped/collapsed counts rather than hiding
the attrition — the exact failure the BRCA precedent never disclosed.

Mapping authority: Human-GEM's `model/genes.tsv` bundles parallel columns
(genes=ENSG, geneEntrezID, geneSymbols), so once the Human-GEM scaffold is
present it doubles as the ENSG↔Entrez↔symbol table. Any equivalent table with
those columns works; pass its path.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: scaffold name -> the gene namespace its model uses
SCAFFOLD_NAMESPACE = {"human_gem": "ensembl", "recon3d": "entrez"}


@dataclass
class MappingReport:
    from_type: str
    to_type: str
    n_input: int
    n_mapped: int
    n_unmapped: int
    n_collapsed: int          # input ids that merged onto a shared target
    unmapped_examples: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "from_type": self.from_type, "to_type": self.to_type,
            "n_input": self.n_input, "n_mapped": self.n_mapped,
            "n_unmapped": self.n_unmapped, "n_collapsed": self.n_collapsed,
            "map_rate": round(self.n_mapped / self.n_input, 4) if self.n_input else 0.0,
            "unmapped_examples": self.unmapped_examples[:10],
        }


def load_gene_table(path: str | Path, *,
                    ensembl_col: str = "genes",
                    entrez_col: str = "geneEntrezID",
                    symbol_col: str = "geneSymbols") -> pd.DataFrame:
    """Load a gene cross-reference table (e.g. Human-GEM model/genes.tsv)."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    cols = {ensembl_col: "ensembl", entrez_col: "entrez", symbol_col: "symbol"}
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"gene table missing columns {missing}; has {list(df.columns)}")
    out = df[list(cols)].rename(columns=cols)
    for c in out.columns:
        out[c] = out[c].astype(str).str.strip()
    return out


def _build_lookup(gene_table: pd.DataFrame, from_type: str, to_type: str) -> dict:
    sub = gene_table[[from_type, to_type]].dropna()
    sub = sub[(sub[from_type] != "") & (sub[from_type].str.lower() != "nan")
              & (sub[to_type] != "") & (sub[to_type].str.lower() != "nan")]
    # symbols/entrez can be multivalued (e.g. "A;B") — explode on common seps
    if from_type in ("symbol", "entrez"):
        sub = sub.assign(**{from_type: sub[from_type].str.split(r"[;,| ]+")}).explode(from_type)
    return dict(zip(sub[from_type], sub[to_type]))


def harmonise(matrix: pd.DataFrame, from_type: str, scaffold: str, *,
              gene_table_path: str | Path,
              aggregate: str = "sum") -> tuple[pd.DataFrame, MappingReport]:
    """Remap a genes×samples matrix index from `from_type` to the scaffold namespace.

    Args:
        matrix: genes (rows) × samples (columns), index in `from_type` namespace.
        from_type: 'ensembl' | 'symbol' | 'entrez' (the incoming namespace).
        scaffold: 'human_gem' | 'recon3d'.
        aggregate: how to combine input rows that map to the same target id
                   ('sum' for counts, 'mean' for corrected expression).
    Returns the remapped matrix (index = scaffold namespace) and a MappingReport.
    """
    to_type = SCAFFOLD_NAMESPACE[scaffold]
    if from_type == to_type:
        return matrix.copy(), MappingReport(from_type, to_type, len(matrix),
                                            len(matrix), 0, 0)

    table = load_gene_table(gene_table_path)
    lut = _build_lookup(table, from_type, to_type)

    mapped_ids = matrix.index.map(lambda g: lut.get(str(g)))
    keep = mapped_ids.notna()
    unmapped = matrix.index[~keep]

    kept = matrix.loc[keep].copy()
    kept.index = mapped_ids[keep]
    n_before = len(kept)
    if kept.index.duplicated().any():
        agg = kept.groupby(level=0)
        kept = agg.sum() if aggregate == "sum" else agg.mean()
    n_collapsed = n_before - len(kept)

    report = MappingReport(
        from_type=from_type, to_type=to_type, n_input=len(matrix),
        n_mapped=len(kept), n_unmapped=int((~keep).sum()),
        n_collapsed=n_collapsed, unmapped_examples=list(unmapped[:10]),
    )
    logger.info("gene map %s→%s (%s): %d/%d mapped (%.1f%%), %d collapsed, %d dropped",
                from_type, to_type, scaffold, report.n_mapped, report.n_input,
                100 * report.n_mapped / max(report.n_input, 1),
                report.n_collapsed, report.n_unmapped)
    return kept, report
