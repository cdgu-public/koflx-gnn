"""The data boundary. CRC_GEM consumes files matching data/README.md — and
imports NO upstream (scTrx / scRNA_gen) code. This module is the only entry for
expression + metadata, and it validates rather than guesses.

Design goal (user 2026-07-27): receiving only conforming DATA must be enough to
run. So the schema is explicit, the metadata column map is caller-supplied (the
upstream project keeps its own column names), and gene IDs are harmonised to the
chosen scaffold namespace here, not upstream.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL_FIELDS = ("status", "provenance", "donor_id", "cell_type")
GENE_ID_TYPES = ("ensembl", "symbol", "entrez")
# "scfoundation_corrected" RETIRED 2026-10-15 (user ruling: open-source tooling only,
# scFoundation is not part of it). Still ACCEPTED so existing declarations do not break,
# but it warns and should be re-declared as "batch_corrected". The correction method
# itself is undecided; see documentation/measurements_20261014.md section 7 item 6.
VALUE_TYPES = ("raw_counts", "batch_corrected", "scfoundation_corrected")
RETIRED_VALUE_TYPES = {"scfoundation_corrected": "batch_corrected"}


class ContractError(ValueError):
    """Raised when a data drop does not match data/README.md."""


@dataclass
class ExpressionInput:
    """Standardised expression + metadata, gene IDs still in source namespace.

    matrix: genes (rows) × samples/cells (columns). For single-cell this may be
            a pseudobulk (genes × groups); per-cell access uses `load_singlecell`.
    meta:   indexed by matrix columns; canonical fields present.
    """

    matrix: pd.DataFrame
    meta: pd.DataFrame
    gene_id_type: str
    value_type: str
    kind: str                     # "bulk" | "single_cell" | "pseudobulk"
    source: str = ""

    def summary(self) -> dict:
        return {
            "kind": self.kind, "source": self.source,
            "gene_id_type": self.gene_id_type, "value_type": self.value_type,
            "n_genes": int(self.matrix.shape[0]),
            "n_samples": int(self.matrix.shape[1]),
            "statuses": sorted(self.meta["status"].dropna().unique().tolist()),
            "provenances": sorted(self.meta["provenance"].dropna().unique().tolist()),
        }


def _apply_column_map(meta: pd.DataFrame, column_map: Mapping[str, str],
                      *, is_single_cell: bool,
                      default_provenance: str = "real") -> pd.DataFrame:
    """Rename incoming metadata columns to canonical fields; validate presence."""
    out = pd.DataFrame(index=meta.index)
    for canon, src in column_map.items():
        if canon not in CANONICAL_FIELDS:
            raise ContractError(f"unknown canonical field {canon!r}; "
                                f"allowed: {CANONICAL_FIELDS}")
        if src not in meta.columns:
            raise ContractError(f"column_map points {canon!r} at missing column {src!r}; "
                                f"available: {list(meta.columns)[:20]}")
        out[canon] = meta[src].values

    if "status" not in out:
        raise ContractError("column_map must supply 'status'")
    if "donor_id" not in out:
        raise ContractError("column_map must supply 'donor_id' "
                            "(needed for donor-level split / no leakage)")
    if "provenance" not in out:
        out["provenance"] = default_provenance
    if is_single_cell and "cell_type" not in out:
        raise ContractError("single-cell input requires 'cell_type' in column_map")
    return out


def _validate_declared(gene_id_type: str, value_type: str) -> None:
    if gene_id_type not in GENE_ID_TYPES:
        raise ContractError(f"gene_id_type must be one of {GENE_ID_TYPES}")
    if value_type not in VALUE_TYPES:
        raise ContractError(f"value_type must be one of {VALUE_TYPES}")
    if value_type in RETIRED_VALUE_TYPES:
        logger.warning(
            "value_type %r is retired (2026-10-15); declare %r instead",
            value_type, RETIRED_VALUE_TYPES[value_type])


def load_bulk(counts_path: str | Path, meta_path: str | Path, *,
              gene_id_type: str, value_type: str,
              column_map: Mapping[str, str],
              default_provenance: str = "real") -> ExpressionInput:
    """Load a bulk counts matrix (genes × samples) + sample metadata."""
    _validate_declared(gene_id_type, value_type)
    counts = pd.read_parquet(counts_path)
    mp = Path(meta_path)
    meta = (pd.read_parquet(mp) if mp.suffix == ".parquet"
            else pd.read_csv(mp, sep="\t", index_col=0))

    common = counts.columns.intersection(meta.index)
    if len(common) == 0:
        raise ContractError(
            "no overlap between counts columns and metadata index — are the "
            "sample ids aligned? (counts cols e.g. "
            f"{list(counts.columns[:3])}; meta index e.g. {list(meta.index[:3])})")
    counts = counts[common]
    meta = _apply_column_map(meta.loc[common], column_map,
                             is_single_cell=False,
                             default_provenance=default_provenance)
    logger.info("bulk: %d genes × %d samples (%s, %s)",
                *counts.shape, gene_id_type, value_type)
    return ExpressionInput(counts, meta, gene_id_type, value_type,
                           kind="bulk", source=str(counts_path))


def load_singlecell(h5ad_path: str | Path, *, gene_id_type: str, value_type: str,
                    column_map: Mapping[str, str],
                    default_provenance: str = "real"):
    """Open an AnnData handle + validate obs schema. Returns (adata, meta).

    Does NOT densify — the caller decides per-cell (Track 1) vs pseudobulk
    (Track 2). Use `pseudobulk` for the latter.
    """
    import anndata as ad

    _validate_declared(gene_id_type, value_type)
    adata = ad.read_h5ad(h5ad_path)
    meta = _apply_column_map(adata.obs, column_map, is_single_cell=True,
                             default_provenance=default_provenance)
    adata.obs[list(meta.columns)] = meta.values
    logger.info("single-cell: %d cells × %d genes (%s, %s)",
                adata.n_obs, adata.n_vars, gene_id_type, value_type)
    return adata, meta


def pseudobulk(adata, groupby: str = "donor_id", *, gene_id_type: str,
               value_type: str, min_cells: int = 20,
               source: str = "") -> ExpressionInput:
    """Aggregate single cells to pseudobulk (genes × groups) by summing counts.

    Summing (not averaging) restores bulk-like depth, which is what tINIT's
    presence/absence calls need — the reason bulk/pseudobulk feeds extraction
    while single cells feed the fluxome distribution.
    """
    import scipy.sparse as sp

    if groupby not in adata.obs:
        raise ContractError(f"groupby {groupby!r} not in obs")
    groups, rows, kept = [], [], []
    X = adata.X
    for g, idx in adata.obs.groupby(groupby, observed=True).groups.items():
        pos = adata.obs.index.get_indexer(idx)
        if len(pos) < min_cells:
            continue
        sub = X[pos]
        summed = np.asarray(sub.sum(axis=0)).ravel() if sp.issparse(sub) else sub.sum(0)
        rows.append(summed)
        groups.append(str(g))
        kept.append(idx)
    if not groups:
        raise ContractError(f"no group had >= {min_cells} cells")

    mat = pd.DataFrame(np.vstack(rows).T, index=adata.var_names, columns=groups)
    # one metadata row per group: take the modal status/provenance/cell mix
    meta_rows = {}
    for g, idx in zip(groups, kept):
        obs = adata.obs.loc[idx]
        meta_rows[g] = {
            "status": obs["status"].mode().iloc[0],
            "provenance": obs["provenance"].mode().iloc[0],
            "donor_id": g if groupby == "donor_id" else obs["donor_id"].mode().iloc[0],
            "cell_type": obs["cell_type"].mode().iloc[0] if "cell_type" in obs else "",
        }
    meta = pd.DataFrame(meta_rows).T
    logger.info("pseudobulk by %s: %d genes × %d groups (>=%d cells each)",
                groupby, mat.shape[0], mat.shape[1], min_cells)
    return ExpressionInput(mat, meta, gene_id_type, value_type,
                           kind="pseudobulk", source=source or str(groupby))
