"""Dual scaffold support: Human-GEM (Human2) and Recon3D.

User 2026-07-27: use BOTH scaffolds. They differ in gene namespace (Human-GEM =
Ensembl ENSG; Recon3D = Entrez), so a run states which one and the incoming data
is harmonised to it (genemap). The scaffold's own gene set is the honest
denominator for "how much of the model does this data cover" — NOT the input
gene count, which is dominated by non-metabolic genes.

Fetch: `fetch_human_gem()` pulls the pinned v2.0.0 SBML + gene table + tasks from
the SysBioChalmers repo. Recon3D is fetched from a mirror; if unreachable the
loader still works on a locally provided file.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SCAFFOLDS = {
    "human_gem": {
        "namespace": "ensembl",
        "version": "2.0.0",
        "sbml_url": "https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/v2.0.0/model/Human-GEM.xml",
        "genes_url": "https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/v2.0.0/model/genes.tsv",
        "tasks_url": "https://raw.githubusercontent.com/SysBioChalmers/Human-GEM/v2.0.0/data/metabolicTasks/metabolicTasks_Essential.txt",
        "gene_id_pattern": "ENSG",
    },
    "recon3d": {
        "namespace": "entrez",
        "version": "3.01",
        # BiGG is often unreachable from sandboxes; several mirrors exist.
        "sbml_url": "http://bigg.ucsd.edu/static/models/Recon3D.xml",
        "genes_url": None,     # Entrez lives in the SBML; gene table via Human-GEM xref
        "tasks_url": None,     # Recon3D ships no clean task file (port if needed)
        "gene_id_pattern": None,
    },
}


@dataclass
class Scaffold:
    name: str
    namespace: str
    version: str
    sbml_path: Path
    genes_path: Optional[Path] = None
    tasks_path: Optional[Path] = None

    def gene_table_path(self) -> Path:
        """Path to an ENSG↔Entrez↔symbol table for genemap.

        Human-GEM's genes.tsv is the authority for BOTH scaffolds (it bundles all
        three id types), so Recon3D reuses it for symbol/ENSG→Entrez mapping.
        """
        if self.genes_path and self.genes_path.exists():
            return self.genes_path
        raise FileNotFoundError(
            "no gene cross-reference table available; fetch Human-GEM genes.tsv "
            "(it serves both scaffolds).")


def fetch_human_gem(dest: str | Path = "data/scaffold") -> Scaffold:
    """Download the pinned Human-GEM v2.0.0 SBML + gene table + tasks."""
    import urllib.request

    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    spec = SCAFFOLDS["human_gem"]
    paths = {}
    for key, url, fname in [
        ("sbml", spec["sbml_url"], "Human-GEM.xml"),
        ("genes", spec["genes_url"], "humangem_genes.tsv"),
        ("tasks", spec["tasks_url"], "metabolicTasks_Essential.txt"),
    ]:
        p = d / fname
        if not p.exists():
            logger.info("fetching %s -> %s", url, p)
            urllib.request.urlretrieve(url, p)
        paths[key] = p
    return Scaffold("human_gem", spec["namespace"], spec["version"],
                    paths["sbml"], paths["genes"], paths["tasks"])


def load_scaffold(name: str, sbml_path: str | Path, *,
                  genes_path: Optional[str | Path] = None,
                  tasks_path: Optional[str | Path] = None) -> Scaffold:
    """Register a scaffold from local files (no fetch)."""
    if name not in SCAFFOLDS:
        raise ValueError(f"unknown scaffold {name!r}; known: {list(SCAFFOLDS)}")
    spec = SCAFFOLDS[name]
    return Scaffold(name, spec["namespace"], spec["version"], Path(sbml_path),
                    Path(genes_path) if genes_path else None,
                    Path(tasks_path) if tasks_path else None)


def scaffold_gene_ids(model) -> set:
    """The scaffold's own gene id set — the honest coverage denominator."""
    return {g.id for g in model.genes}


def coverage(matrix: pd.DataFrame, model) -> dict:
    """How much of the SCAFFOLD's gene set the (already-harmonised) data covers.

    This is the meaningful number, unlike genemap's input-side map_rate: the
    model has ~2,848 genes (Human-GEM) and most of a 60k input is non-metabolic.
    """
    model_genes = scaffold_gene_ids(model)
    present = model_genes & set(matrix.index.astype(str))
    return {
        "scaffold_genes": len(model_genes),
        "covered": len(present),
        "coverage_rate": round(len(present) / max(len(model_genes), 1), 4),
        "missing_examples": sorted(model_genes - present)[:10],
    }
