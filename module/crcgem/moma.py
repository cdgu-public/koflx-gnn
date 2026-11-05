"""MOMA perturbation screens: single deletion and anchored double deletion.

Departures from the BRCA precedent, each deliberate (see the project decision log
"Precedent defects"):
  P1  GPR-aware deletion. BRCA knocked out `gene.reactions` wholesale, discarding
      isozyme redundancy and collapsing complex members to identical values.
      Here deletion goes through cobra's GPR evaluation.
  P2  One lethality definition (config.moma.lethal_growth_fraction = 0.05),
      not 5% in one script and 10% in another.
  P3  No Bliss formula is hard-coded here. Interaction scoring lives in
      `interaction.py` with the reference model stated explicitly.
  P8  Full per-reaction flux vectors are persisted, so downstream analyses are
      reproducible from the archive.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .reference_flux import ReferenceFlux

logger = logging.getLogger(__name__)


@dataclass
class MomaResult:
    """Outcome of one perturbation."""

    label: str
    status: str
    growth: float
    growth_ratio: float          # relative to WT FBA optimum
    objective_value: float       # the MOMA distance
    fluxes: Optional[pd.Series] = None

    def is_lethal(self, fraction: float) -> bool:
        return self.growth_ratio < fraction


def _biomass_reaction(model, biomass_id: Optional[str] = None):
    if biomass_id:
        return model.reactions.get_by_id(biomass_id)
    objs = [r for r in model.reactions if r.objective_coefficient != 0]
    if len(objs) != 1:
        raise ValueError(
            f"cannot infer biomass reaction ({len(objs)} objective reactions); "
            "pass biomass_id explicitly")
    return objs[0]


def _solve_moma(model, reference: pd.Series, formulation: str,
                label: str, wt_growth: float, biomass,
                keep_fluxes: bool) -> MomaResult:
    """Solve one MOMA problem against `reference`."""
    from cobra.flux_analysis.moma import add_moma

    linear = formulation == "linear"
    with model:
        # cobra's add_moma expects a Solution-like object carrying .fluxes
        add_moma(model, solution=_as_solution(reference), linear=linear)
        sol = model.optimize()
        if sol.status != "optimal":
            return MomaResult(label, sol.status, np.nan, np.nan, np.nan)
        growth = sol.fluxes[biomass.id]
        return MomaResult(
            label=label,
            status=sol.status,
            growth=growth,
            growth_ratio=growth / wt_growth if wt_growth else np.nan,
            objective_value=sol.objective_value,
            fluxes=sol.fluxes.copy() if keep_fluxes else None,
        )


def _as_solution(fluxes: pd.Series):
    from cobra.core import Solution

    return Solution(objective_value=np.nan, status="optimal", fluxes=fluxes)


def single_gene_screen(model, reference: ReferenceFlux, *, formulation: str,
                       genes: Optional[Sequence[str]] = None,
                       biomass_id: Optional[str] = None,
                       keep_fluxes: bool = True,
                       progress: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """MOMA over single gene deletions (GPR-aware).

    Returns:
        summary: one row per gene (status, growth, growth_ratio, moma objective)
        flux_matrix: genes x reactions flux vectors (empty if keep_fluxes=False).
            Persisting this is what makes the screen reproducible — the BRCA
            archive omitted it and its ML half became irreproducible.
    """
    biomass = _biomass_reaction(model, biomass_id)
    gene_ids = list(genes) if genes is not None else [g.id for g in model.genes]
    rows, flux_rows = [], {}

    it = _maybe_tqdm(gene_ids, progress, "single-gene MOMA")
    for gid in it:
        with model:
            model.genes.get_by_id(gid).knock_out()   # GPR-aware (fixes P1)
            res = _solve_moma(model, reference.fluxes, formulation, gid,
                              reference.growth, biomass, keep_fluxes)
        rows.append({"gene": gid, "status": res.status, "growth": res.growth,
                     "growth_ratio": res.growth_ratio,
                     "moma_objective": res.objective_value})
        if keep_fluxes and res.fluxes is not None:
            flux_rows[gid] = res.fluxes

    summary = pd.DataFrame(rows).set_index("gene")
    flux_matrix = pd.DataFrame(flux_rows).T if flux_rows else pd.DataFrame()
    return summary, flux_matrix


def anchored_double_screen(model, reference: ReferenceFlux, anchor_genes: Sequence[str],
                           *, formulation: str, genes: Optional[Sequence[str]] = None,
                           biomass_id: Optional[str] = None,
                           progress: bool = True) -> pd.DataFrame:
    """MOMA over (anchor + partner) double deletions — the BRCA 1xN design.

    O(N) per anchor rather than O(N^2), which is why the precedent was tractable.
    Unlike the precedent, the anchor is a *gene* (GPR-aware) and is supplied by
    the caller rather than hard-coded, so anchor selection is a recorded step.
    """
    biomass = _biomass_reaction(model, biomass_id)
    gene_ids = list(genes) if genes is not None else [g.id for g in model.genes]
    rows = []

    for anchor in anchor_genes:
        it = _maybe_tqdm(gene_ids, progress, f"double MOMA (anchor={anchor})")
        for gid in it:
            if gid == anchor:
                continue
            with model:
                model.genes.get_by_id(anchor).knock_out()
                model.genes.get_by_id(gid).knock_out()
                res = _solve_moma(model, reference.fluxes, formulation,
                                  f"{anchor}+{gid}", reference.growth, biomass,
                                  keep_fluxes=False)
            rows.append({"anchor": anchor, "partner": gid, "status": res.status,
                         "growth": res.growth, "growth_ratio": res.growth_ratio,
                         "moma_objective": res.objective_value})
    return pd.DataFrame(rows)


def screen_over_reference_ensemble(model, reference: ReferenceFlux, *,
                                   formulation: str,
                                   genes: Optional[Sequence[str]] = None,
                                   biomass_id: Optional[str] = None
                                   ) -> pd.DataFrame:
    """Repeat the single-gene screen against each reference in an ensemble.

    Directly quantifies how much of a KO growth call is an artefact of which
    optimum was chosen as reference. Reports per-gene min/median/max growth
    ratio, i.e. the KO effect as an INTERVAL over the optimal face.
    """
    if reference.ensemble is None:
        raise ValueError("reference has no ensemble; use reference_mode='face_ensemble'")

    biomass = _biomass_reaction(model, biomass_id)
    gene_ids = list(genes) if genes is not None else [g.id for g in model.genes]
    per_ref = []

    for i, (_, ref_vec) in enumerate(reference.ensemble.iterrows()):
        ref_i = ReferenceFlux(fluxes=ref_vec, growth=reference.growth,
                              mode=f"{reference.mode}[{i}]")
        summary, _ = single_gene_screen(model, ref_i, formulation=formulation,
                                        genes=gene_ids, biomass_id=biomass.id,
                                        keep_fluxes=False, progress=False)
        per_ref.append(summary["growth_ratio"].rename(i))

    wide = pd.concat(per_ref, axis=1)
    return pd.DataFrame({
        "growth_ratio_min": wide.min(axis=1),
        "growth_ratio_median": wide.median(axis=1),
        "growth_ratio_max": wide.max(axis=1),
        "growth_ratio_spread": wide.max(axis=1) - wide.min(axis=1),
        "n_references": wide.notna().sum(axis=1),
    })


def call_lethality(summary: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Apply the lethality cut, keeping the continuous ratio visible.

    The threshold is a *reporting* cut, not a stored-away decision, so its
    sensitivity stays auditable.
    """
    out = summary.copy()
    out["lethal"] = out["growth_ratio"] < fraction
    out["lethal_threshold"] = fraction
    return out


def _maybe_tqdm(seq: Iterable, enabled: bool, desc: str):
    if not enabled:
        return seq
    try:
        from tqdm import tqdm

        return tqdm(seq, desc=desc)
    except ImportError:
        return seq
