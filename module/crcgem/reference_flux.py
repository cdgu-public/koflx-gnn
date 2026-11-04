"""Wild-type reference flux for MOMA, and the degeneracy problem it inherits.

User ruling: reference = WT FBA solution; growth-max is justified because the
context is cancer. Lethality = KO growth < 5% of WT FBA optimum.

MEASURED complication (result/pilot_pfba_degeneracy/): the FBA optimal *value*
is unique, but the optimal *vector* is not, and MOMA minimises distance from
that vector — so KO growth depends on which optimum is handed to it. pFBA
removes most of this at E. coli-core scale (2/95 reactions still free) but NOT
at human-GEM scale: on BRCA functional_MDAMB231, 1,261/3,991 reactions (31.6%,
1,080 with a GPR) still had non-zero range on the parsimonious optimal face.
The residue is dominated by transporters and isozyme/compartment duplicates —
exactly the classes synthetic lethality is defined on. DHFR was in the set.

Three reference modes are therefore offered; the project must choose one
explicitly (config.moma.reference_mode) rather than defaulting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TOL = 1e-6


@dataclass
class ReferenceFlux:
    """A WT reference for MOMA, plus the provenance needed to defend it."""

    fluxes: pd.Series            # reference vector, indexed by reaction id
    growth: float                # WT FBA optimum (unique value)
    mode: str                    # how `fluxes` was chosen
    degenerate_reactions: Optional[list[str]] = None  # non-unique on the face
    ensemble: Optional[pd.DataFrame] = None           # face_ensemble mode only

    @property
    def n_degenerate(self) -> int:
        return len(self.degenerate_reactions or [])

    def summary(self) -> dict:
        return {
            "mode": self.mode,
            "growth": self.growth,
            "n_reactions": len(self.fluxes),
            "n_degenerate_on_face": self.n_degenerate,
            "ensemble_size": None if self.ensemble is None else len(self.ensemble),
        }


def _objective_reactions(model):
    return [r for r in model.reactions if r.objective_coefficient != 0]


def _total_flux_expression(model):
    return sum(r.forward_variable + r.reverse_variable for r in model.reactions)


def _pin_to_parsimonious_face(model, growth: float):
    """Constrain model to {growth = FBA optimum} ∩ {total flux = pFBA minimum}.

    Mutates `model` — call inside a `with model:` context.
    Returns the minimal total flux.
    """
    for r in _objective_reactions(model):
        r.lower_bound = growth * (1 - 1e-9)

    total = _total_flux_expression(model)
    model.objective = model.problem.Objective(total, direction="min")
    sol = model.optimize()
    if sol.status != "optimal":
        raise RuntimeError(f"pFBA failed: status={sol.status}")
    t_min = sol.objective_value

    model.add_cons_vars(
        [model.problem.Constraint(total, lb=0, ub=t_min * (1 + 1e-7),
                                  name="crcgem_pfba_cap")]
    )
    return t_min


def face_degeneracy(model, growth: float, tol: float = TOL) -> pd.DataFrame:
    """FVA over the parsimonious optimal face.

    Reactions with a non-zero range were NOT uniquely determined by pFBA.
    Returns a frame with minimum/maximum/range, sorted by range descending.
    """
    from cobra.flux_analysis import flux_variability_analysis

    with model:
        _pin_to_parsimonious_face(model, growth)
        fva = flux_variability_analysis(model, fraction_of_optimum=1.0, processes=1)
    fva["range"] = (fva["maximum"] - fva["minimum"]).abs()
    return fva.sort_values("range", ascending=False)


def compute(model, mode: str, *, ensemble_n: Optional[int] = None,
            seed: int = 0, diagnose: bool = True,
            thinning: int = 100) -> ReferenceFlux:
    """Build the WT reference flux for a PROLIFERATIVE state (growth-max justified).

    For cancer / adenoma: growth-max is meaningful, so the reference sits on the
    FBA optimum. The optimal *vector* is non-unique, hence:

    Args:
        mode: 'pfba'            — single parsimonious solution (cheapest; leaves
                                  ~1/3 of reactions solver-dependent at GEM scale)
              'chebyshev_face'  — analytic centre of the parsimonious face
                                  (deterministic, defensible; preferred)
              'face_ensemble'   — sample the face, return an ensemble so KO
                                  effects can be reported as intervals
        diagnose: also run FVA over the face to record which reactions are
                  non-unique. Costs one FVA (~130 s at 4k reactions, GLPK).

    For NON-proliferative states (normal, disease) use `sample_references`
    instead — there is no growth objective to anchor a reference to.
    """
    sol = model.optimize()
    if sol.status != "optimal":
        raise RuntimeError(f"WT FBA failed: status={sol.status}")
    growth = sol.objective_value
    logger.info("WT FBA optimum = %.10g", growth)

    degenerate = None
    if diagnose:
        fva = face_degeneracy(model, growth)
        degenerate = fva.index[fva["range"] > TOL].tolist()
        logger.info("%d/%d reactions non-unique on the parsimonious face (%.1f%%)",
                    len(degenerate), len(model.reactions),
                    100 * len(degenerate) / len(model.reactions))

    if mode == "pfba":
        fluxes, ens = _reference_pfba(model, growth), None
    elif mode == "chebyshev_face":
        fluxes, ens = _reference_chebyshev(model, growth), None
    elif mode == "face_ensemble":
        if not ensemble_n:
            raise ValueError("face_ensemble mode requires ensemble_n")
        ens = _reference_ensemble(model, growth, ensemble_n, seed)
        fluxes = ens.mean(axis=0)
    else:
        raise ValueError(f"unknown reference mode {mode!r}")

    return ReferenceFlux(fluxes=fluxes, growth=growth, mode=mode,
                         degenerate_reactions=degenerate, ensemble=ens)


def sample_references(model, *, n: int, thinning: int = 100, seed: int = 0,
                      method: str = "optgp") -> ReferenceFlux:
    """OptGP-sample the expression-constrained feasible space as a MOMA reference
    ensemble, for a NON-PROLIFERATIVE state (normal, non-adenoma disease).

    User ruling 2026-07-27: these states have no justified growth objective, so
    the reference is not an FBA optimum but a *sample of the whole feasible
    polytope* the expression bounds define. MOMA is then run from each sampled
    flux (ensemble MOMA), giving a distribution of KO effects. This differs from
    `face_ensemble`, which samples only the parsimonious OPTIMAL face; here the
    growth objective is not imposed at all.

    `growth` is recorded (from an unconstrained FBA read) for reference only; it
    is not used to constrain the sampling.
    """
    from cobra.sampling import sample

    try:
        growth = model.slim_optimize()
    except Exception:
        growth = float("nan")

    ens = sample(model, n, method=method, thinning=thinning, seed=seed)
    logger.info("sampled %d reference fluxes (non-proliferative, %s, thinning=%d)",
                n, method, thinning)
    return ReferenceFlux(fluxes=ens.mean(axis=0), growth=growth,
                         mode=f"sampling_ensemble[{method},n={n}]",
                         degenerate_reactions=None, ensemble=ens)


def _reference_pfba(model, growth: float) -> pd.Series:
    from cobra.flux_analysis import pfba

    with model:
        for r in _objective_reactions(model):
            r.lower_bound = growth * (1 - 1e-9)
        return pfba(model).fluxes.copy()


def _reference_chebyshev(model, growth: float) -> pd.Series:
    """Chebyshev centre of the parsimonious optimal face.

    Maximise r subject to every bounded reaction staying r away from its own
    bound. Gives the deterministic 'most interior' point of the face rather
    than an arbitrary vertex, which is what a solver would otherwise return.

    Same device as the quantum_gem PHR fix (Bug 15).
    """
    with model:
        _pin_to_parsimonious_face(model, growth)

        r = model.problem.Variable("crcgem_cheb_r", lb=0)
        cons = []
        for rxn in model.reactions:
            lb, ub = rxn.lower_bound, rxn.upper_bound
            if np.isfinite(lb) and lb > -1e29:
                cons.append(model.problem.Constraint(
                    rxn.flux_expression - r, lb=lb, name=f"cheb_lo_{rxn.id}"))
            if np.isfinite(ub) and ub < 1e29:
                cons.append(model.problem.Constraint(
                    rxn.flux_expression + r, ub=ub, name=f"cheb_hi_{rxn.id}"))
        model.add_cons_vars([r] + cons)
        model.objective = model.problem.Objective(r, direction="max")

        sol = model.optimize()
        if sol.status != "optimal":
            raise RuntimeError(f"Chebyshev centre LP failed: status={sol.status}")
        logger.info("Chebyshev radius on parsimonious face = %.6g",
                    sol.objective_value)
        return sol.fluxes.copy()


def _reference_ensemble(model, growth: float, n: int, seed: int) -> pd.DataFrame:
    """Sample the parsimonious optimal face to get a reference ensemble."""
    from cobra.sampling import sample

    with model:
        _pin_to_parsimonious_face(model, growth)
        return sample(model, n, method="optgp", seed=seed)
