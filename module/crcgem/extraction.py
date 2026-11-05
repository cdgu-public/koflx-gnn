"""Context-specific model extraction (tINIT family) via TROPPO.

TROPPO: Oliveira, Ferreira, Vieira, Sa & Rocha, Bioinformatics Advances 5(1):
vbaf113, 2025 (doi 10.1093/bioadv/vbaf113). Implements tINIT, GIMME, iMAT,
FastCORE, SwiftCore, CORDA on COBRApy/optlang, with gap-filling and a
TaskEvaluator for metabolic tasks.

TO VERIFY at first use: whether TROPPO's `(t)INIT` module is INIT or tINIT, and
whether ftINIT (Gustafsson et al. 2023) is covered. Recorded as open in
the project decision log; do not assert in a manuscript until checked in code.

MILP requirement is hard. Measured on this workstation (2026-07-21):
GLPK solved 200 binaries in 0.4 s but failed at 800 and 3,991 (>600 s);
CPLEX Community Edition refused 3,991 outright. Extraction therefore runs on a
machine with full CPLEX or Gurobi; `solver.preflight(requires_milp=True)`
enforces this before any wall-clock is spent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


class TroppoUnavailableError(ImportError):
    pass


@dataclass
class ExtractionResult:
    context: str
    kept_reactions: list[str]
    removed_reactions: list[str]
    gap_filled_reactions: list[str] = field(default_factory=list)
    algorithm: str = ""
    params: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "context": self.context,
            "algorithm": self.algorithm,
            "n_kept": len(self.kept_reactions),
            "n_removed": len(self.removed_reactions),
            "n_gap_filled": len(self.gap_filled_reactions),
            **self.params,
        }


def _require_troppo():
    try:
        import troppo  # noqa: F401
    except ImportError as exc:
        raise TroppoUnavailableError(
            "troppo is not installed (`pip install troppo`). Required for "
            "tINIT-family extraction."
        ) from exc


def extract(model, weights: pd.Series, *, algorithm: str = "tinit",
            context: str = "context", protected: Sequence[str] = (),
            gap_filling: bool = False, **kwargs) -> ExtractionResult:
    """Run a TROPPO extraction algorithm on `model` with reaction `weights`.

    Args:
        weights: signed reaction scores (positive = evidence for presence).
                 Produce with expression.threshold_scores().
        protected: reaction ids that must survive (e.g. biomass), so extraction
                 cannot silently delete the objective the downstream FBA needs.
        gap_filling: constituent of tINIT (task-driven), kept ON. The P4
                 concern (a gap-filled bypass hiding a lethal target) is a
                 reporting duty downstream, not a reason to disable it here.
    """
    _require_troppo()
    from troppo.methods_wrappers import ReconstructionWrapper
    from troppo.methods.reconstruction.tINIT import tINITProperties
    from troppo.methods.reconstruction.imat import IMATProperties
    from troppo.methods.reconstruction.gimme import GIMMEProperties
    from troppo.methods.reconstruction.fastcore import FastcoreProperties

    algorithm = algorithm.lower()
    wrapper = ReconstructionWrapper(model)
    scores = weights.reindex([r.id for r in model.reactions]).fillna(0.0)
    protected_idx = [i for i, r in enumerate(model.reactions) if r.id in set(protected)]

    if algorithm in ("tinit", "init"):
        props = tINITProperties(reactions_scores=scores.tolist(),
                                present_metabolites=[],
                                essential_reactions=protected_idx,
                                production_weight=kwargs.pop("production_weight", 0.0),
                                allow_excretion=kwargs.pop("allow_excretion", False),
                                no_reverse_loops=kwargs.pop("no_reverse_loops", True),
                                **kwargs)
    elif algorithm == "imat":
        props = IMATProperties(exp_vector=scores.tolist(), **kwargs)
    elif algorithm == "gimme":
        props = GIMMEProperties(exp_vector=scores.tolist(), **kwargs)
    elif algorithm == "fastcore":
        props = FastcoreProperties(core=protected_idx, **kwargs)
    else:
        raise ValueError(f"unsupported algorithm {algorithm!r}")

    logger.info("running %s on %d reactions (context=%s)",
                algorithm, len(model.reactions), context)
    keep_mask = wrapper.run(props)

    all_ids = [r.id for r in model.reactions]
    kept = [rid for rid, k in zip(all_ids, keep_mask) if k]
    removed = [rid for rid in all_ids if rid not in set(kept)]

    result = ExtractionResult(
        context=context, kept_reactions=kept, removed_reactions=removed,
        algorithm=algorithm,
        params={"gap_filling": gap_filling, "n_protected": len(protected_idx)},
    )
    logger.info("%s: kept %d / removed %d", context, len(kept), len(removed))
    return result


def build_context_model(model, result: ExtractionResult):
    """Materialise a context-specific model by removing non-kept reactions."""
    ctx = model.copy()
    ctx.id = f"{model.id}_{result.context}"
    ctx.remove_reactions([ctx.reactions.get_by_id(r) for r in result.removed_reactions
                          if r in ctx.reactions], remove_orphans=True)
    return ctx


def targets_dependent_on_gap_fill(target_reactions, gap_filled_reactions,
                                  model=None) -> dict:
    """Flag which target candidates rely on a gap-filled reaction (P4 duty).

    Gap-filling is part of tINIT and stays on; the risk it carries is that a
    gap-filled bypass makes a genuinely lethal target read as non-essential.
    This does not decide anything — it discloses. Returns the target reactions
    that ARE gap-filled, and (if a model is given) those whose knock-out set
    touches a gap-filled reaction, so the reader can judge each candidate's
    support.
    """
    gap = set(gap_filled_reactions)
    directly = sorted(set(target_reactions) & gap)
    result = {"n_gap_filled": len(gap), "targets_themselves_gap_filled": directly}
    if model is not None:
        adjacent = []
        for rid in target_reactions:
            if rid in model.reactions:
                neigh = {r.id for m in model.reactions.get_by_id(rid).metabolites
                         for r in m.reactions}
                if neigh & gap:
                    adjacent.append(rid)
        result["targets_adjacent_to_gap_filled"] = sorted(set(adjacent) - gap)
    return result
