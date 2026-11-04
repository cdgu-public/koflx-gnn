"""Expression -> reaction score, and expression-derived flux bounds.

Two jobs, kept separate because they serve different tracks:
  * `reaction_scores`  — GPR-aware gene->reaction mapping, used to feed
    extraction (tINIT family) on depth-adequate data.
  * `eflux_bounds`     — E-Flux style bound scaling (Colijn et al. 2009), used
    to set per-context/per-cell bounds on a FIXED scaffold.

GPR evaluation follows the standard convention: AND -> min (a complex is limited
by its scarcest subunit), OR -> max (isozymes are alternatives). The BRCA
precedent bypassed GPR entirely; that is defect P1 and is not repeated.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"\(|\)|\band\b|\bor\b|[^\s()]+", re.IGNORECASE)


def eval_gpr(rule: str, gene_values: Mapping[str, float],
             missing: float = np.nan) -> float:
    """Evaluate a GPR string to a scalar with AND=min, OR=max.

    Genes absent from `gene_values` contribute `missing`; a term made entirely
    of missing genes stays NaN rather than silently becoming 0, so "no data"
    and "not expressed" remain distinguishable.
    """
    rule = (rule or "").strip()
    if not rule:
        return np.nan
    tokens = _TOKEN.findall(rule)
    pos = 0

    def parse_or():
        nonlocal pos
        vals = [parse_and()]
        while pos < len(tokens) and tokens[pos].lower() == "or":
            pos += 1
            vals.append(parse_and())
        vals = [v for v in vals if not np.isnan(v)]
        return np.nan if not vals else float(np.max(vals))

    def parse_and():
        nonlocal pos
        vals = [parse_atom()]
        while pos < len(tokens) and tokens[pos].lower() == "and":
            pos += 1
            vals.append(parse_atom())
        vals = [v for v in vals if not np.isnan(v)]
        return np.nan if not vals else float(np.min(vals))

    def parse_atom():
        nonlocal pos
        if pos >= len(tokens):
            return np.nan
        tok = tokens[pos]
        if tok == "(":
            pos += 1
            val = parse_or()
            if pos < len(tokens) and tokens[pos] == ")":
                pos += 1
            return val
        pos += 1
        v = gene_values.get(tok, missing)
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    return parse_or()


def reaction_scores(model, gene_expression: pd.Series,
                    missing: float = np.nan) -> pd.Series:
    """Map a gene-level expression vector onto reactions via GPR."""
    values = gene_expression.to_dict()
    return pd.Series(
        {r.id: eval_gpr(r.gene_reaction_rule, values, missing)
         for r in model.reactions},
        name=gene_expression.name,
    )


def threshold_scores(scores: pd.Series, strategy: str,
                     value: Optional[float] = None) -> pd.Series:
    """Convert reaction scores to extraction weights (positive = keep).

    strategy:
      'global_percentile' — value in [0,100]; one cutoff across all reactions
      'absolute'          — value is the raw expression cutoff
      'local_t1_t2'       — two-tier (value = (t1, t2)); not implemented here,
                            deferred until the thresholding scheme is ratified

    The BRCA precedent's thresholds were unrecoverable from its own archive.
    Whatever is chosen here must be stated in config and lands in the manifest.
    """
    s = scores.dropna()
    if strategy == "global_percentile":
        if value is None:
            raise ValueError("global_percentile requires a percentile value")
        cutoff = np.percentile(s, value)
    elif strategy == "absolute":
        if value is None:
            raise ValueError("absolute requires a cutoff value")
        cutoff = value
    else:
        raise NotImplementedError(
            f"threshold strategy {strategy!r} not implemented; "
            "ratify the scheme before use")
    logger.info("threshold strategy=%s value=%s -> cutoff=%.6g "
                "(%d/%d reactions above)",
                strategy, value, cutoff, int((scores > cutoff).sum()), len(scores))
    return scores - cutoff


def eflux_bounds(model, scores: pd.Series, *, scale: float = 1000.0,
                 default: Optional[float] = None,
                 preserve_reversibility: bool = True) -> dict[str, tuple[float, float]]:
    """E-Flux style bounds: |v| <= scale * normalised expression.

    Reference: Colijn et al., PLoS Comput Biol 2009 (E-Flux).
    Reactions with no expression evidence keep `default` (or their original
    bound if default is None) rather than being forced to zero.

    Returns {reaction_id: (lower, upper)} — applied by the caller so the caller
    controls whether it mutates a model or a copy.
    """
    finite = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        raise ValueError("no finite reaction scores")
    top = float(finite.max())
    if top <= 0:
        raise ValueError("max reaction score must be positive to normalise")

    out: dict[str, tuple[float, float]] = {}
    for rxn in model.reactions:
        s = scores.get(rxn.id, np.nan)
        if np.isnan(s):
            if default is None:
                continue
            cap = default
        else:
            cap = scale * max(float(s), 0.0) / top
        lo = -cap if (rxn.lower_bound < 0 and preserve_reversibility) else max(rxn.lower_bound, -cap)
        hi = cap if rxn.upper_bound > 0 else min(rxn.upper_bound, cap)
        out[rxn.id] = (max(lo, -cap), min(hi, cap))
    return out


def apply_bounds(model, bounds: Mapping[str, tuple[float, float]]):
    """Apply bounds in place. Use inside `with model:` to keep it reversible."""
    for rid, (lo, hi) in bounds.items():
        try:
            model.reactions.get_by_id(rid).bounds = (lo, hi)
        except KeyError:
            logger.debug("reaction %s not in model; bound skipped", rid)
    return model
