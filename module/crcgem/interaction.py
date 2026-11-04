"""Genetic-interaction scoring for the anchored double-deletion screen.

The BRCA precedent (defect P3) computed `expected = a + b - a*b` on *relative
growth* values. That is the Bliss-independence form for FRACTION AFFECTED; on
growth (fraction surviving) the independent expectation is `a * b`. As written
the expectation was always >= max(a, b), making the reference systematically
permissive and the resulting combination index inflated. `#DIV/0!` values also
reached the shipped supplementary.

Here the reference model is an explicit argument with no default, both
conventions are implemented correctly, and division guards are applied. Which
model to report is a stated choice, not a buried formula.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Model = Literal["bliss", "multiplicative", "min", "hsa"]
EPS = 1e-12


def expected_growth(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray,
                    model: Model) -> pd.Series | np.ndarray:
    """Expected double-KO relative growth under a stated independence model.

    Args:
        a, b: single-KO relative growth (1.0 = unaffected, 0.0 = no growth).
        model:
          'bliss' / 'multiplicative' — a * b. Independence on the surviving
              fraction; the correct Bliss form when the quantity is growth.
          'min' / 'hsa' — min(a, b). Highest Single Agent: the combination is
              expected to be no worse than its better single agent.
    """
    if model in ("bliss", "multiplicative"):
        return a * b
    if model in ("min", "hsa"):
        return np.minimum(a, b)
    raise ValueError(f"unknown interaction model {model!r}")


def score(single_a: pd.Series, single_b: pd.Series, double: pd.Series, *,
          model: Model, eps: float = EPS) -> pd.DataFrame:
    """Score genetic interaction for each partner gene.

    Returns a frame with the expected growth, the observed double-KO growth,
    the additive deviation, and a guarded ratio.

    `deviation = expected - observed`: positive means the double KO is WORSE
    than independence predicts, i.e. synergy / synthetic lethality.
    `ratio = observed / expected`: below 1 means synergy. Guarded against the
    division-by-zero that reached the BRCA supplementary.
    """
    idx = single_a.index.intersection(single_b.index).intersection(double.index)
    a, b, ab = single_a[idx], single_b[idx], double[idx]

    exp = expected_growth(a, b, model)
    denom = np.where(np.abs(exp) < eps, np.nan, exp)

    out = pd.DataFrame({
        "single_a": a,
        "single_b": b,
        "double_observed": ab,
        "expected": exp,
        "deviation": exp - ab,
        "ratio": ab / denom,
        "model": model,
    }, index=idx)
    out["synergistic"] = out["deviation"] > 0
    return out.sort_values("deviation", ascending=False)


def synthetic_lethal_calls(scores: pd.DataFrame, *, lethal_fraction: float,
                           min_deviation: float,
                           require_single_viable: bool = True) -> pd.DataFrame:
    """Apply the synthetic-lethality definition explicitly.

    A pair is called synthetic lethal when the double KO is lethal while each
    single KO is not — the defining asymmetry. `require_single_viable` enforces
    it; without it, pairs that merely contain an essential gene are counted,
    which is the commonest way an SL screen inflates its hit list.
    """
    out = scores.copy()
    double_lethal = out["double_observed"] < lethal_fraction
    singles_viable = ((out["single_a"] >= lethal_fraction) &
                      (out["single_b"] >= lethal_fraction))
    out["double_lethal"] = double_lethal
    out["singles_viable"] = singles_viable
    out["synthetic_lethal"] = (
        double_lethal & (out["deviation"] > min_deviation)
        & (singles_viable if require_single_viable else True)
    )
    out.attrs["criteria"] = {
        "lethal_fraction": lethal_fraction,
        "min_deviation": min_deviation,
        "require_single_viable": require_single_viable,
    }
    return out
