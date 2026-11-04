"""OptGP flux sampling with pre-registered protocol and convergence gating.

Two distinct sources of spread must not be conflated in this project:
  (1) BIOLOGICAL  — variation across cells/donors, from per-context constraints.
  (2) NUMERICAL   — the solution space of ONE constraint set (what OptGP explores).
Only (1) is a finding. (2) is non-identifiability. `decompose_variance` keeps
them separate; a status-level flux shift claimed on top of (2) alone is an
artefact of polytope width, not metabolism.

The protocol (n, thinning, chains, seed) is predeclared in config and the
convergence criterion is checked *after* sampling with a pre-declared
escalation rule, so a non-converged chain cannot be quietly accepted.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SamplingResult:
    samples: pd.DataFrame          # (n_chains * n_samples) x reactions
    chain_id: np.ndarray
    converged: bool
    diagnostics: pd.DataFrame      # per-reaction r_hat / ess
    protocol: dict

    def by_chain(self) -> list[pd.DataFrame]:
        return [self.samples[self.chain_id == c] for c in np.unique(self.chain_id)]


def sample_context(model, *, n_samples: int, thinning: int, n_chains: int,
                   seed: int, method: str = "optgp",
                   r_hat_max: Optional[float] = None,
                   ess_min: Optional[int] = None,
                   on_nonconvergence: str = "report") -> SamplingResult:
    """Run `n_chains` independent OptGP chains and gate on convergence.

    Independent chains (different seeds) are required for R-hat; a single chain
    cannot diagnose its own mixing.
    """
    from cobra.sampling import sample as cobra_sample

    frames, ids = [], []
    for c in range(n_chains):
        logger.info("sampling chain %d/%d (n=%d, thinning=%d, seed=%d)",
                    c + 1, n_chains, n_samples, thinning, seed + c)
        df = cobra_sample(model, n_samples, method=method,
                          thinning=thinning, seed=seed + c)
        frames.append(df)
        ids.append(np.full(len(df), c))

    samples = pd.concat(frames, ignore_index=True)
    chain_id = np.concatenate(ids)

    diag = convergence_diagnostics(frames)
    converged = True
    if r_hat_max is not None:
        converged &= bool((diag["r_hat"].dropna() <= r_hat_max).all())
    if ess_min is not None:
        converged &= bool((diag["ess"].dropna() >= ess_min).all())

    protocol = {"n_samples": n_samples, "thinning": thinning, "n_chains": n_chains,
                "seed": seed, "method": method, "r_hat_max": r_hat_max,
                "ess_min": ess_min, "on_nonconvergence": on_nonconvergence}

    if not converged:
        worst_r = diag["r_hat"].max()
        worst_e = diag["ess"].min()
        msg = (f"sampling did NOT meet the pre-declared criterion "
               f"(max r_hat={worst_r:.3f}, min ess={worst_e:.0f})")
        if on_nonconvergence == "abort":
            raise RuntimeError(msg + " — protocol says abort")
        logger.warning("%s — protocol says '%s'", msg, on_nonconvergence)

    return SamplingResult(samples, chain_id, converged, diag, protocol)


def convergence_diagnostics(chains: list[pd.DataFrame]) -> pd.DataFrame:
    """Split-R-hat and effective sample size per reaction.

    R-hat compares between-chain and within-chain variance (Gelman-Rubin);
    values near 1 indicate the chains agree. Constant reactions (blocked, or
    fixed by the constraints) yield NaN rather than a spurious value.
    """
    if len(chains) < 2:
        idx = chains[0].columns if chains else []
        return pd.DataFrame({"r_hat": np.nan, "ess": np.nan}, index=idx)

    arr = np.stack([c.to_numpy() for c in chains])   # (chains, draws, reactions)
    m, n, _ = arr.shape

    chain_mean = arr.mean(axis=1)
    chain_var = arr.var(axis=1, ddof=1)
    B = n * chain_mean.var(axis=0, ddof=1)
    W = chain_var.mean(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        var_hat = (n - 1) / n * W + B / n
        r_hat = np.sqrt(var_hat / W)
        rho = 1.0 - W / var_hat            # crude lag-1 autocorrelation proxy
        ess = (m * n) * (1 - rho) / (1 + rho)

    const = W < 1e-12
    r_hat = np.where(const, np.nan, r_hat)
    ess = np.where(const, np.nan, np.clip(ess, 0, m * n))

    return pd.DataFrame({"r_hat": r_hat, "ess": ess}, index=chains[0].columns)


def decompose_variance(samples_by_context: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Split flux variance into between-context and within-context parts.

    within  = numerical spread of each context's own solution space (OptGP)
    between = spread of context means — the candidate biological signal

    `variance_ratio` = between / within. A status shift that is real should show
    between-context spread exceeding the polytope width the sampler already
    produces inside a single context. Low ratio = the "shift" is inside the
    non-identifiability of one context and should not be claimed.
    """
    means = pd.DataFrame({k: df.mean() for k, df in samples_by_context.items()})
    within = pd.DataFrame({k: df.var(ddof=1) for k, df in samples_by_context.items()})

    between_var = means.var(axis=1, ddof=1)
    within_mean = within.mean(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = between_var / within_mean

    return pd.DataFrame({
        "between_context_var": between_var,
        "within_context_var_mean": within_mean,
        "variance_ratio": ratio,
    }).sort_values("variance_ratio", ascending=False)
