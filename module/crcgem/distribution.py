"""Real vs generated identity testing on flux distributions.

This is the validation this chapter owes. The previous chapter (scTrx) validates
the generated transcriptome; here the claim must be re-established AFTER
context-specific GEM construction, i.e. on flux.

Test design (per-reaction, on a shared scaffold so coordinates are comparable):
  * distance   — Wasserstein-1 per reaction, plus optional multivariate MMD
  * null       — split-half of REAL cells gives the empirical floor for "same",
                 since two halves of one population are identical by
                 construction; anything at or below that is indistinguishable
  * positive control — real HN vs real C must be SEPARABLE by the same metric.
                 Without it, a "pass" may only mean the metric is insensitive.

The positive control is not optional. A test that cannot detect a difference
that certainly exists cannot support a claim that a difference is absent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IdentityReport:
    per_reaction: pd.DataFrame      # distance + null quantile per reaction
    null_distribution: pd.DataFrame
    positive_control: Optional[pd.DataFrame]
    summary: dict

    def failed_reactions(self, alpha: float = 0.05) -> pd.Index:
        """Reactions whose real-vs-generated distance exceeds the null band."""
        return self.per_reaction.index[self.per_reaction["exceeds_null"]]


def wasserstein_per_reaction(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """Wasserstein-1 distance per shared reaction."""
    from scipy.stats import wasserstein_distance

    shared = a.columns.intersection(b.columns)
    return pd.Series(
        {c: wasserstein_distance(a[c].to_numpy(), b[c].to_numpy()) for c in shared},
        name="wasserstein",
    )


def split_half_null(real: pd.DataFrame, *, n_splits: int = 100,
                    seed: int = 0) -> pd.DataFrame:
    """Null band from repeated random halves of the REAL cells.

    This is the distance one expects when two samples genuinely come from the
    same distribution at this sample size — the floor against which the
    real-vs-generated distance must be judged.
    """
    rng = np.random.default_rng(seed)
    n = len(real)
    draws = []
    for _ in range(n_splits):
        idx = rng.permutation(n)
        half = n // 2
        draws.append(wasserstein_per_reaction(real.iloc[idx[:half]],
                                              real.iloc[idx[half:2 * half]]))
    null = pd.concat(draws, axis=1)
    return pd.DataFrame({
        "null_median": null.median(axis=1),
        "null_q95": null.quantile(0.95, axis=1),
        "null_q99": null.quantile(0.99, axis=1),
    })


def mmd_rbf(a: np.ndarray, b: np.ndarray, *, gamma: Optional[float] = None,
            max_n: int = 2000, seed: int = 0) -> float:
    """Unbiased squared MMD with an RBF kernel (multivariate whole-flux test).

    gamma defaults to the median-heuristic bandwidth.
    """
    rng = np.random.default_rng(seed)
    if len(a) > max_n:
        a = a[rng.choice(len(a), max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), max_n, replace=False)]

    from scipy.spatial.distance import cdist

    if gamma is None:
        pooled = np.vstack([a, b])
        med = np.median(cdist(pooled, pooled, "sqeuclidean"))
        gamma = 1.0 / med if med > 0 else 1.0

    kaa = np.exp(-gamma * cdist(a, a, "sqeuclidean"))
    kbb = np.exp(-gamma * cdist(b, b, "sqeuclidean"))
    kab = np.exp(-gamma * cdist(a, b, "sqeuclidean"))
    na, nb = len(a), len(b)
    np.fill_diagonal(kaa, 0.0)
    np.fill_diagonal(kbb, 0.0)
    return float(kaa.sum() / (na * (na - 1)) + kbb.sum() / (nb * (nb - 1))
                 - 2 * kab.mean())


def mahalanobis_between(a: np.ndarray, b: np.ndarray, *, shrinkage: float = 0.1
                        ) -> float:
    """Mahalanobis distance between the means of two distributions.

    Uses a shrunk pooled covariance so it stays invertible when the number of
    reactions exceeds the number of samples (the usual GEM case) — this is why
    the user's instinct to run it "on an embedding" is right; on a reduced
    embedding the covariance is well-conditioned without shrinkage.
    """
    pooled = np.vstack([a, b])
    cov = np.cov(pooled, rowvar=False)
    cov = (1 - shrinkage) * cov + shrinkage * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    diff = a.mean(0) - b.mean(0)
    return float(np.sqrt(diff @ np.linalg.solve(cov, diff)))


#: Named distribution-distance metrics for the therapeutic readout. Pluggable
#: because the choice is under discussion (Pearson-on-means is cheap but
#: discards the fluxome diversity we explicitly want to compare; MMD /
#: Mahalanobis-on-embedding keep it).
def _pearson_dist(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import pearsonr

    r, _ = pearsonr(a.mean(0), b.mean(0))
    return float(1.0 - r)


DISTRIBUTION_METRICS = {
    "mmd": lambda a, b: mmd_rbf(a, b),
    "mahalanobis": mahalanobis_between,
    "pearson_mean": _pearson_dist,   # baseline only; collapses the distribution
}


def normalization_score(disease: pd.DataFrame, normal: pd.DataFrame,
                        disease_perturbed: pd.DataFrame,
                        normal_perturbed: pd.DataFrame, *,
                        metric: str = "mmd", penalty: float = 1.0) -> dict:
    r"""Rank a candidate TREATMENT by how selectively it normalises disease.

    PURPOSE (corrected 2026-10-19, user ruling). The objective is a single
    ANTI-CANCER one and the screen is SEQUENTIAL, cancer first. This function is
    STEP 2, applied to genes that already passed the cancer growth gate:
      * STEP 1 (`moma.call_lethality`): KO in the CRC context model, keep those
        dropping growth below 5% of the wild-type FBA optimum -> cancer-lethal
        gene list.
      * STEP 2 (here): carry that list to each intermediate and to normal and
        check each plays its role -- the intermediate fluxome moves TOWARD
        unperturbed normal, and normal does NOT move.
      * Reverse mode, for validation: start from a disease gold-standard drug and
        require an anti-cancer effect in CRC with no effect in normal.
    This is NOT a separate "treat the disease" objective; the earlier wording to
    that effect is superseded.

    Each D member is evaluated PER ROUTE, separately; a target need not satisfy
    every D. ADENOMA is a separate case: read in the D -> N direction like the
    other intermediates, but it must satisfy D -> N TOGETHER WITH a growth limit,
    and it is the one disease group for which growth maximisation is justified.

    NORMAL REFERENCE (user ruling, standing; upstream D6 / O3). HN and AN are
    separate and never merged. HN is the GLOBAL STANDARD and carries the primary
    score: pass `normal` = HN. AN is an ADDITIONAL EVIDENCE SCORE where a matched
    donor exists, not a co-equal second reference -- so the single `normal`
    argument is correct as it stands. Call this function TWICE: once with HN
    (primary, always) and once with AN (supporting, where matched). Do not pool AN
    into the HN reference and do not combine the two results into one number; AN
    covers only part of the cohort, so a combined score is not comparable across
    groups.
    AN's role is INDIVIDUAL BACKGROUND CORRECTION, and that bounds what it can
    claim: the perturbation is simulated on a context model built from the
    UNPERTURBED transcriptome, so no post-treatment transcriptome is ever observed
    and AN cannot give a direct per-drug outcome. Because correction is the whole
    of its job, AN works as a reference approach across most groups.

    `disease_perturbed` / `normal_perturbed` are the fluxome distributions AFTER
    applying the candidate treatment (gene KO, or a reaction-bound change
    standing for a drug) to the disease / normal context respectively.

    NAMING (corrected 2026-10-19; authority = documentation/terminology.md).
    This is NOT MTA. Cite by component:
      * goal (revert disease toward healthy instead of killing) = MTA,
        Yizhak et al. Nat Commun 2013;4:2632, and rMTA, Valcarcel et al.
        Bioinformatics 2019;35(21):4350-4355 -- framing only, not the algorithm
        implemented here (no MIQP, no R_F/R_B/R_S partition). MTA's own penalty
        is the DENOMINATOR of its score, over reactions that should stay
        untransformed, inside the single diseased state -- not a control arm.
      * two-arm structure (same perturbation on a separate normal model,
        penalized by its effect there) = selectivity score, Yizhak et al.
        eLife 2014;3:e03641, defined there as (Gi_NCI60 - Gi_HapMap)*Gs_HapMap.
        Same structure as the cytostatic score (Folger et al. Mol Syst Biol
        2011;7:501) and the therapeutic window (Bordel, Oncotarget 2018).
      * control-arm readout as flux-distribution displacement = NO established
        published term found; every published form reads the control arm as a
        growth or ATP-production loss. Do not coin a name for it.

    KNOWN DEFECT (measured 2026-10-19, NOT fixed -- methodology, needs a ruling).
    With metric="mmd" and gamma=None, mmd_rbf re-estimates the kernel bandwidth
    by the median heuristic on each pair it is given. The three distances below
    are three separate calls on three different pairs, so each is computed under
    a DIFFERENT bandwidth and they are then added and subtracted. They are not on
    a common footing, so `gap_before - gap_after` is not a well-defined
    improvement and penalty=1.0 is not a meaningful exchange rate. Fixing the
    bandwidth once (e.g. from the D-vs-N pair) and reusing it would resolve it.
    Note also that only "mmd" is a distribution distance: "mahalanobis" and
    "pearson_mean" compare MEANS, so they do not support a distributional claim.

        score = [d(D, N) - d(D|Rx, N)]  -  penalty * d(N|Rx, N)
                 \_____ treatment closes the gap _____/  \__ collateral on normal __/

    All four inputs are fluxome DISTRIBUTIONS on a shared reaction set (the
    common-reaction invariant), so a distribution metric — not a point
    comparison — is used. Positive `selective_score` = a selectively normalising
    (therapeutically promising) perturbation.

    metric: key into DISTRIBUTION_METRICS. Choice is a live discussion;
    'mmd' (kernel, no covariance inversion) is the robust default, 'mahalanobis'
    if working on a fluxome embedding, 'pearson_mean' only as a sanity baseline.
    """
    cols = (disease.columns.intersection(normal.columns)
            .intersection(disease_perturbed.columns)
            .intersection(normal_perturbed.columns))
    if len(cols) == 0:
        raise ValueError("inputs share no reactions — different scaffolds?")
    d = DISTRIBUTION_METRICS[metric]
    D, N = disease[cols].to_numpy(), normal[cols].to_numpy()
    Dk, Nk = disease_perturbed[cols].to_numpy(), normal_perturbed[cols].to_numpy()

    gap_before = d(D, N)
    gap_after = d(Dk, N)
    collateral = d(Nk, N)
    return {
        "metric": metric,
        "n_reactions": int(len(cols)),
        "gap_disease_to_normal": gap_before,
        "gap_perturbed_to_normal": gap_after,
        "normalisation": gap_before - gap_after,       # want large positive
        "collateral_on_normal": collateral,             # want small
        "selective_score": (gap_before - gap_after) - penalty * collateral,
    }


def test_identity(real: pd.DataFrame, generated: pd.DataFrame, *,
                  positive_control: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None,
                  n_splits: int = 100, seed: int = 0,
                  null_quantile: str = "null_q95") -> IdentityReport:
    """Full real-vs-generated identity test on flux distributions.

    Args:
        real, generated: cells x reactions flux matrices on the SAME scaffold.
        positive_control: (real_A, real_B) from two statuses that must differ
            (e.g. real HN vs real C). Omitting it weakens the result to the
            point of being uninterpretable — a null result could equally mean
            the metric is blind.
    """
    shared = real.columns.intersection(generated.columns)
    if len(shared) == 0:
        raise ValueError("real and generated share no reactions — different scaffolds?")
    real, generated = real[shared], generated[shared]

    dist = wasserstein_per_reaction(real, generated)
    null = split_half_null(real, n_splits=n_splits, seed=seed)

    per_rxn = pd.concat([dist, null], axis=1)
    per_rxn["exceeds_null"] = per_rxn["wasserstein"] > per_rxn[null_quantile]

    pc = None
    pc_summary = {}
    if positive_control is not None:
        a, b = positive_control
        cols = shared.intersection(a.columns).intersection(b.columns)
        pc_dist = wasserstein_per_reaction(a[cols], b[cols])
        pc = pd.concat([pc_dist.rename("wasserstein"), null.loc[cols]], axis=1)
        pc["exceeds_null"] = pc["wasserstein"] > pc[null_quantile]
        pc_summary = {
            "positive_control_frac_separable": float(pc["exceeds_null"].mean()),
            "positive_control_n_reactions": int(len(pc)),
        }
        if pc["exceeds_null"].mean() < 0.05:
            logger.warning(
                "POSITIVE CONTROL FAILED: statuses that should differ are not "
                "separable by this metric. A real-vs-generated 'pass' is "
                "uninterpretable under these conditions.")

    frac = float(per_rxn["exceeds_null"].mean())
    summary = {
        "n_reactions": int(len(per_rxn)),
        "n_real_cells": int(len(real)),
        "n_generated_cells": int(len(generated)),
        "frac_reactions_exceeding_null": frac,
        "mmd_rbf": mmd_rbf(real.to_numpy(), generated.to_numpy(), seed=seed),
        "null_quantile_used": null_quantile,
        **pc_summary,
    }
    logger.info("identity test: %.1f%% of reactions exceed the split-half null",
                100 * frac)
    return IdentityReport(per_rxn, null, pc, summary)
