"""Unit tests exercising the logic that must not silently regress.

Runs on E. coli core (95 reactions) so it is fast and solver-light.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crcgem import config, distribution, expression, interaction, reference_flux, sampling

ECOLI = "textbook"  # E. coli core model bundled with COBRApy (cobra.io.load_model)


@pytest.fixture(scope="module")
def model():
    cobra = pytest.importorskip("cobra")
    m = cobra.io.load_model(ECOLI)
    m.solver = "glpk"
    return m


# ---------------------------------------------------------------- config gating
def test_open_decisions_have_no_silent_default():
    """An unratified decision must abort the run, not pick a value."""
    with pytest.raises(config.ConfigError):
        config.SamplingConfig().validate()
    with pytest.raises(config.ConfigError):
        config.MomaConfig().validate()
    with pytest.raises(config.ConfigError):
        config.ExtractionConfig().validate()


def test_sampling_n_restricted_to_predeclared_values():
    c = config.SamplingConfig(n_samples=137, thinning=100, n_chains=4, seed=0,
                              r_hat_max=1.05, ess_min=100,
                              on_nonconvergence="abort")
    with pytest.raises(config.ConfigError):
        c.validate()
    c.n_samples = 200
    c.validate()


def test_gpr_aware_deletion_cannot_be_silently_disabled():
    c = config.MomaConfig(formulation="linear", proliferative_statuses=["CRC"],
                          proliferative_reference="pfba",
                          nonproliferative_readout="not_applicable",
                          gpr_aware_deletion=False)
    with pytest.raises(config.ConfigError, match="P1"):
        c.validate()


# ------------------------------------------------------------------ GPR logic
@pytest.mark.parametrize("rule,vals,expected", [
    ("a", {"a": 3.0}, 3.0),
    ("a and b", {"a": 3.0, "b": 1.0}, 1.0),          # complex -> min
    ("a or b", {"a": 3.0, "b": 1.0}, 3.0),           # isozymes -> max
    ("(a or b) and c", {"a": 3.0, "b": 1.0, "c": 2.0}, 2.0),
    ("a and (b or c)", {"a": 5.0, "b": 1.0, "c": 4.0}, 4.0),
    ("", {}, None),
])
def test_eval_gpr(rule, vals, expected):
    got = expression.eval_gpr(rule, vals)
    if expected is None:
        assert np.isnan(got)
    else:
        assert got == pytest.approx(expected)


def test_eval_gpr_missing_gene_is_nan_not_zero():
    """'no data' must stay distinguishable from 'not expressed'."""
    assert np.isnan(expression.eval_gpr("ghost", {"a": 1.0}))
    # a present isozyme rescues the term rather than being dragged to 0
    assert expression.eval_gpr("ghost or a", {"a": 2.0}) == pytest.approx(2.0)


# ------------------------------------------------------ interaction scoring (P3)
def test_bliss_on_growth_is_multiplicative_not_additive():
    """The precedent used a+b-ab on growth; independence on growth is a*b."""
    a = pd.Series([0.98], index=["g"])
    b = pd.Series([0.35], index=["g"])
    exp = interaction.expected_growth(a, b, "bliss")
    assert exp.iloc[0] == pytest.approx(0.343)
    wrong = a + b - a * b            # the BRCA formula
    assert wrong.iloc[0] > max(a.iloc[0], b.iloc[0])   # always permissive


def test_interaction_score_guards_division_by_zero():
    s = interaction.score(pd.Series([0.0], index=["g"]),
                          pd.Series([0.0], index=["g"]),
                          pd.Series([0.0], index=["g"]), model="bliss")
    assert np.isnan(s["ratio"].iloc[0])      # not inf, not #DIV/0!


def test_synthetic_lethal_requires_viable_singles():
    scores = interaction.score(
        pd.Series([0.9, 0.01], index=["viable", "essential"]),
        pd.Series([0.9, 0.9], index=["viable", "essential"]),
        pd.Series([0.01, 0.005], index=["viable", "essential"]),
        model="bliss")
    calls = interaction.synthetic_lethal_calls(
        scores, lethal_fraction=0.05, min_deviation=0.1)
    assert calls.loc["viable", "synthetic_lethal"]
    # a pair containing an already-essential gene is NOT synthetic lethality
    assert not calls.loc["essential", "synthetic_lethal"]


# -------------------------------------------------------------- reference flux
def test_reference_flux_and_face_degeneracy(model):
    ref = reference_flux.compute(model, "pfba", diagnose=True)
    assert ref.growth > 0
    assert len(ref.fluxes) == len(model.reactions)
    # E. coli core: pFBA nearly unique (only the SUCDi/FRD7 loop remains)
    assert ref.n_degenerate <= 5


def test_chebyshev_reference_is_deterministic(model):
    a = reference_flux.compute(model, "chebyshev_face", diagnose=False)
    b = reference_flux.compute(model, "chebyshev_face", diagnose=False)
    np.testing.assert_allclose(a.fluxes.to_numpy(), b.fluxes.to_numpy(), atol=1e-6)


# -------------------------------------------------------------------- sampling
def test_variance_decomposition_separates_between_from_within():
    rng = np.random.default_rng(0)
    cols = ["r1", "r2"]
    # r1 differs strongly between contexts; r2 does not
    ctx = {
        "A": pd.DataFrame({"r1": rng.normal(0, 0.1, 200), "r2": rng.normal(0, 1, 200)}, columns=cols),
        "B": pd.DataFrame({"r1": rng.normal(5, 0.1, 200), "r2": rng.normal(0, 1, 200)}, columns=cols),
    }
    d = sampling.decompose_variance(ctx)
    assert d.loc["r1", "variance_ratio"] > d.loc["r2", "variance_ratio"]


def test_r_hat_flags_disagreeing_chains():
    rng = np.random.default_rng(0)
    same = [pd.DataFrame({"r": rng.normal(0, 1, 500)}) for _ in range(3)]
    diff = [pd.DataFrame({"r": rng.normal(m, 1, 500)}) for m in (0, 10, 20)]
    assert sampling.convergence_diagnostics(same)["r_hat"].iloc[0] < 1.1
    assert sampling.convergence_diagnostics(diff)["r_hat"].iloc[0] > 1.5


# ---------------------------------------------------------------- identity test
def test_identity_passes_for_same_distribution_and_fails_for_shifted():
    rng = np.random.default_rng(0)
    cols = [f"r{i}" for i in range(10)]
    real = pd.DataFrame(rng.normal(0, 1, (400, 10)), columns=cols)
    same = pd.DataFrame(rng.normal(0, 1, (400, 10)), columns=cols)
    shifted = pd.DataFrame(rng.normal(3, 1, (400, 10)), columns=cols)

    ok = distribution.test_identity(real, same, n_splits=30)
    bad = distribution.test_identity(real, shifted, n_splits=30)
    assert ok.summary["frac_reactions_exceeding_null"] < 0.3
    assert bad.summary["frac_reactions_exceeding_null"] > 0.9
    assert bad.summary["mmd_rbf"] > ok.summary["mmd_rbf"]


def test_identity_rejects_mismatched_scaffolds():
    a = pd.DataFrame({"r1": [1.0, 2.0]})
    b = pd.DataFrame({"other": [1.0, 2.0]})
    with pytest.raises(ValueError, match="scaffold"):
        distribution.test_identity(a, b, n_splits=2)
