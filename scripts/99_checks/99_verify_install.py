#!/usr/bin/env python
"""Self-contained verification — no pytest required.

Exercises every module against E. coli core and synthetic data, so a fresh
machine (e.g. the CPLEX server) can confirm the package actually runs before a
real job is submitted.

    python script_python/99_verify_install.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module"))

from crcgem import (config, distribution, expression, interaction,  # noqa: E402
                    reference_flux, sampling, solver as solver_mod)

ECOLI = "/data/workspace/cdgu/Winery/seeding/quantum/quantum_gem/data/e_coli_core.xml"
_results: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        try:
            fn()
            _results.append((name, True, ""))
            print(f"  PASS  {name}")
        except Exception as exc:
            _results.append((name, False, traceback.format_exc()))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        return fn
    return deco


def approx(a, b, tol=1e-6):
    assert abs(a - b) < tol, f"{a} != {b}"


# ------------------------------------------------------------------ config
@check("config: open decisions have no silent default")
def _():
    for c in (config.SamplingConfig(), config.MomaConfig(), config.ExtractionConfig()):
        try:
            c.validate()
        except config.ConfigError:
            continue
        raise AssertionError(f"{type(c).__name__} validated while unset")


@check("config: n_samples restricted to predeclared 200/1000")
def _():
    c = config.SamplingConfig(n_samples=137, thinning=100, n_chains=4, seed=0,
                              r_hat_max=1.05, ess_min=100, on_nonconvergence="abort")
    try:
        c.validate()
        raise AssertionError("accepted n_samples=137")
    except config.ConfigError:
        pass
    c.n_samples = 200
    c.validate()


@check("config: gpr_aware_deletion cannot be silently disabled (BRCA P1)")
def _():
    c = config.MomaConfig(formulation="quadratic",
                          proliferative_statuses=["CRC"],
                          proliferative_reference="pfba",
                          nonproliferative_readout="flux_shift",
                          gpr_aware_deletion=False)
    try:
        c.validate()
        raise AssertionError("accepted gpr_aware_deletion=False")
    except config.ConfigError as exc:
        assert "P1" in str(exc)


@check("config: MOMA needs proliferative-status list and non-prolif readout")
def _():
    # missing proliferative_statuses
    c = config.MomaConfig(formulation="quadratic", proliferative_reference="pfba",
                          nonproliferative_readout="flux_shift")
    try:
        c.validate(); raise AssertionError("accepted empty proliferative_statuses")
    except config.ConfigError:
        pass
    # missing nonproliferative_readout
    c = config.MomaConfig(formulation="quadratic", proliferative_statuses=["CRC"],
                          proliferative_reference="pfba")
    try:
        c.validate(); raise AssertionError("accepted missing nonproliferative_readout")
    except config.ConfigError:
        pass
    # complete
    config.MomaConfig(formulation="quadratic", proliferative_statuses=["CRC", "Adenoma"],
                      proliferative_reference="pfba",
                      nonproliferative_readout="flux_shift").validate()


# --------------------------------------------------------------------- GPR
@check("expression: GPR AND=min, OR=max, nesting")
def _():
    approx(expression.eval_gpr("a and b", {"a": 3.0, "b": 1.0}), 1.0)
    approx(expression.eval_gpr("a or b", {"a": 3.0, "b": 1.0}), 3.0)
    approx(expression.eval_gpr("(a or b) and c", {"a": 3.0, "b": 1.0, "c": 2.0}), 2.0)
    approx(expression.eval_gpr("a and (b or c)", {"a": 5.0, "b": 1.0, "c": 4.0}), 4.0)


@check("expression: missing gene is NaN, not 0")
def _():
    assert np.isnan(expression.eval_gpr("ghost", {"a": 1.0}))
    approx(expression.eval_gpr("ghost or a", {"a": 2.0}), 2.0)


# ------------------------------------------------------------- interaction
@check("interaction: Bliss on growth is a*b, not a+b-a*b (BRCA P3)")
def _():
    a = pd.Series([0.98], index=["g"])
    b = pd.Series([0.35], index=["g"])
    approx(interaction.expected_growth(a, b, "bliss").iloc[0], 0.343)
    wrong = (a + b - a * b).iloc[0]
    assert wrong > max(a.iloc[0], b.iloc[0]), "precedent formula should be permissive"


@check("interaction: division guarded (no #DIV/0!)")
def _():
    z = pd.Series([0.0], index=["g"])
    s = interaction.score(z, z, z, model="bliss")
    assert np.isnan(s["ratio"].iloc[0])


@check("interaction: SL requires viable singles")
def _():
    s = interaction.score(pd.Series([0.9, 0.01], index=["viable", "essential"]),
                          pd.Series([0.9, 0.9], index=["viable", "essential"]),
                          pd.Series([0.01, 0.005], index=["viable", "essential"]),
                          model="bliss")
    calls = interaction.synthetic_lethal_calls(s, lethal_fraction=0.05,
                                               min_deviation=0.1)
    assert calls.loc["viable", "synthetic_lethal"]
    assert not calls.loc["essential", "synthetic_lethal"]


# ---------------------------------------------------------------- sampling
@check("sampling: variance decomposition separates between from within")
def _():
    rng = np.random.default_rng(0)
    ctx = {"A": pd.DataFrame({"r1": rng.normal(0, .1, 200), "r2": rng.normal(0, 1, 200)}),
           "B": pd.DataFrame({"r1": rng.normal(5, .1, 200), "r2": rng.normal(0, 1, 200)})}
    d = sampling.decompose_variance(ctx)
    assert d.loc["r1", "variance_ratio"] > d.loc["r2", "variance_ratio"]


@check("sampling: R-hat flags disagreeing chains")
def _():
    rng = np.random.default_rng(0)
    same = [pd.DataFrame({"r": rng.normal(0, 1, 500)}) for _ in range(3)]
    diff = [pd.DataFrame({"r": rng.normal(m, 1, 500)}) for m in (0, 10, 20)]
    assert sampling.convergence_diagnostics(same)["r_hat"].iloc[0] < 1.1
    assert sampling.convergence_diagnostics(diff)["r_hat"].iloc[0] > 1.5


# ------------------------------------------------------------- distribution
@check("distribution: identity passes for same, fails for shifted")
def _():
    rng = np.random.default_rng(0)
    cols = [f"r{i}" for i in range(10)]
    real = pd.DataFrame(rng.normal(0, 1, (400, 10)), columns=cols)
    same = pd.DataFrame(rng.normal(0, 1, (400, 10)), columns=cols)
    shift = pd.DataFrame(rng.normal(3, 1, (400, 10)), columns=cols)
    ok = distribution.test_identity(real, same, n_splits=30)
    bad = distribution.test_identity(real, shift, n_splits=30)
    assert ok.summary["frac_reactions_exceeding_null"] < 0.3, ok.summary
    assert bad.summary["frac_reactions_exceeding_null"] > 0.9, bad.summary
    assert bad.summary["mmd_rbf"] > ok.summary["mmd_rbf"]


@check("distribution: selectivity score rewards move-toward-normal")
def _():
    rng = np.random.default_rng(0)
    cols = [f"r{i}" for i in range(8)]
    normal = pd.DataFrame(rng.normal(0, 1, (200, 8)), columns=cols)
    disease = pd.DataFrame(rng.normal(4, 1, (200, 8)), columns=cols)
    # a good target: disease moves to normal, normal barely moves
    good_dk = pd.DataFrame(rng.normal(0.2, 1, (200, 8)), columns=cols)
    good_nk = pd.DataFrame(rng.normal(0.0, 1, (200, 8)), columns=cols)
    # a bad target: disease stays put, normal gets wrecked
    bad_dk = pd.DataFrame(rng.normal(4, 1, (200, 8)), columns=cols)
    bad_nk = pd.DataFrame(rng.normal(5, 1, (200, 8)), columns=cols)
    good = distribution.normalization_score(disease, normal, good_dk, good_nk, metric="mmd")
    bad = distribution.normalization_score(disease, normal, bad_dk, bad_nk, metric="mmd")
    assert good["selective_score"] > bad["selective_score"]
    assert good["normalisation"] > 0
    # mahalanobis path also runs
    distribution.normalization_score(disease, normal, good_dk, good_nk, metric="mahalanobis")


@check("datacontract: rejects missing required canonical fields")
def _():
    from crcgem import datacontract as dc
    meta = pd.DataFrame({"grp": ["CRC", "CRC"], "pat": ["p1", "p2"]}, index=["s1", "s2"])
    # missing donor_id in column_map -> must raise
    try:
        dc._apply_column_map(meta, {"status": "grp"}, is_single_cell=False)
        raise AssertionError("accepted without donor_id")
    except dc.ContractError:
        pass
    # complete map works, provenance defaults to real
    out = dc._apply_column_map(meta, {"status": "grp", "donor_id": "pat"},
                               is_single_cell=False)
    assert list(out["provenance"].unique()) == ["real"]
    # single-cell without cell_type -> raise
    try:
        dc._apply_column_map(meta, {"status": "grp", "donor_id": "pat"},
                             is_single_cell=True)
        raise AssertionError("accepted single-cell without cell_type")
    except dc.ContractError:
        pass


@check("genemap: symbol/ensembl -> scaffold namespace with honest reporting")
def _():
    from crcgem import genemap as gm
    table = pd.DataFrame({
        "genes": ["ENSG1", "ENSG2", "ENSG3"],
        "geneEntrezID": ["10", "20", "30"],
        "geneSymbols": ["AAA", "BBB", "CCC"],
    })
    tmp = Path("/tmp/claude-1011/_gt.tsv")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(tmp, sep="\t", index=False)
    mat = pd.DataFrame(np.arange(8).reshape(4, 2),
                       index=["AAA", "BBB", "CCC", "GHOST"], columns=["s1", "s2"])
    out, rep = gm.harmonise(mat, "symbol", "human_gem", gene_table_path=tmp)
    assert set(out.index) == {"ENSG1", "ENSG2", "ENSG3"}       # mapped to ENSG
    assert rep.n_mapped == 3 and rep.n_unmapped == 1           # GHOST dropped
    assert "GHOST" in rep.unmapped_examples
    # same-namespace is identity (no table needed to remap)
    out2, rep2 = gm.harmonise(mat.rename(index={"GHOST": "AAA"}), "ensembl",
                              "human_gem", gene_table_path=tmp)
    assert rep2.from_type == rep2.to_type


@check("config: input 2x2 axes validated")
def _():
    c = config.InputConfig()
    try:
        c.validate(); raise AssertionError("accepted empty input")
    except config.ConfigError:
        pass
    config.InputConfig(data_source="single_cell", data_provenance="generated").validate()
    config.InputConfig(data_source="bulk", data_provenance="real").validate()


@check("distribution: mismatched scaffolds rejected")
def _():
    try:
        distribution.test_identity(pd.DataFrame({"r1": [1.0, 2.0]}),
                                   pd.DataFrame({"other": [1.0, 2.0]}), n_splits=2)
        raise AssertionError("accepted mismatched scaffolds")
    except ValueError as exc:
        assert "scaffold" in str(exc)


# ------------------------------------------------------- solver + real model
@check("solver: probe and preflight")
def _():
    found = solver_mod.available_solvers()
    assert found, "no solver found"
    solver_mod.select_solver(requires_milp=False)


@check("solver: MILP-incapable machine refuses tINIT rather than hanging")
def _():
    found = solver_mod.available_solvers()
    if any(i.milp_capable for i in found):
        return  # capable machine; nothing to refuse
    try:
        solver_mod.select_solver(requires_milp=True)
        raise AssertionError("should have refused")
    except solver_mod.SolverUnavailableError as exc:
        assert "MILP" in str(exc) or "CPLEX" in str(exc)


def _ecoli():
    import cobra

    m = cobra.io.read_sbml_model(ECOLI)
    m.solver = "glpk"
    return m


@check("reference_flux: pFBA reference + face degeneracy on E. coli core")
def _():
    if not Path(ECOLI).exists():
        raise FileNotFoundError(f"{ECOLI} absent — skipping model-based checks")
    ref = reference_flux.compute(_ecoli(), "pfba", diagnose=True)
    assert ref.growth > 0
    assert ref.n_degenerate <= 5, f"expected near-unique at core scale, got {ref.n_degenerate}"


@check("reference_flux: Chebyshev centre is deterministic")
def _():
    if not Path(ECOLI).exists():
        raise FileNotFoundError("E. coli core absent")
    a = reference_flux.compute(_ecoli(), "chebyshev_face", diagnose=False)
    b = reference_flux.compute(_ecoli(), "chebyshev_face", diagnose=False)
    np.testing.assert_allclose(a.fluxes.to_numpy(), b.fluxes.to_numpy(), atol=1e-6)


@check("moma: GPR-aware single-gene screen runs and finds essentials")
def _():
    if not Path(ECOLI).exists():
        raise FileNotFoundError("E. coli core absent")
    from crcgem import moma

    model = _ecoli()
    ref = reference_flux.compute(model, "pfba", diagnose=False)
    genes = [g.id for g in model.genes][:15]
    summary, fluxes = moma.single_gene_screen(model, ref, formulation="linear",
                                              genes=genes, progress=False)
    assert len(summary) == len(genes)
    assert not fluxes.empty and fluxes.shape[1] == len(model.reactions)
    called = moma.call_lethality(summary, 0.05)
    assert "lethal" in called and "growth_ratio" in called


@check("expression: reaction_scores + E-Flux bounds on a real model")
def _():
    if not Path(ECOLI).exists():
        raise FileNotFoundError("E. coli core absent")
    model = _ecoli()
    rng = np.random.default_rng(0)
    expr = pd.Series(rng.lognormal(2, 1, len(model.genes)),
                     index=[g.id for g in model.genes], name="ctx")
    scores = expression.reaction_scores(model, expr)
    assert len(scores) == len(model.reactions)
    assert scores.notna().sum() > 0
    bounds = expression.eflux_bounds(model, scores)
    assert bounds
    for lo, hi in bounds.values():
        assert lo <= hi


def main() -> int:
    print("crcgem verification\n")
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    if n_fail:
        print("\n--- failures ---")
        for name, ok, tb in _results:
            if not ok:
                print(f"\n### {name}\n{tb}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
