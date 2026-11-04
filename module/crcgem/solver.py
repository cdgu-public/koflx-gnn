"""Solver selection, capability preflight, and provenance recording.

Motivation (measured 2026-07-21, see result/pilot_solver_capability/):
  GLPK solved a 200-binary INIT-style MILP in 0.4 s but FAILED at 800 and at
  3,991 binaries (>600 s, time limit). CPLEX Community Edition refused the
  3,991-reaction model outright (Error 1016, size limit).
  => tINIT-scale MILP requires full CPLEX or Gurobi. This module refuses to
     start such a job on a solver that cannot finish it, instead of discovering
     it after hours of wall-clock.

Every run records which solver/version actually solved it, because MOMA and
sampling results are solver-dependent in ways this project must report.
"""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

#: Solvers able to handle tINIT-scale MILP (~4,000 binaries), best first.
MILP_CAPABLE = ("gurobi", "cplex")
#: Solvers acceptable for pure LP/QP work (FBA, pFBA, FVA, MOMA, sampling).
LP_ACCEPTABLE = ("gurobi", "cplex", "glpk")

#: Problem sizes at/above which an open-source solver is known to fail (measured).
GLPK_MILP_BINARY_CEILING = 200


class SolverUnavailableError(RuntimeError):
    """Raised when no solver present can complete the requested job class."""


@dataclass
class SolverInfo:
    """What actually solved the problem — goes into every result manifest."""

    name: str
    version: str = "unknown"
    milp_capable: bool = False
    edition_note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _probe(name: str) -> Optional[SolverInfo]:
    """Return SolverInfo if `name` is importable and usable, else None."""
    try:
        from cobra.util.solver import solvers
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("cobrapy is required") from exc

    if name not in solvers:
        return None

    version, note = "unknown", ""
    if name == "cplex":
        try:
            mod = importlib.import_module("cplex")
            version = getattr(mod, "__version__", "unknown")
            note = _cplex_edition_note()
        except Exception:
            return None
    elif name == "gurobi":
        try:
            mod = importlib.import_module("gurobipy")
            v = mod.gurobi.version()  # (major, minor, tech)
            version = ".".join(str(x) for x in v)
            note = _gurobi_licence_note(mod)
        except Exception:
            return None
    elif name == "glpk":
        try:
            mod = importlib.import_module("swiglpk")
            version = getattr(mod, "glp_version", lambda: "unknown")()
        except Exception:
            version = "unknown"

    return SolverInfo(
        name=name,
        version=str(version),
        milp_capable=(name in MILP_CAPABLE and "Community" not in note),
        edition_note=note,
    )


def _cplex_edition_note() -> str:
    """Detect CPLEX Community Edition, which silently caps problem size."""
    try:
        import cplex

        p = cplex.Cplex()
        p.set_log_stream(None)
        p.set_results_stream(None)
        p.set_warning_stream(None)
        p.set_error_stream(None)
        # CE caps at 1000 vars / 1000 constraints.
        p.variables.add(lb=[0.0] * 1500)
        p.solve()
        p.end()
        return "full"
    except Exception as exc:
        if "1016" in str(exc) or "Community" in str(exc):
            return "Community Edition (size-limited; NOT usable for tINIT)"
        return f"unverified ({type(exc).__name__})"


def _gurobi_licence_note(mod) -> str:
    """Detect Gurobi size-limited (pip/trial) licence."""
    try:
        env = mod.Env(params={"OutputFlag": 0})
        m = mod.Model(env=env)
        m.addVars(3000, vtype=mod.GRB.BINARY)
        m.update()
        m.optimize()
        m.dispose()
        env.dispose()
        return "full"
    except Exception as exc:
        if "size-limited" in str(exc).lower() or "license" in str(exc).lower():
            return "size-limited licence (NOT usable for tINIT)"
        return f"unverified ({type(exc).__name__})"


def available_solvers() -> list[SolverInfo]:
    """All usable solvers on this machine, MILP-capable first."""
    found = [i for i in (_probe(n) for n in ("gurobi", "cplex", "glpk")) if i]
    return sorted(found, key=lambda i: (not i.milp_capable, i.name))


def select_solver(requires_milp: bool, prefer: Optional[str] = None) -> SolverInfo:
    """Pick a solver, refusing jobs this machine cannot finish.

    Args:
        requires_milp: True for tINIT/INIT-family extraction.
        prefer: force a specific solver name; raises if unusable.
    """
    found = available_solvers()
    if not found:
        raise SolverUnavailableError("no COBRApy-compatible solver found")

    if prefer:
        match = [i for i in found if i.name == prefer]
        if not match:
            raise SolverUnavailableError(
                f"requested solver {prefer!r} unavailable; found "
                f"{[i.name for i in found]}"
            )
        info = match[0]
        if requires_milp and not info.milp_capable:
            raise SolverUnavailableError(
                f"{prefer} cannot handle tINIT-scale MILP ({info.edition_note})"
            )
        return info

    pool = [i for i in found if i.milp_capable] if requires_milp else found
    if not pool:
        raise SolverUnavailableError(
            "tINIT-scale MILP needs full CPLEX or Gurobi. Present: "
            + "; ".join(f"{i.name} {i.version} [{i.edition_note}]" for i in found)
            + ". MEASURED: GLPK fails above ~200 binaries; this job needs ~4,000."
        )
    return pool[0]


def apply(model, info: SolverInfo, *, threads: Optional[int] = None,
          timeout: Optional[int] = None, verbose: bool = False):
    """Attach solver + configuration to a cobra model. Returns the model."""
    model.solver = info.name
    cfg = model.solver.configuration
    for attr, val in (("threads", threads), ("timeout", timeout),
                      ("verbosity", 3 if verbose else 0)):
        if val is None:
            continue
        try:
            setattr(cfg, attr, val)
        except Exception as exc:
            logger.debug("solver config %s=%s not applied: %s", attr, val, exc)
    return model


def preflight(requires_milp: bool, prefer: Optional[str] = None) -> SolverInfo:
    """Log the solver landscape and return the selected one. Call before long jobs."""
    t0 = time.time()
    found = available_solvers()
    logger.info("solver probe (%.1fs): %s", time.time() - t0,
                ", ".join(f"{i.name} {i.version} milp={i.milp_capable} "
                          f"[{i.edition_note}]" for i in found) or "none")
    info = select_solver(requires_milp=requires_milp, prefer=prefer)
    logger.info("selected solver: %s %s", info.name, info.version)
    return info
