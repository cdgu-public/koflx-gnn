#!/usr/bin/env python
"""Report solver capability on THIS machine and whether tINIT can run here.

Run this first on any new machine (e.g. the CPLEX server) before spending
wall-clock. Optionally re-measures the MILP ceiling empirically.

    python script_python/00_solver_preflight.py
    python script_python/00_solver_preflight.py --model <sbml> --measure-milp
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "module"))

from crcgem import solver as solver_mod  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("preflight")


def measure_milp_ceiling(model_path: str, sizes, timeout: int, solver_name=None):
    """Build an INIT-style MILP at increasing sizes and time each solve."""
    import cobra
    import numpy as np

    results = []
    for n in sizes:
        model = cobra.io.read_sbml_model(model_path)
        rxns = model.reactions if n is None else model.reactions[:n]
        info = solver_mod.select_solver(requires_milp=False, prefer=solver_name)
        solver_mod.apply(model, info, timeout=timeout)

        rng = np.random.default_rng(0)
        w = rng.choice([1.0, -1.0], size=len(rxns), p=[0.7, 0.3])
        cons, obj = [], []
        for r, wi in zip(rxns, w):
            y = model.problem.Variable(f"y_{r.id}", type="binary")
            flux = r.forward_variable + r.reverse_variable
            cons.append(model.problem.Constraint(flux - 1e-3 * y, lb=0, name=f"lo_{r.id}"))
            cons.append(model.problem.Constraint(flux - 1000.0 * y, ub=0, name=f"hi_{r.id}"))
            obj.append(wi * y)
        model.add_cons_vars(cons)
        model.objective = model.problem.Objective(sum(obj), direction="max")

        t0 = time.time()
        try:
            val = model.slim_optimize(error_value=None)
            rec = {"solver": info.name, "binaries": len(rxns),
                   "status": model.solver.status, "objective": val,
                   "seconds": round(time.time() - t0, 1)}
        except Exception as exc:
            rec = {"solver": info.name, "binaries": len(rxns), "status": "FAILED",
                   "error": f"{type(exc).__name__}: {exc}",
                   "seconds": round(time.time() - t0, 1)}
        log.info("%s", rec)
        results.append(rec)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="SBML model for the empirical MILP test")
    ap.add_argument("--measure-milp", action="store_true")
    ap.add_argument("--sizes", default="200,800,0",
                    help="comma-separated binary counts; 0 = full model")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--solver")
    ap.add_argument("--out", default="result/solver_preflight.json")
    args = ap.parse_args()

    found = solver_mod.available_solvers()
    report = {"available": [i.as_dict() for i in found]}
    print("\n=== solvers on this machine ===")
    for i in found:
        print(f"  {i.name:<8} {i.version:<12} milp_capable={i.milp_capable}  {i.edition_note}")
    if not found:
        print("  (none)")

    try:
        sel = solver_mod.select_solver(requires_milp=True, prefer=args.solver)
        report["tinit_ready"] = True
        report["tinit_solver"] = sel.as_dict()
        print(f"\ntINIT CAN run here using: {sel.name} {sel.version}")
    except solver_mod.SolverUnavailableError as exc:
        report["tinit_ready"] = False
        report["tinit_blocker"] = str(exc)
        print(f"\ntINIT CANNOT run here:\n  {exc}")

    if args.measure_milp:
        if not args.model:
            print("\n--measure-milp requires --model", file=sys.stderr)
            return 2
        sizes = [None if s.strip() == "0" else int(s) for s in args.sizes.split(",")]
        report["milp_measurements"] = measure_milp_ceiling(
            args.model, sizes, args.timeout, args.solver)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
