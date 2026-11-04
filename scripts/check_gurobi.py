"""Check that gurobipy is importable, report its version and license type, and solve a tiny LP and a tiny QP.

The context-specific models have up to about 13,000 reactions (Human-GEM scaffold 12,931). The size-limited license that
ships with the pip/conda gurobipy package allows at most 2,000 variables and 2,000 linear constraints, so the pipeline
needs a full license (for academic users, the free Gurobi academic license).

Usage: python scripts/check_gurobi.py
"""
import sys

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as e:
    sys.exit(f"FAIL: gurobipy not importable ({e}). Install it into the conda environment (see README).")

print("gurobipy version:", ".".join(map(str, gp.gurobi.version())))

# 1. tiny LP: max x + y s.t. x + 2y <= 4, 3x + y <= 6, x, y >= 0  (optimum 2.8 at x = 1.6, y = 1.2)
try:
    m = gp.Model("lp")
    m.Params.OutputFlag = 0
    x = m.addVar(lb=0, name="x"); y = m.addVar(lb=0, name="y")
    m.addConstr(x + 2 * y <= 4); m.addConstr(3 * x + y <= 6)
    m.setObjective(x + y, GRB.MAXIMIZE)
    m.optimize()
    assert m.Status == GRB.OPTIMAL and abs(m.ObjVal - 2.8) < 1e-6, (m.Status, m.ObjVal)
    print(f"LP : optimal, objective {m.ObjVal:.6f} (expected 2.8)")
except gp.GurobiError as e:
    sys.exit(f"FAIL: LP solve raised GurobiError {e.errno}: {e}. No usable license was found.")

# 2. tiny QP (MOMA form): min (x - 1)^2 + (y - 2)^2 s.t. x + y = 1  (optimum 2.0 at x = 0, y = 1)
m = gp.Model("qp")
m.Params.OutputFlag = 0
x = m.addVar(lb=-GRB.INFINITY, name="x"); y = m.addVar(lb=-GRB.INFINITY, name="y")
m.addConstr(x + y == 1)
m.setObjective((x - 1) * (x - 1) + (y - 2) * (y - 2), GRB.MINIMIZE)
m.optimize()
assert m.Status == GRB.OPTIMAL and abs(m.ObjVal - 2.0) < 1e-6, (m.Status, m.ObjVal)
print(f"QP : optimal, objective {m.ObjVal:.6f} (expected 2.0)")

# 3. license size check: a model above the size-limited license bound (2,000 variables / constraints)
n = 2500
m = gp.Model("size")
m.Params.OutputFlag = 0
v = m.addVars(n, lb=0, ub=1)
m.addConstrs(v[i] <= 1 for i in range(n))
m.setObjective(v.sum(), GRB.MAXIMIZE)
try:
    m.optimize()
    print(f"License: FULL (a {n}-variable model solved). Genome-scale models (about 13,000 reactions) can be solved.")
except gp.GurobiError as e:
    if e.errno == 10010:  # GRB_ERROR_SIZE_LIMIT_EXCEEDED
        print("License: SIZE-LIMITED (restricted pip/conda license, at most 2,000 variables and constraints).\n"
              "The genome-scale models in this pipeline (about 13,000 reactions) need a full license, for example the free\n"
              "Gurobi academic license: https://www.gurobi.com/academia/academic-program-and-licenses/")
        sys.exit(1)
    raise
