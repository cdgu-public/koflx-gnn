"""Reactions protected in tINIT so that every context-specific model can perform the essential
metabolic tasks of Human-GEM (data/scaffold/metabolicTasks_Essential.txt, RAVEN task format).

For each task that should not fail: all exchanges closed; IN/OUT metabolites get artificial
input/output reactions with the task bounds (RAVEN defaults when empty: LB 0, UB 1000;
ALLMETSIN[comp] = every metabolite of that compartment as input); EQU rows are added as
reactions; CHANGED RXN bounds are applied. Feasibility is checked, then a minimal total flux
solution (pFBA with a zero objective) gives the reactions used. The union over passing tasks is
written as the --essential-reactions file of script 22. Tasks that fail on the scaffold are
listed and not protected.
Env: crcgem. Output: result/tinit_localT2_20261027/task_protection/
"""
import os, re, json
import numpy as np, pandas as pd, cobra
from cobra.flux_analysis.parsimonious import add_pfba

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
TASKS = f"{ROOT}/data/scaffold/metabolicTasks_Essential.txt"
OUT = f"{ROOT}/result/tinit_localT2_20261027/task_protection"
os.makedirs(OUT, exist_ok=True)
TOL = 1e-7


def parse_tasks(path):
    tasks, cur = [], None
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        p += [""] * (19 - len(p))
        if p[0].startswith("#") or p[1] == "ID":
            continue
        if p[1].strip():
            cur = dict(id=p[1].strip(), desc=p[2].strip(), fail=p[3].strip() not in ("", "0"),
                       ins=[], outs=[], equ=[], chg=[])
            tasks.append(cur)
        if cur is None:
            continue
        num = lambda x, d: float(x) if x.strip() else d
        if p[4].strip():
            for x in p[4].split(";"):
                cur["ins"].append((x.strip(), num(p[5], 0.0), num(p[6], 1000.0)))
        if p[7].strip():
            for x in p[7].split(";"):
                cur["outs"].append((x.strip(), num(p[8], 0.0), num(p[9], 1000.0)))
        if p[10].strip():
            rev = "<=>" in p[10]
            cur["equ"].append((p[10].strip(), num(p[11], -1000.0 if rev else 0.0), num(p[12], 1000.0)))
        if p[13].strip():
            cur["chg"].append((p[13].strip(), num(p[14], -1000.0), num(p[15], 1000.0)))
    return tasks


m = cobra.io.read_sbml_model(f"{ROOT}/data/scaffold/Human-GEM.xml")
m.solver = "gurobi"
m.solver.configuration.verbosity = 0
by_nc = {}
for mt in m.metabolites:
    by_nc[(mt.name, mt.compartment)] = mt
TOK = re.compile(r"^(.*)\[(\w+)\]$")


def met(token):
    g = TOK.match(token)
    return by_nc.get((g.group(1), g.group(2))) if g else None


def side(expr):
    d = {}
    for term in [t.strip() for t in expr.split(" + ") if t.strip()]:
        mm = re.match(r"^([0-9.eE+-]+)\s+(.*)$", term)
        coef, tok = (float(mm.group(1)), mm.group(2)) if mm else (1.0, term)
        x = met(tok)
        if x is None:
            raise KeyError(tok)
        d[x] = d.get(x, 0.0) + coef
    return d


orig = {r.id for r in m.reactions}
per_task = {}
tasks = parse_tasks(TASKS)
rows, protect = [], set()
for t in tasks:
    with m:
        for r in m.exchanges:
            r.bounds = (0.0, 0.0)
        try:
            for tok, lb, ub in t["ins"]:
                g = TOK.match(tok)
                mets = ([x for x in m.metabolites if x.compartment == g.group(2)]
                        if g and g.group(1) == "ALLMETSIN" else [met(tok)])
                for k, x in enumerate(mets):
                    if x is None:
                        raise KeyError(tok)
                    rx = cobra.Reaction(f"TIN{'ALL' if len(mets) > 1 else ''}_{x.id}"); rx.add_metabolites({x: 1}); rx.bounds = (lb, ub)
                    m.add_reactions([rx])
            for tok, lb, ub in t["outs"]:
                g = TOK.match(tok)
                mets = ([x for x in m.metabolites if x.compartment == g.group(2)]
                        if g and g.group(1) == "ALLMETSIN" else [met(tok)])
                for x in mets:
                    if x is None:
                        raise KeyError(tok)
                    rx = cobra.Reaction(f"TOUT{'ALL' if len(mets) > 1 else ''}_{x.id}"); rx.add_metabolites({x: -1}); rx.bounds = (lb, ub)
                    m.add_reactions([rx])
            for i, (eq, lb, ub) in enumerate(t["equ"]):
                arrow = "<=>" if "<=>" in eq else "=>"
                left, right = eq.split(arrow)
                st = {x: -c for x, c in side(left).items()}
                for x, c in side(right).items():
                    st[x] = st.get(x, 0.0) + c
                rx = cobra.Reaction(f"TEQU_{i}"); rx.add_metabolites(st); rx.bounds = (lb, ub)
                m.add_reactions([rx])
            for rid, lb, ub in t["chg"]:
                m.reactions.get_by_id(rid).bounds = (lb, ub)
        except KeyError as e:
            rows.append(dict(task=t["id"], desc=t["desc"], should_fail=t["fail"], status="unmapped:" + str(e)))
            continue
        m.objective = 0
        v = m.slim_optimize()
        feasible = v is not None and v == v
        used = []
        if feasible and not t["fail"]:
            add_pfba(m, fraction_of_optimum=0.0)
            sol = m.optimize()
            used = [r for r in orig if abs(sol.fluxes.get(r, 0.0)) > TOL]
            protect.update(used)
            per_task[t["desc"]] = sorted(used)
        rows.append(dict(task=t["id"], desc=t["desc"], should_fail=t["fail"], feasible=feasible,
                         passes=(feasible != t["fail"]), n_reactions_used=len(used)))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/task_results.tsv", sep="\t", index=False)
json.dump(per_task, open(f"{OUT}/task_reactions.json", "w"), indent=0)
with open(f"{OUT}/essential_reactions.txt", "w") as fh:
    fh.write("\n".join(sorted(protect)))
print(df.to_string(index=False))
print(f"tasks {len(df)}; pass {int(df.get('passes', pd.Series(dtype=bool)).fillna(False).sum())}; "
      f"protected reactions {len(protect)}")
