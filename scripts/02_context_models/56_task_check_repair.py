"""Essential-task check and repair of the tINIT context models (script 22 output).

Each context model is tested on the 57 essential tasks of Human-GEM with the same task code as
script 54 (RAVEN semantics; exchanges closed, task inputs/outputs added). For every task that
fails, the reactions that task uses in the scaffold (result/tinit_localT2_20261027/task_protection/
task_reactions.json, pFBA solution of script 54) are added back from the scaffold, and the task is
re-tested. Writes <line>.repaired.xml next to the tINIT model plus task_check.tsv.
Usage: python 56_task_check_repair.py <ModelID>   Env: crcgem.
"""
import os, re, sys, json
import numpy as np, pandas as pd, cobra
from cobra.flux_analysis.parsimonious import add_pfba

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
TASKS = f"{ROOT}/data/scaffold/metabolicTasks_Essential.txt"
TP = f"{ROOT}/result/tinit_localT2_20261027/task_protection"
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




def run_tasks(m, tasks):
    orig = {r.id for r in m.reactions}
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



    rows = []
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
            rows.append(dict(task=t["id"], desc=t["desc"], should_fail=t["fail"], feasible=feasible,
                             passes=(feasible != t["fail"]), n_reactions_used=len(used)))

    return rows


line = sys.argv[1]
# optional argv[2]: models root (out-of-bag lines, script 76; default the CRC models)
D = f"{sys.argv[2] if len(sys.argv) > 2 else ROOT + '/result/tinit_localT2_20261027/models'}/{line}"
scaf = cobra.io.read_sbml_model(f"{ROOT}/data/scaffold/Human-GEM.xml")
ctx = cobra.io.read_sbml_model(f"{D}/{line}.xml")
for mm in (scaf, ctx):
    mm.solver = "gurobi"; mm.solver.configuration.verbosity = 0
tasks = parse_tasks(TASKS)
per_task = json.load(open(f"{TP}/task_reactions.json"))
r0 = pd.DataFrame(run_tasks(ctx, tasks))
failed = r0[~r0.passes.fillna(False).astype(bool)].desc.tolist()
added = set()
for d in failed:
    need = [x for x in per_task.get(d, []) if x not in ctx.reactions]
    ctx.add_reactions([scaf.reactions.get_by_id(x).copy() for x in need])
    added.update(need)
r1 = pd.DataFrame(run_tasks(ctx, tasks)) if failed else r0
out = r0[["task", "desc", "passes"]].rename(columns={"passes": "passes_tinit"})
out["passes_repaired"] = r1["passes"].values
out.to_csv(f"{D}/task_check.tsv", sep="\t", index=False)
cobra.io.write_sbml_model(ctx, f"{D}/{line}.repaired.xml")
json.dump({"line": line, "n_failed_tinit": len(failed), "failed": failed, "n_added_reactions": len(added),
           "n_failed_after_repair": int((~r1.passes.fillna(False).astype(bool)).sum()),
           "n_reactions_repaired": len(ctx.reactions)}, open(f"{D}/task_repair.json", "w"), indent=1)
print(line, "failed", len(failed), "added", len(added), "after", int((~r1.passes.fillna(False).astype(bool)).sum()))
