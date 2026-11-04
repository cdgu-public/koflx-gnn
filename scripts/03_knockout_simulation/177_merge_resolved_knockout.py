"""Build a MOMA root for 64d in which re-solved knockouts replace their rows (2026-11-04).

The other cell lines are symlinked from the base root unchanged. For each patched cell line the base directory is copied,
the summary row of every re-solved gene is overwritten with the re-solve (status, growth, growth_ratio, moma_objective,
n_disabled, feasibility_tol) and its flux row is added to flux_matrix.parquet in the base column order and dtype.
Usage: 177_merge_resolved_knockout.py BASE_ROOT RESOLVE_ROOT OUT_ROOT
"""
import io, shutil, sys
from datetime import datetime
from pathlib import Path
import pandas as pd

base, res, out = map(Path, sys.argv[1:4])
assert not out.exists(), f"{out} exists"
out.mkdir(parents=True)
patched_lines = {p.name for p in res.iterdir() if (p / "summary.tsv").exists()}
for d in sorted(base.iterdir()):
    if d.name not in patched_lines:
        (out / d.name).symlink_to(d.resolve())
        continue
    shutil.copytree(d, out / d.name)
    head = [l for l in open(d / "summary.tsv") if l.startswith("#")]
    sm = pd.read_csv(d / "summary.tsv", sep="\t", comment="#", index_col=0)
    rs = pd.read_csv(res / d.name / "summary.tsv", sep="\t", comment="#", index_col=0)
    genes = rs.index[rs.patched.astype(str) == "True"].tolist()
    fx = pd.read_parquet(d / "flux_matrix.parquet")
    fr = pd.read_parquet(res / d.name / "flux_matrix.parquet").loc[genes, fx.columns].astype(fx.dtypes.iloc[0])
    for g in genes:
        for k in ("status", "growth", "growth_ratio", "moma_objective", "n_disabled", "feasibility_tol"):
            sm.loc[g, k] = rs.loc[g, k]
        sm.loc[g, "patched"] = True
        sm.loc[g, "patch_status"] = "patched_resolve_20261104"
    fx = pd.concat([fx.drop(index=[g for g in genes if g in fx.index]), fr])
    assert set(fx.index) == set(sm.index[sm.status == "optimal"]), "flux rows != optimal genes"
    fx.to_parquet(out / d.name / "flux_matrix.parquet")
    buf = io.StringIO(); sm.to_csv(buf, sep="\t")
    note = (f"# 177 ({datetime.now():%F %T} KST): rows {', '.join(genes)} replaced from {res / d.name} "
            "(script 104 --genes, same shipped reference, FeasibilityTol of the re-solve in column feasibility_tol)\n")
    (out / d.name / "summary.tsv").write_text("".join(head) + note + buf.getvalue())
    print(d.name, "patched", genes, "flux rows", len(fx))
