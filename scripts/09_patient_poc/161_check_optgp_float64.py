"""Compare float64 OptGP samples (110 --sample-dtype float64) with the earlier float32 samples of the same seed (2026-11-03, POC normals).
Reports: (1) whether float32(new) equals the stored float32 samples exactly (same draws, i.e. 110 is deterministic per seed);
(2) max |S v| and max bound violation per sample for the float64 samples, the float32 cast of them, and the old file.
Usage: python 161_check_optgp_float64.py <models_root> <line> <old_samples.parquet> <new_samples.parquet>
"""
import sys
import numpy as np, pandas as pd, cobra
from cobra.util.array import create_stoichiometric_matrix

models_root, line, f_old, f_new = sys.argv[1:5]
m = cobra.io.read_sbml_model(f"{models_root}/{line}/{line}.repaired.xml")
rids = [r.id for r in m.reactions]
old, new = pd.read_parquet(f_old), pd.read_parquet(f_new)
print(f"{line}: old dtype {old.dtypes.iloc[0]} shape {old.shape}; new dtype {new.dtypes.iloc[0]} shape {new.shape}")
same_cols = set(old.columns) == set(new.columns)
print(f"columns equal: {same_cols}; model reactions covered: {set(rids) <= set(new.columns)}")
eq = np.array_equal(new[old.columns].astype("float32").to_numpy(), old.to_numpy())
print(f"float32(new) == old exactly: {eq}; max |float32(new) - old| {np.abs(new[old.columns].astype('float32').to_numpy() - old.to_numpy()).max():.3g}")
S = create_stoichiometric_matrix(m, array_type="lil").tocsr()
for name, X in (("float64 new", new[rids].to_numpy(np.float64)), ("float32 cast of new", new[rids].astype("float32").to_numpy(np.float64)),
                ("old stored", old[rids].to_numpy(np.float64))):
    sv = np.abs(S @ X.T).max(0)
    print(f"  {name}: max|Sv| per sample median {np.median(sv):.3g} max {sv.max():.3g}; max |v| {np.abs(X).max():.4g}")
