"""Overall and KEGG-map-wise fluxome distance between MOMA knockout fluxes and their reference flux
(design: result/external_plan_20261103/design_kegg_distance_20261103.md). Auxiliary relative readout only; no threshold.

Input, one directory per model under --moma-root (format of script 58 / 104):
  summary.tsv          '#' header lines; index gene (cell lines) or columns reference_id + gene (per-sample references);
                       columns status, growth_ratio, moma_objective (float64)
  flux_matrix.parquet  rows = optimal knockouts (index gene, or MultiIndex (reference_id, gene)), columns = model reactions, float32 or float64
  reference.parquet    index = model reactions; one column 'reference' (pFBA reference) or one column per reference_id
                       (e.g. OptGP samples used as MOMA references, script 142)
Mapping: data/kegg_<date>/humangem_reaction_kegg_map.tsv (script 145). Two routes, both reported (column route):
  kegg_reaction        maps from the kegg.reaction annotation only
  kegg_reaction_or_ec  kegg.reaction first, complete EC as fallback (145 rule)
  Reactions with no map under a route are grouped by their Human-GEM subsystem (group_kind humangem_subsystem_no_kegg_map);
  model reactions absent from the Human-GEM scaffold form one group (group_kind not_in_humangem_scaffold).
Definitions (per model, reference r, knockout g; tol = 1e-6, the solver tolerance scale):
  storage rounding: the reference is cast to the dtype of the stored knockout fluxes (float32 or float64) and back before differencing,
                    so reactions the knockout leaves unchanged difference to exactly 0
  thresholded flux: v_t = v if |v| > tol else 0 (applied to reference and knockout)
  active          : |v_t| > 0, i.e. |v| > tol;  changed: |v_KO_t - v_ref_t| > tol
  d_all           = sqrt(moma_objective) / ||v_ref||_2           (float64 objective; addendum section 5)
  d_all_parquet   = ||v_KO - v_ref||_2 / ||v_ref||_2 from the stored fluxes (reproduction of d_all; raw, no threshold)
  d_all_t         = same on thresholded fluxes (the footing of the map-wise columns)
  per group G (reactions of G present in the model, thresholded fluxes):
  d_rel_ref       = ||D_G|| / ||v_ref,G||                  NA if n_active_ref == 0
  d_sym           = ||D_G|| / (||v_ref,G|| + ||v_KO,G||)   in [0, 1]; NA if n_active_either == 0
  rms_delta       = ||D_G|| / sqrt(n_in_model)            flux units, per-reaction size correction
  share_sq_delta  = ||D_G||^2 / ||D_all||^2               NA if ||D_all|| == 0; maps overlap, shares do not sum to 1
Non-optimal knockouts: kept in overall_distance.tsv with status and NA distances; absent from pathway_distance.tsv.
No existing output is overwritten (new file gets __YYYYMMDD_HHMMSS). Usage:
  python 141_fluxome_distance.py --moma-root result/moma_20261030_tolpatch --set cell_lines_59 [--lines ACH-000007 ACH-000009]
  Env: crcgem. Output: result/fluxome_distance_<date>/<set>/<model>/{overall_distance.tsv, pathway_distance.tsv.gz, group_size.tsv}
"""
import argparse, gzip, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/data/workspace/cdgu/Winery/seeding/CRC_GEM")
ap = argparse.ArgumentParser()
ap.add_argument("--moma-root", required=True)
ap.add_argument("--set", required=True, help="name of the model set, used in the output path")
ap.add_argument("--lines", nargs="*", help="model directories to process (default: all with flux_matrix.parquet)")
ap.add_argument("--mapping", default="data/kegg_20261103/humangem_reaction_kegg_map.tsv")
ap.add_argument("--tol", type=float, default=1e-6)
ap.add_argument("--out-root", default=f"result/fluxome_distance_{datetime.now():%Y%m%d}")
args = ap.parse_args()
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
TOL = args.tol
mroot = ROOT / args.moma_root
outroot = ROOT / args.out_root / args.set


def safe_path(p: Path) -> Path:
    return p if not p.exists() else p.with_name(p.name.replace(".tsv", f"__{STAMP}.tsv"))


def write(df, p, header):
    p = safe_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with (gzip.open(p, "wt") if p.suffix == ".gz" else open(p, "w")) as fh:
        fh.write("\n".join(header) + "\n")
        df.to_csv(fh, sep="\t", index=False, float_format="%.6g")
    return p


mp = pd.read_csv(ROOT / args.mapping, sep="\t", comment="#", dtype=str, keep_default_na=False)
map_hdr = [l.rstrip() for l in open(ROOT / args.mapping) if l.startswith("#")][:2]
sub_of = mp.drop_duplicates("reaction").set_index("reaction").subsystem
n_hg_cache = {}


def groups_for(route):
    """reaction -> list of (group_kind, group_id, group_name) over all Human-GEM reactions."""
    m = mp[mp.map_id != ""]
    if route == "kegg_reaction":
        m = m[m.source == "kegg_reaction"]
    g = {}
    for rx, mid, name in zip(m.reaction, m.map_id, m.map_name):
        g.setdefault(rx, []).append(("kegg_map", mid, name))
    for rx in sub_of.index:
        if rx not in g:
            s = sub_of[rx] or "(no subsystem)"
            g[rx] = [("humangem_subsystem_no_kegg_map", s, s)]
    return g


ROUTES = {r: groups_for(r) for r in ("kegg_reaction", "kegg_reaction_or_ec")}
for route, g in ROUTES.items():
    cnt = {}
    for lst in g.values():
        for k in lst:
            cnt[k] = cnt.get(k, 0) + 1
    n_hg_cache[route] = cnt

lines = args.lines or sorted(p.name for p in mroot.iterdir() if (p / "flux_matrix.parquet").exists())
print(f"{len(lines)} models", flush=True)
for line in lines:
    d = mroot / line
    summ = pd.read_csv(d / "summary.tsv", sep="\t", comment="#")
    flux = pd.read_parquet(d / "flux_matrix.parquet")
    FDT = flux.dtypes.iloc[0]; assert (flux.dtypes == FDT).all() and FDT in (np.float32, np.float64), flux.dtypes.unique()
    refs = pd.read_parquet(d / "reference.parquet")
    per_sample = isinstance(flux.index, pd.MultiIndex)
    if not per_sample:
        assert list(refs.columns) == ["reference"], refs.columns
        summ["reference_id"] = "reference"
        flux.index = pd.MultiIndex.from_arrays([["reference"] * len(flux), flux.index], names=["reference_id", "gene"])
    assert {"reference_id", "gene", "status", "moma_objective"} <= set(summ.columns), summ.columns
    rxns = list(flux.columns)
    assert set(rxns) <= set(refs.index), "flux columns not covered by the reference"
    n_opt = int((summ.status == "optimal").sum())
    assert len(flux) == n_opt, f"{line}: {len(flux)} flux rows vs {n_opt} optimal knockouts in summary"
    overall_rows, path_frames, size_frames = [], [], []
    # group membership matrices per route, restricted to model reactions
    mats = {}
    for route, g in ROUTES.items():
        keys = sorted({k for rx in rxns for k in g.get(rx, [("not_in_humangem_scaffold", "not_in_humangem_scaffold", "")])})
        kidx = {k: i for i, k in enumerate(keys)}
        M = np.zeros((len(rxns), len(keys)))
        for i, rx in enumerate(rxns):
            for k in g.get(rx, [("not_in_humangem_scaffold", "not_in_humangem_scaffold", "")]):
                M[i, kidx[k]] = 1.0
        assert (M.sum(1) >= 1).all(), "a model reaction belongs to no group"
        mats[route] = (keys, M)
    for ref_id, s_ref in summ.groupby("reference_id", sort=False):
        r64 = refs.loc[rxns, ref_id].to_numpy(np.float64)
        r = r64.astype(FDT).astype(np.float64)                  # storage rounding to the flux dtype, see docstring
        rt = np.where(np.abs(r) > TOL, r, 0.0)
        nr = np.linalg.norm(r64)
        sub = flux.xs(ref_id, level="reference_id") if ref_id in flux.index.get_level_values(0) else flux.iloc[:0]
        V = sub.to_numpy(np.float64)
        Vt = np.where(np.abs(V) > TOL, V, 0.0)
        D, Dt = V - r, Vt - rt
        ss_all_t = (Dt ** 2).sum(1)
        obj = s_ref.set_index("gene").moma_objective.reindex(sub.index).to_numpy(np.float64)
        # QP objectives slightly below 0 (observed down to -2.5e-14) are solver rounding of 0: set to 0 when >= -1e-9, else NA
        obj_c = np.where(obj >= 0, obj, np.where(obj >= -1e-9, 0.0, np.nan))
        ov = pd.DataFrame({"gene": sub.index, "d_all": np.sqrt(obj_c) / nr, "objective_negative_set_to_0": (obj < 0) & (obj >= -1e-9),
                           "d_all_parquet": np.linalg.norm(D, axis=1) / nr,
                           "d_all_t": np.sqrt(ss_all_t) / np.linalg.norm(rt),
                           "repro_rel_diff": np.abs((D ** 2).sum(1) - obj) / np.where(obj > 0, obj, np.nan),
                           "n_active_ref": int((rt != 0).sum()), "n_active_ko": (Vt != 0).sum(1),
                           "n_changed": (np.abs(Dt) > TOL).sum(1)})
        ov = s_ref.merge(ov, on="gene", how="left")
        ov.insert(0, "reference_id", ov.pop("reference_id"))
        ov.insert(0, "model", line)
        ov["norm_ref"] = nr
        overall_rows.append(ov)
        for route, (keys, M) in mats.items():
            act_r = (rt != 0).astype(float) @ M
            n_in = M.sum(0)
            ar, ak = (rt != 0), (Vt != 0)
            n_ak = ak.astype(float) @ M
            n_either = (ak | ar).astype(float) @ M
            n_chg = (np.abs(Dt) > TOL).astype(float) @ M
            ss_d, ss_r, ss_k = (Dt ** 2) @ M, (rt ** 2) @ M, (Vt ** 2) @ M
            nd, nrg, nk = np.sqrt(ss_d), np.sqrt(ss_r), np.sqrt(ss_k)
            with np.errstate(invalid="ignore", divide="ignore"):
                d_rel = np.where(act_r[None, :] > 0, nd / nrg, np.nan)
                d_sym = np.where(n_either > 0, nd / (nrg + nk), np.nan)
                share = np.where(ss_all_t[:, None] > 0, ss_d / ss_all_t[:, None], np.nan)
            G, K = V.shape[0], len(keys)
            pf = pd.DataFrame({"model": line, "reference_id": ref_id, "gene": np.repeat(sub.index.to_numpy(), K),
                               "route": route,
                               "group_kind": np.tile([k[0] for k in keys], G), "group_id": np.tile([k[1] for k in keys], G),
                               "group_name": np.tile([k[2] for k in keys], G),
                               "n_in_humangem": np.tile([n_hg_cache[route].get(k, 0) for k in keys], G),
                               "n_in_model": np.tile(n_in, G).astype(int), "n_active_ref": np.tile(act_r, G).astype(int),
                               "n_active_ko": n_ak.ravel().astype(int), "n_active_either": n_either.ravel().astype(int),
                               "n_changed": n_chg.ravel().astype(int), "norm_delta": nd.ravel(), "norm_ref": np.tile(nrg, G),
                               "norm_ko": nk.ravel(), "d_rel_ref": d_rel.ravel(), "d_sym": d_sym.ravel(),
                               "rms_delta": (nd / np.sqrt(n_in)).ravel(), "share_sq_delta": share.ravel()})
            path_frames.append(pf)
            size_frames.append(pd.DataFrame({"model": line, "reference_id": ref_id, "route": route,
                                             "group_kind": [k[0] for k in keys], "group_id": [k[1] for k in keys],
                                             "group_name": [k[2] for k in keys],
                                             "n_in_humangem": [n_hg_cache[route].get(k, 0) for k in keys],
                                             "n_in_model": n_in.astype(int), "n_active_ref": act_r.astype(int),
                                             "norm_ref": nrg}))
            absent = [k for k in n_hg_cache[route] if k not in set(keys)]
            size_frames.append(pd.DataFrame({"model": line, "reference_id": ref_id, "route": route,
                                             "group_kind": [k[0] for k in absent], "group_id": [k[1] for k in absent],
                                             "group_name": [k[2] for k in absent],
                                             "n_in_humangem": [n_hg_cache[route][k] for k in absent],
                                             "n_in_model": 0, "n_active_ref": 0, "norm_ref": 0.0}))
            # invariant: per-reaction sums over a partition-free check -> every reaction counted >= once, total ss recoverable
            assert np.allclose(((Dt ** 2) * (1.0 / M.sum(1))[None, :]) @ M @ np.ones(K), ss_all_t, rtol=1e-9, atol=1e-30)
    ov = pd.concat(overall_rows)
    pw = pd.concat(path_frames)
    sz = pd.concat(size_frames)
    assert len(ov) == len(summ), "overall rows must equal summary rows"
    hdr = [f"# {line}: fluxome distance (script_python/141_fluxome_distance.py, {datetime.now().isoformat(timespec='seconds')}); "
           f"design result/external_plan_20261103/design_kegg_distance_20261103.md",
           f"# source: {d} (summary.tsv, flux_matrix.parquet {FDT}, reference.parquet); mapping {ROOT / args.mapping}",
           f"# mapping header: {' | '.join(h.lstrip('# ') for h in map_hdr)}",
           f"# tol {TOL} (active |v| > tol; changed |dv| > tol; thresholded flux v_t); reference cast to {FDT} before differencing; "
           "excluded maps map01100-map01299; routes kegg_reaction and kegg_reaction_or_ec; auxiliary readout, no threshold",
           f"# python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}",
           "# model reactions in >= 1 KEGG map (excl. global/overview): "
           + "; ".join(f"{rt_} {int((M_[:, [k[0] == 'kegg_map' for k in ks]].sum(1) > 0).sum())}/{len(rxns)}"
                       for rt_, (ks, M_) in mats.items())]
    p1 = write(ov, outroot / line / "overall_distance.tsv",
               hdr + [f"# n rows {len(ov)} (all screened knockouts; non-optimal kept with NA distances); "
                      "d_all = sqrt(moma_objective)/||v_ref||; d_all_parquet and repro_rel_diff reproduce it from stored fluxes"])
    p2 = write(pw, outroot / line / "pathway_distance.tsv.gz",
               hdr + [f"# n rows {len(pw)} = optimal knockouts {n_opt} x groups present in the model, per route; "
                      "groups with n_in_model 0 are listed only in group_size.tsv"])
    p3 = write(sz, outroot / line / "group_size.tsv", hdr + [f"# n rows {len(sz)}"])
    print(line, len(ov), len(pw), p1.parent, flush=True)
