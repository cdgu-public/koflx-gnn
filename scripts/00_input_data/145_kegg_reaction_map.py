"""Human-GEM reaction -> KEGG pathway map assignment (design_kegg_distance_20261103.md section 2).

Fetches from KEGG REST (rest.kegg.jp) and saves the raw files only under data/kegg_<date>/ with SOURCE.txt
(KEGG raw files are not redistributed; this script regenerates them):
  info_kegg.txt, info_pathway.txt, list_pathway.txt, list_reaction.txt, list_enzyme.txt,
  link_pathway_reaction.txt, link_pathway_enzyme.txt
Rule (precedence, applied per Human-GEM reaction):
  1. kegg.reaction annotation -> link/pathway/reaction -> maps (source kegg_reaction).
  2. only if step 1 yields no map outside the excluded range: ec-code annotation (complete 4-level EC only)
     -> link/pathway/enzyme -> maps (source ec).
  Excluded maps: KEGG global and overview maps, map number 01100-01299 (011xx, 012xx).
  A reaction linked to several maps is a member of every one of them (KEGG maps overlap by design).
  Reactions with no map keep a row with map_id empty and their reason (no_annotation, id_not_in_kegg,
  ec_incomplete_only, annotated_no_map, only_global_overview_map); 141 groups them by Human-GEM subsystem.
Outputs (date = run date, YYYYMMDD):
  data/kegg_<date>/humangem_reaction_kegg_map.tsv   long: one row per (reaction, map), unmapped reactions one row each
  data/kegg_<date>/kegg_map_size.tsv                per map: n Human-GEM reactions (all maps incl. excluded, column excluded)
  result/external_plan_20261103/kegg_map_coverage_<date>.tsv   coverage counts (no KEGG content beyond counts)
No existing output is overwritten: an existing file gets the new file written with suffix __YYYYMMDD_HHMMSS.
Usage: python 145_kegg_reaction_map.py [--date 20261103] [--refetch]   Env: crcgem.
"""
import argparse, re, sys, time, urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path("/data/workspace/cdgu/Winery/seeding/CRC_GEM")
GEM = ROOT / "data/scaffold/Human-GEM.xml"
BASE = "https://rest.kegg.jp"
FILES = {"info_kegg.txt": "info/kegg", "info_pathway.txt": "info/pathway", "list_pathway.txt": "list/pathway",
         "list_reaction.txt": "list/reaction", "list_enzyme.txt": "list/enzyme",
         "link_pathway_reaction.txt": "link/pathway/reaction", "link_pathway_enzyme.txt": "link/pathway/enzyme"}
EXCL_LO, EXCL_HI = 1100, 1299   # global and overview maps (011xx, 012xx)

ap = argparse.ArgumentParser()
ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
ap.add_argument("--refetch", action="store_true", help="fetch again even if raw files exist (new files get a suffix)")
args = ap.parse_args()
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
KDIR = ROOT / f"data/kegg_{args.date}"
KDIR.mkdir(parents=True, exist_ok=True)


def safe_path(p: Path) -> Path:
    return p if not p.exists() else p.with_name(f"{p.stem}__{STAMP}{p.suffix}")


# ---- 1. fetch ----------------------------------------------------------------------------------------
fetched = {}
for fn, q in FILES.items():
    p = KDIR / fn
    if p.exists() and not args.refetch:
        fetched[fn] = (p, "existing file reused")
        continue
    url = f"{BASE}/{q}"
    for attempt in range(3):
        try:
            txt = urllib.request.urlopen(url, timeout=120).read().decode()
            break
        except Exception as e:  # noqa: BLE001
            print(f"retry {url}: {e}", flush=True); time.sleep(5)
    else:
        sys.exit(f"fetch failed: {url}")
    p = safe_path(p)
    p.write_text(txt)
    fetched[fn] = (p, f"{url} fetched {datetime.now().isoformat(timespec='seconds')}")
    time.sleep(1)   # KEGG REST courtesy pause

info = fetched["info_kegg.txt"][0].read_text()
rel = re.findall(r"Release\s+\S+", info)
release = rel[0] if rel else ("no 'Release' string in info/kegg; per-database last-update lines: "
                              + "; ".join(l.strip() for l in info.splitlines()[1:12] if l.strip()))
src = KDIR / "SOURCE.txt"
src_lines = [f"# KEGG REST raw files for Human-GEM -> KEGG map assignment; written by script_python/145_kegg_reaction_map.py",
             f"# NOT for redistribution (KEGG license); regenerate with the script.",
             f"date: {datetime.now().isoformat(timespec='seconds')}", f"KEGG release: {release}"]
src_lines += [f"{fn}\t{p.name}\t{how}" for fn, (p, how) in fetched.items()]
if any(how != "existing file reused" for _, how in fetched.values()):   # only when something was fetched
    safe_path(src).write_text("\n".join(src_lines) + "\n")   # an existing SOURCE.txt is kept; the new one gets a suffix


def read_link(fn, prefix):
    d = defaultdict(set)
    for line in fetched[fn][0].read_text().splitlines():
        if not line.strip():
            continue
        a, b = line.split("\t")
        m = re.match(r"path:(?:map|rn|ec|ko)(\d{5})$", b)
        if m:
            d[a.replace(prefix, "")].add(int(m.group(1)))
    return d


rxn2map = read_link("link_pathway_reaction.txt", "rn:")
ec2map = read_link("link_pathway_enzyme.txt", "ec:")
names = {}
for line in fetched["list_pathway.txt"][0].read_text().splitlines():
    if line.strip():
        a, b = line.split("\t", 1)
        names[int(re.sub(r"\D", "", a))] = b
kegg_rxns = {l.split("\t")[0].replace("rn:", "") for l in fetched["list_reaction.txt"][0].read_text().splitlines() if l}
kegg_ecs = {l.split("\t")[0].replace("ec:", "") for l in fetched["list_enzyme.txt"][0].read_text().splitlines() if l}

# ---- 2. Human-GEM annotations ------------------------------------------------------------------------
import cobra  # noqa: E402
gem = cobra.io.read_sbml_model(str(GEM))


def as_list(x):
    return [] if x is None else ([x] if isinstance(x, str) else list(x))


rows, stat = [], defaultdict(int)
for r in gem.reactions:
    kr = [k.strip() for k in as_list(r.annotation.get("kegg.reaction")) if k.strip()]
    ec = [e.strip() for e in as_list(r.annotation.get("ec-code")) if e.strip()]
    ec_full = [e for e in ec if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", e)]
    base = dict(reaction=r.id, subsystem=r.subsystem or "", boundary=bool(r.boundary),
                kegg_reaction=";".join(kr), ec_code=";".join(ec),
                kegg_reaction_not_in_kegg=";".join(k for k in kr if k not in kegg_rxns),
                ec_not_in_kegg=";".join(e for e in ec_full if e not in kegg_ecs))
    maps_r = set().union(*[rxn2map.get(k, set()) for k in kr]) if kr else set()
    keep_r = {m for m in maps_r if not EXCL_LO <= m <= EXCL_HI}
    maps_e = set().union(*[ec2map.get(e, set()) for e in ec_full]) if ec_full else set()
    keep_e = {m for m in maps_e if not EXCL_LO <= m <= EXCL_HI}
    if keep_r:
        source, keep = "kegg_reaction", keep_r
    elif keep_e:
        source, keep = "ec", keep_e
    else:
        source, keep = "", set()
        if not kr and not ec:
            reason = "no_annotation"
        elif (maps_r - keep_r) or (maps_e - keep_e):
            reason = "only_global_overview_map"
        elif kr and all(k not in kegg_rxns for k in kr) and not ec_full:
            reason = "id_not_in_kegg"
        elif ec and not ec_full and not kr:
            reason = "ec_incomplete_only"
        else:
            reason = "annotated_no_map"
    stat[source or reason] += 1
    if keep:
        for m in sorted(keep):
            rows.append({**base, "map_id": f"map{m:05d}", "map_name": names.get(m, ""), "source": source,
                         "n_maps_of_reaction": len(keep), "unmapped_reason": ""})
    else:
        rows.append({**base, "map_id": "", "map_name": "", "source": "", "n_maps_of_reaction": 0,
                     "unmapped_reason": reason})
tab = pd.DataFrame(rows)

# ---- 3. write ----------------------------------------------------------------------------------------
n_rx = len(gem.reactions)
mapped = tab[tab.map_id != ""]
n_mapped = mapped.reaction.nunique()
hdr = [f"# Human-GEM reaction -> KEGG pathway map (script_python/145_kegg_reaction_map.py, {datetime.now().isoformat(timespec='seconds')})",
       f"# sources: {GEM} ({n_rx} reactions; cobra {cobra.__version__}); KEGG REST raw files {KDIR} ({release})",
       "# rule: kegg.reaction -> link/pathway/reaction first; ec-code (complete 4-level EC only) -> link/pathway/enzyme only when the first gives no map outside 01100-01299",
       "# excluded maps: KEGG global and overview maps map01100-map01299; multi-map membership kept (one row per reaction x map); unmapped reactions kept with map_id empty and unmapped_reason",
       f"# n reactions {n_rx}; mapped {n_mapped} (kegg_reaction {stat['kegg_reaction']}, ec {stat['ec']}); rows {len(tab)}; maps {mapped.map_id.nunique()}",
       "# NOT for redistribution (contains KEGG map names); regenerate with the script"]


def write(df, p, header):
    p = safe_path(p)
    with open(p, "w") as fh:
        fh.write("\n".join(header) + "\n")
        df.to_csv(fh, sep="\t", index=False)
    print("wrote", p)


write(tab, KDIR / "humangem_reaction_kegg_map.tsv", hdr)

# per-map size, all maps a Human-GEM reaction links to (excluded ones listed with excluded=True, from the raw link)
allmap = defaultdict(set)
for r in gem.reactions:
    for k in as_list(r.annotation.get("kegg.reaction")):
        for m in rxn2map.get(k.strip(), ()):
            allmap[m].add(r.id)
size = mapped.groupby(["map_id", "map_name"]).agg(n_reactions_humangem=("reaction", "nunique"),
                                                   n_via_kegg_reaction=("source", lambda s: int((s == "kegg_reaction").sum())),
                                                   n_via_ec=("source", lambda s: int((s == "ec").sum()))).reset_index()
excl = pd.DataFrame([{"map_id": f"map{m:05d}", "map_name": names.get(m, ""), "n_reactions_humangem": len(v),
                      "n_via_kegg_reaction": len(v), "n_via_ec": 0} for m, v in allmap.items() if EXCL_LO <= m <= EXCL_HI])
size["excluded"] = False
excl["excluded"] = True
size = pd.concat([size, excl]).sort_values("map_id")
write(size, KDIR / "kegg_map_size.tsv", hdr[:2] + ["# per map: n distinct Human-GEM reactions assigned by the rule; excluded rows (global/overview) count kegg.reaction links only, for information"])

# EC-route check on reactions mapped by kegg.reaction that also carry a complete EC: how far the EC route's map set
# agrees with the direct route (the EC route is the fallback for 'ec' rows, so its over-assignment is reported)
n_both = n_sub = n_same = 0; jac = 0.0
for r in gem.reactions:
    kr = [k.strip() for k in as_list(r.annotation.get("kegg.reaction")) if k.strip()]
    ecf = [e.strip() for e in as_list(r.annotation.get("ec-code")) if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", e.strip())]
    R_ = {m for k in kr for m in rxn2map.get(k, ()) if not EXCL_LO <= m <= EXCL_HI}
    E_ = {m for e in ecf for m in ec2map.get(e, ()) if not EXCL_LO <= m <= EXCL_HI}
    if R_ and E_:
        n_both += 1; n_sub += R_ <= E_; n_same += R_ == E_; jac += len(R_ & E_) / len(R_ | E_)
cov = pd.DataFrame([("reactions_HumanGEM", n_rx),
                    ("with_kegg_reaction_annotation", int((tab.drop_duplicates("reaction").kegg_reaction != "").sum())),
                    ("with_ec_annotation", int((tab.drop_duplicates("reaction").ec_code != "").sum())),
                    ("mapped_any", n_mapped), ("mapped_via_kegg_reaction", stat["kegg_reaction"]), ("mapped_via_ec", stat["ec"])]
                   + [(f"unmapped_{k}", v) for k, v in sorted(stat.items()) if k not in ("kegg_reaction", "ec")]
                   + [("unmapped_boundary_reactions", int(tab[(tab.map_id == "") & tab.boundary].reaction.nunique())),
                      ("unmapped_transport_subsystem", int(tab[(tab.map_id == "") & tab.subsystem.str.contains("Transport", case=False)].reaction.nunique())),
                      ("maps_with_ge1_reaction", mapped.map_id.nunique()),
                      ("reactions_in_ge2_maps", int(tab.drop_duplicates("reaction").n_maps_of_reaction.ge(2).sum())),
                      ("ec_route_check_reactions_with_both_routes", n_both),
                      ("ec_route_check_reaction_maps_subset_of_ec_maps", n_sub),
                      ("ec_route_check_identical_map_sets", n_same),
                      ("ec_route_check_mean_jaccard_x1000", round(1000 * jac / max(n_both, 1))),
                      ("kegg_reaction_ids_not_in_kegg_list", int(sum(len(x.split(";")) for x in tab.drop_duplicates("reaction").kegg_reaction_not_in_kegg if x)))],
                   columns=["quantity", "n"])
write(cov, ROOT / f"result/external_plan_20261103/kegg_map_coverage_{args.date}.tsv", hdr[:5])
print(cov.to_string(index=False))
