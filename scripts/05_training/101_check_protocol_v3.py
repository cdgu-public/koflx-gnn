"""Protocol v3 checker (2026-10-30; review_stopping.md section 3, items 2 and 3).
(a) Reader check: every script under script_python/ and script_bash/ is grepped for references to the monitoring files
    (monitor_val.tsv / monitor_test.tsv) or the per-epoch checkpoints (ckpt/ep<k>.pt). Only the writers and the runner
    listed in the lock file may mention them; any other file, in particular 85_eval_gnn_v2.py, fails the check.
    85_eval_gnn_v2.py must read prediction files named pred_test.tsv only.
(b) Lock check: every fold config.json under <out_root>/*fixedcos*/fold*/ is compared with the lock file
    (result/gnn_20261028/protocol_lock_v3_20261030.json): the common keys must match, and (graph, readout, split,
    seed, fold, max_epochs) must belong to one registered arm (grid or dry_test).
    Control arms (added 2026-10-30, status.md 08:40 KST ruling): an arm may carry "script" (90_gnn_e2e3_v2.py or
    93_mlp_flat_v2.py; configs of those scripts record their script name under --protocol fixed_cosine, 81b configs carry no
    script key), "common_override" (common keys with another value for this arm, e.g. out_root of the MLP runs),
    "common_skip" (common keys the script has no counterpart for) and "flags" (control flags that must match exactly,
    e.g. shuffle_graph, rxn_embed, wt_only, recipe). Arms without "graphs" skip the graph/readout check (MLP).
    The config sweep covers the lock out_root and every out_root named in a common_override.
check_config(cfg, lock) is also imported by 81b_gnn_train_v2.py, 90_gnn_e2e3_v2.py and 93_mlp_flat_v2.py at the end of
every fixed_cosine run.
Usage: python 101_check_protocol_v3.py [--lock FILE] [--configs] [--readers]   (default: both). Exit 0 = PASS, 1 = FAIL.
Env: crcgem_gnn (standard library only).
"""
import argparse, glob, json, os, re, sys

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
LOCK = f"{ROOT}/result/gnn_20261028/protocol_lock_v3_20261030.json"
PAT = re.compile(r"monitor_(val|test|\*|\w*)\.tsv|monitor_\{|ckpt/ep|/ckpt\b|ckpt_dir")


def check_config(cfg, lock):
    """Return (ok, arm_name, mismatches). The arm with the fewest mismatches is reported when none matches."""
    best = (None, None)
    for arm, spec in lock["arms"].items():
        mism = []
        if cfg.get("script") != spec.get("script"):   # 81b configs have no script key; 90 / 93 record theirs
            mism.append(f"script: {cfg.get('script')!r} != {arm} {spec.get('script')!r}")
        common = dict(lock["common"]); common.update(spec.get("common_override", {}))
        for k in spec.get("common_skip", []):
            common.pop(k, None)
        for k, v in common.items():
            if k not in cfg:
                mism.append(f"{k}: missing")
            elif cfg[k] != v:
                mism.append(f"{k}: {cfg[k]!r} != lock {v!r}")
        if cfg.get("max_epochs") != spec["max_epochs"]:
            mism.append(f"max_epochs: {cfg.get('max_epochs')!r} != {arm} {spec['max_epochs']}")
        if "graphs" in spec:
            g = cfg.get("graph")
            if g not in spec["graphs"]:
                mism.append(f"graph: {g!r} not in {arm} {sorted(spec['graphs'])}")
            elif cfg.get("readout") != spec["graphs"][g]["readout"]:
                mism.append(f"readout: {cfg.get('readout')!r} != {arm} {spec['graphs'][g]['readout']} for graph {g}")
        for k in ("split", "seed", "fold"):
            if cfg.get(k) not in spec[k + "s"]:
                mism.append(f"{k}: {cfg.get(k)!r} not in {arm} {spec[k + 's']}")
        for k, v in spec.get("flags", {}).items():
            if k not in cfg:
                mism.append(f"{k}: missing (control flag)")
            elif cfg[k] != v:
                mism.append(f"{k}: {cfg[k]!r} != {arm} flag {v!r}")
        if not mism:
            return True, arm, []
        if best[1] is None or len(mism) < len(best[1]):
            best = (arm, mism)
    return False, best[0], best[1]


def check_readers(lock):
    allowed = set(lock["monitoring"]["allowed_readers_of_monitor_or_ckpt"])
    # 2026-11-03: path mentions that open no file (126: manifest description string :75, .gitignore pattern :294), checked by
    # result/gnn_20261030_tolpatch/verify_medium_input_20261103.md; kept outside the frozen locks
    allowed |= {"script_python/126_build_code_package_20261101.py"}
    fails, hits = [], []
    for path in sorted(glob.glob(f"{ROOT}/script_python/*.py") + glob.glob(f"{ROOT}/script_bash/*.sh")):
        rel = os.path.relpath(path, ROOT)
        for i, line in enumerate(open(path, errors="replace"), 1):
            if PAT.search(line):
                hits.append(f"{rel}:{i}: {line.rstrip()[:120]}")
                if rel not in allowed:
                    fails.append(f"{rel}:{i}: {line.rstrip()[:120]}")
    ev = f"{ROOT}/script_python/85_eval_gnn_v2.py"
    pred_names = set(re.findall(r"pred_[a-z_]*\.tsv", open(ev).read()))
    if pred_names != {lock["monitoring"]["evaluation_reads_only"]}:
        fails.append(f"85_eval_gnn_v2.py references prediction files {sorted(pred_names)}, allowed {lock['monitoring']['evaluation_reads_only']}")
    return fails, hits


def check_configs(lock, out_root=None):
    roots = [out_root] if out_root else sorted({lock["common"]["out_root"]} | {spec["common_override"]["out_root"]
                                                for spec in lock["arms"].values() if "out_root" in spec.get("common_override", {})})
    res = []
    for root in roots:
        for p in sorted(glob.glob(f"{root}/*fixedcos*/fold*/config.json")):
            ok, arm, mism = check_config(json.load(open(p)), lock)
            res.append((os.path.relpath(p, ROOT), ok, arm, mism))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", default=LOCK)
    ap.add_argument("--out-root", default=None, help="run root to sweep for *fixedcos*/fold*/config.json (default: lock out_root plus every arm out_root)")
    ap.add_argument("--configs", action="store_true"); ap.add_argument("--readers", action="store_true")
    a = ap.parse_args()
    do_all = not (a.configs or a.readers)
    lock = json.load(open(a.lock)); failed = False
    print(f"protocol lock: {a.lock} ({lock['name']}, written {lock['written']})")
    if do_all or a.readers:
        fails, hits = check_readers(lock)
        print(f"[readers] {len(hits)} reference(s) to monitor_*.tsv / ckpt in script_python + script_bash:")
        for h in hits:
            print("   ", h)
        print(f"[readers] {'FAIL' if fails else 'PASS'}: " + ("; ".join(fails) if fails else "only the allowed writers/runner reference them; 85_eval reads pred_test.tsv only"))
        failed |= bool(fails)
    if do_all or a.configs:
        res = check_configs(lock, a.out_root)
        print(f"[configs] {len(res)} config.json under *fixedcos*/fold*/ (roots: {a.out_root or 'lock out_root + arm out_roots'})")
        for rel, ok, arm, mism in res:
            print(f"    {'PASS' if ok else 'FAIL'} {rel} arm={arm}" + ("" if ok else " " + "; ".join(mism)))
        bad = [r for r in res if not r[1]]
        print(f"[configs] {'FAIL' if bad else 'PASS'}: {len(res) - len(bad)}/{len(res)} match the lock")
        failed |= bool(bad)
    print("PROTOCOL CHECK " + ("FAIL" if failed else "PASS"))
    sys.exit(1 if failed else 0)
