"""Post-hoc monitoring curves for fixed_cosine runs of 81b (2026-10-30; review_stopping.md section 3).
For each fold folder with ckpt/ep<k>.pt: re-executes 81b up to the model definition from the fold's config.json (same data,
split, graph builder; split.json is compared to the fold's), loads every per-epoch checkpoint, and scores the validation and
the test fold with each one in a single pass over the data (every batch is built once and forwarded through all checkpoints).
Writes <fold>/monitor_val.tsv and <fold>/monitor_test.tsv: # header lines (source run, protocol parameters, n, date, purpose),
columns epoch, auroc, auprc, bce, n. Nothing else is written into the fold folder.
MONITORING ONLY. These files are never read by 85_eval_gnn_v2.py or by any selection step (checker 101_check_protocol_v3.py).
Control runs (2026-10-30, decision log 08:40 KST ruling): config.json key "script" selects the training script to re-execute,
90_gnn_e2e3_v2.py (E3 relabel / identity embedding / WT-only; control flags passed from config.json) or 93_mlp_flat_v2.py (MLP on
the flat fluxome; checkpoints scored on the in-memory feature matrix). 81b configs carry no script key (unchanged path).
Usage: CUDA_VISIBLE_DEVICES=<g> python 100_monitor_v3.py --run <run folder or fold folder> [more ...] [--workers 4] [--cpu]
Env: crcgem_gnn.
"""
import argparse, copy, glob, json, os, re, shutil, sys, tempfile, time
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = "/data/workspace/cdgu/Winery/seeding/CRC_GEM"
TRAINERS = {None: f"{ROOT}/script_python/81b_gnn_train_v2.py", "90_gnn_e2e3_v2.py": f"{ROOT}/script_python/90_gnn_e2e3_v2.py",
            "93_mlp_flat_v2.py": f"{ROOT}/script_python/93_mlp_flat_v2.py"}
ap = argparse.ArgumentParser()
ap.add_argument("--run", nargs="+", required=True, help="run folder(s) (all fold*/ inside) or fold folder(s)")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--cpu", action="store_true")
margs = ap.parse_args()

folds = []
for r in margs.run:
    r = r.rstrip("/")
    folds += [r] if os.path.basename(r).startswith("fold") else sorted(glob.glob(f"{r}/fold*"))
SRCS = {}
for script, path in TRAINERS.items():
    src = open(path).read()
    src = src[:src.index("\ncfg = dict(vars(args)")]   # everything up to and including the model definition (81b/90: net/opt/predict/bce; 93: net/opt)
    if margs.cpu:
        src = src.replace('dev = torch.device("cuda:0")', 'dev = torch.device("cpu")')   # 81b; 90 and 93 take --device
    SRCS[script] = src


def rel(p):
    p = os.path.abspath(p); return os.path.relpath(p, ROOT) if p.startswith(ROOT + "/") else p


def build_argv(cfg, tmp):
    """Command line that re-executes the fold's training script up to the model definition with the fold's own settings."""
    script = cfg.get("script")
    if script == "93_mlp_flat_v2.py":
        argv = ["93", "--split", cfg["split"], "--fold", str(cfg["fold"]), "--recipe", cfg["recipe"], "--seed", str(cfg["seed"]),
                "--nfold", str(cfg["nfold"]), "--max-epochs", str(cfg["max_epochs"]), "--batch", str(cfg["batch"]), "--lr", str(cfg["lr"]),
                "--hidden", str(cfg["hidden"][0]), str(cfg["hidden"][1]), "--train-probe", str(cfg["train_probe"]),
                "--device", "cpu" if margs.cpu else cfg["device"], "--mask-file", cfg["mask_file"], "--ref-root", cfg["ref_root"],
                "--out-root", tmp, "--tag", cfg["tag"], "--protocol", "fixed_cosine", "--lr-min", str(cfg["lr_min"]), "--protocol-lock", "none"]
        argv += ["--smoke"] if cfg["smoke"] else []
        argv += ["--no-medium"] if cfg["no_medium"] else []
        argv += ["--dataset", cfg["dataset"]] if cfg.get("dataset") else []          # 2026-10-30 tolpatch: configs record their dataset
        argv += ["--ref-run", cfg["ref_run"]] if cfg.get("ref_run") else []
        return argv
    argv = ["81b" if script is None else "90", "--graph", cfg["graph"], "--split", cfg["split"], "--fold", str(cfg["fold"]), "--nfold", str(cfg["nfold"]),
            "--seed", str(cfg["seed"]), "--batch", str(cfg["batch"]), "--hidden", str(cfg["hidden"]), "--workers", str(margs.workers),
            "--mask-file", cfg["mask_file"], "--protocol", "fixed_cosine", "--max-epochs", str(cfg["max_epochs"]),
            "--lr", str(cfg["lr"]), "--lr-min", str(cfg["lr_min"]), "--out-root", tmp, "--tag", cfg["tag"]]
    argv += ["--presence"] if cfg["presence"] else []
    argv += ["--no-medium"] if cfg["no_medium"] else []
    argv += ["--dataset", cfg["dataset"]] if cfg.get("dataset") else []              # 2026-10-30 tolpatch: configs record their dataset
    if script is None:
        argv += ["--readout", cfg["readout"]]
    else:   # 90: control flags (the readout is script 81's attention + max pooling, not a flag)
        argv += ["--device", "cpu" if margs.cpu else cfg["device"], "--train-probe", str(cfg["train_probe"]), "--rxn-embed", str(cfg["rxn_embed"]),
                 "--stop-metric", cfg["stop_metric"], "--train-subset", str(cfg["train_subset"]), "--protocol-lock", "none"]
        argv += ["--shuffle-graph", cfg["shuffle_graph"]] if cfg["shuffle_graph"] else []
        argv += ["--wt-only"] if cfg["wt_only"] else []
        argv += ["--disabled-flag"] if cfg.get("disabled_flag") else []          # 2026-11-01 flag arm (90 --disabled-flag; missing key = off)
        argv += ["--permute-labels"] if cfg["permute_labels"] else []
    return argv


for fd in folds:
    t0 = time.time()
    cfg = json.load(open(f"{fd}/config.json"))
    assert cfg.get("protocol") == "fixed_cosine", f"{fd}: not a fixed_cosine run (protocol={cfg.get('protocol')})"
    script = cfg.get("script")
    assert script in TRAINERS, f"{fd}: unknown training script {script!r}"
    TRAIN, SRC = TRAINERS[script], SRCS[script]
    ckpts = sorted(glob.glob(f"{fd}/ckpt/ep*.pt"))
    assert ckpts, f"{fd}: no ckpt/ep*.pt"
    tmp = tempfile.mkdtemp(prefix="monitor_v3_")
    sys.argv = build_argv(cfg, tmp)
    g = {"__name__": "monitor"}
    exec(compile(SRC, TRAIN, "exec"), g)
    # the re-executed split must be the fold's split
    sj, sj0 = json.load(open(f"{g['OUT']}/split.json")), json.load(open(f"{fd}/split.json"))
    assert sj == sj0, f"{fd}: re-executed split differs from split.json"
    shutil.rmtree(tmp)
    dev, lab, tgt = g["dev"], g["lab"], g["tgt"]
    nets = []
    for ck in ckpts:   # the re-executed script's own freshly built model, one copy per checkpoint
        n = copy.deepcopy(g["net"]).to(dev)
        n.load_state_dict(torch.load(ck, weights_only=True, map_location=dev)); n.eval(); nets.append(n)
    last_sd = torch.load(f"{fd}/model.pt", weights_only=True, map_location=dev)
    last_is_model = all(torch.equal(last_sd[k], v) for k, v in nets[-1].state_dict().items())
    epochs = [int(re.search(r"ep(\d+)\.pt$", c).group(1)) for c in ckpts]

    if script == "93_mlp_flat_v2.py":
        Xt = g["Xt"]

        def predict_all(ids):
            ids = np.asarray(ids); xb = Xt[torch.tensor(ids, device=dev)]
            with torch.no_grad():
                return ids, [torch.sigmoid(n(xb).squeeze(1)).cpu().numpy() for n in nets]
    else:
        loader = g["loader"]

        def predict_all(ids):
            I, P = [], [[] for _ in nets]
            with torch.no_grad():
                for b in loader(ids, False):
                    b = b.to(dev); I.append(b.idx.cpu().numpy())
                    for k, n in enumerate(nets):
                        P[k].append(torch.sigmoid(n(b)).cpu().numpy())
            return np.concatenate(I), [np.concatenate(p) for p in P]

    def bce(i, p):
        p = np.clip(p, 1e-7, 1 - 1e-7); t = tgt[i]
        return float(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean())

    for name, ids in (("val", g["val"]), ("test", g["test"])):
        ii, PP = predict_all(ids)
        rows = [dict(epoch=e, auroc=round(float(roc_auc_score(lab[ii], p)), 5), auprc=round(float(average_precision_score(lab[ii], p)), 5),
                     bce=round(bce(ii, p), 5), n=len(ii)) for e, p in zip(epochs, PP)]
        hdr = [f"# monitor_{name}.tsv, MONITORING ONLY: per-epoch {name} metrics from the saved checkpoints; not read by 85_eval_gnn_v2.py or any selection step",
               f"# source run: {rel(fd)}  (script {script or '81b_gnn_train_v2.py'}, " + (f"MLP recipe {cfg['recipe']}, hidden {cfg['hidden']}" if script == "93_mlp_flat_v2.py" else
                 f"graph {cfg['graph']}, readout {cfg['readout']}" + (f", shuffle_graph {cfg['shuffle_graph']!r}, rxn_embed {cfg['rxn_embed']}, wt_only {cfg['wt_only']}" if script else "") + f", presence {cfg['presence']}")
               + f", split {cfg['split']}, fold {cfg['fold']}, seed {cfg['seed']}, no_medium {cfg['no_medium']})",
               f"# protocol: {cfg['protocol']}, fixed budget {cfg['max_epochs']} epochs, {cfg['lr_schedule']} lr {cfg['lr']} -> {cfg['lr_min']}, checkpoint {cfg['checkpoint']}, early_stopping {cfg['early_stopping']}, batch {cfg['batch']}, hidden {cfg['hidden']}, mask {os.path.basename(cfg['mask_file'])}",
               f"# checkpoints: {len(ckpts)} files ckpt/ep*.pt; last checkpoint equals model.pt: {last_is_model}",
               f"# n_{name} {len(ii)}, positive rate {lab[ii].mean():.3f} (label = dependency >= 0.5; bce on the soft target)",
               f"# written {time.strftime('%Y-%m-%d %H:%M:%S %Z')} by script_python/100_monitor_v3.py; python {sys.version.split()[0]}, torch {torch.__version__}; auroc/auprc higher is better, bce lower is better"]
        with open(f"{fd}/monitor_{name}.tsv", "w") as fh:
            fh.write("\n".join(hdr) + "\n"); pd.DataFrame(rows).to_csv(fh, sep="\t", index=False)
        print(f"{rel(fd)} monitor_{name}.tsv: {len(rows)} epochs, n {len(ii)}, auroc first {rows[0]['auroc']:.3f} last {rows[-1]['auroc']:.3f}", flush=True)
    print(f"done monitor {rel(fd)} {time.time() - t0:.0f}s", flush=True)
