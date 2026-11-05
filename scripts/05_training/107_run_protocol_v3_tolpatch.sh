#!/bin/bash
# Protocol v3 grid on the solver-tolerance-patched dataset (user decision 2026-10-30 13:18 KST, option A). New file; the 2026-10-28
# runners (99, 102) and their stopped runs are untouched. Same training rule as script_bash/99_run_protocol_v3.sh grid
# (81b --protocol fixed_cosine, fixed 30 epochs, cosine 1e-3 -> 1e-5, last-epoch checkpoint, per-epoch ckpt/, --no-medium), with
#   dataset  result/gnn_20261030_tolpatch/dataset (81b --dataset; recorded in config.json)
#   lock     result/gnn_20261030_tolpatch/protocol_lock_v3_tolpatch_20261030.json (dataset is a locked parameter)
#   out root result/gnn_20261030_tolpatch/runs_v2, folder tag _tolpatch: <graph>_<split>_nomed[_meanmax]_fixedcos30_s<seed>_tolpatch/fold<k>
# After every fold: 100_monitor_v3.py (monitoring only). Stage end: 101_check_protocol_v3.py --lock <new lock>.
# Usage: bash script_bash/107_run_protocol_v3_tolpatch.sh dry|grid|monitor|check
#   dry      one short training job (B gene fold 3 seed 0, 2 epochs, 150 batches, tag _tolpatch_dry, lock check skipped) on GPU 0;
#            output folder B_gene_nomed_meanmax_fixedcos2_s0_tolpatch_dry is a smoke artefact, never evaluated (85 sees the variant token
#            "dry"; delete or ignore). Reports epoch seconds for the wall-time estimate.
#   grid     A' (attn_max) and B (--readout mean_max), --no-medium, splits gene + line, seeds 0-2, folds 0-4, 30 epochs = 60 runs.
#            Placement as 99 (one B ~15 GB + one A' ~3 GB per 24 GB GPU; GPUs 0-2 only, NEVER GPU 3):
#              GPU 0: B gene s0,s1 | A' gene s0,s1      GPU 1: B gene s2 + B line s0 | A' gene s2 + A' line s0      GPU 2: B line s1,s2 | A' line s1,s2
#            Env WORKERS (default 6), NICE (default 5). Worker count does not change results (sampler indices drawn in the main process).
#   monitor  re-run 100_monitor_v3.py on every *fixedcos*tolpatch run (GPUs 0-2, three in parallel).
#   check    101_check_protocol_v3.py --lock <new lock> only.
# Log lines: "done <tag>" per fold, "TRAIN FAILED <tag>" / "MONITOR FAILED <tag>" on failure, final "done protocol v3 tolpatch <stage>".
# Launch: nohup bash script_bash/107_run_protocol_v3_tolpatch.sh grid > result/gnn_20261030_tolpatch/runs_v2/logs/run_protocol_v3_grid.log 2>&1 &
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$R/result/gnn_20261030_tolpatch; DS=$T/${DSN:-dataset_v2}; V=$T/runs_v2; L=$V/logs; mkdir -p $L   # dataset_v2 = 14 non-optimal knockouts excluded (user ruling 2026-10-30 17:05); absolute: config.json dataset / out_root must equal the lock's
W=${WORKERS:-6}; NICE=${NICE:-5}; LOCK=$T/protocol_lock_v3_tolpatch_20261030.json; TAG=_tolpatch
STAGE=$1; [ -z "$STAGE" ] && { echo "usage: $0 dry|grid|monitor|check"; exit 2; }
[ -s $DS/samples.tsv ] || { echo "dataset missing: $DS/samples.tsv"; exit 3; }
echo "protocol v3 tolpatch $STAGE start $(date); dataset $DS (samples.tsv md5 $(md5sum $DS/samples.tsv | cut -c1-32)); workers $W nice $NICE; lock $LOCK; uptime: $(uptime)"
job() { g=$1 gr=$2 sp=$3 s=$4 f=$5 E=$6
    tok=${gr}_${sp}_nomed$([ $gr = B ] && echo _meanmax)_fixedcos${E}_s${s}${TAG}; t=${tok}_fold$f
    ro=$([ $gr = B ] && echo "--readout mean_max")
    CUDA_VISIBLE_DEVICES=$g nice -n $NICE $PYG script_python/81b_gnn_train_v2.py --graph $gr --split $sp --seed $s --fold $f --no-medium \
        --max-epochs $E --protocol fixed_cosine --protocol-lock $LOCK --dataset $DS --out-root $V --tag $TAG --workers $W $ro > $L/train_$t.log 2>&1 \
        || { echo "TRAIN FAILED $t gpu$g $(date)"; return 1; }
    CUDA_VISIBLE_DEVICES=$g nice -n $NICE $PYG script_python/100_monitor_v3.py --run $V/$tok/fold$f --workers $W > $L/monitor_$t.log 2>&1 \
        || echo "MONITOR FAILED $t gpu$g $(date)"
    echo "done $t gpu$g $(date)"; }
case $STAGE in
  dry)
    t=B_gene_nomed_meanmax_fixedcos2_s0${TAG}_dry_fold3
    mkdir -p $T/_dry_smoke; CUDA_VISIBLE_DEVICES=0 nice -n $NICE $PYG script_python/81b_gnn_train_v2.py --graph B --split gene --seed 0 --fold 3 --no-medium --readout mean_max \
        --max-epochs 2 --min-epochs 1 --max-train-batches 150 --protocol fixed_cosine --protocol-lock none --dataset $DS --out-root $T/_dry_smoke --tag ${TAG}_dry --workers $W \
        > $T/_dry_smoke/train_$t.log 2>&1 && echo "dry OK $t $(date)" || echo "DRY FAILED $t $(date)"
    grep -E "samples|graph B fixed|epoch_s|done" $T/_dry_smoke/train_$t.log | cut -c1-300;;
  grid)
    { for s in 0 1; do for f in 0 1 2 3 4; do job 0 B gene $s $f 30; done; done; } &
    { for s in 0 1; do for f in 0 1 2 3 4; do job 0 A gene $s $f 30; done; done; } &
    { for f in 0 1 2 3 4; do job 1 B gene 2 $f 30; done; for f in 0 1 2 3 4; do job 1 B line 0 $f 30; done; } &
    { for f in 0 1 2 3 4; do job 1 A gene 2 $f 30; done; for f in 0 1 2 3 4; do job 1 A line 0 $f 30; done; } &
    { for s in 1 2; do for f in 0 1 2 3 4; do job 2 B line $s $f 30; done; done; } &
    { for s in 1 2; do for f in 0 1 2 3 4; do job 2 A line $s $f 30; done; done; } &
    wait
    $PYG script_python/101_check_protocol_v3.py --lock $LOCK > $L/check_protocol_v3_grid.log 2>&1 || echo "PROTOCOL CHECK FAILED grid $(date)"
    tail -1 $L/check_protocol_v3_grid.log;;
  monitor)
    runs=($(ls -d $V/*fixedcos*${TAG} 2>/dev/null)); n=${#runs[@]}
    for g in 0 1 2; do { for ((k=g; k<n; k+=3)); do
        CUDA_VISIBLE_DEVICES=$g nice -n $NICE $PYG script_python/100_monitor_v3.py --run ${runs[k]} --workers $W \
            > $L/monitor_$(basename ${runs[k]}).log 2>&1 || echo "MONITOR FAILED $(basename ${runs[k]}) $(date)"
        echo "done monitor $(basename ${runs[k]}) gpu$g $(date)"; done; } & done
    wait;;
  check)
    $PYG script_python/101_check_protocol_v3.py --lock $LOCK;;
  *) echo "unknown stage $STAGE"; exit 2;;
esac
echo "done protocol v3 tolpatch $STAGE $(date)"
