#!/bin/bash
# Medium-input GNN on dataset_v2, low priority (user ruling 2026-11-02: after the main-flow items, on idle GPUs).
# A' gene-disjoint, 81b without --no-medium, fixed_cosine 30 epochs, seeds 0-2 x folds 0-4, lock protocol_lock_v3_tolpatch_medinput_20261102.json
# (script_python/134), out runs_v2_med. Waits for the PIDs given as arguments (main-flow GPU jobs) to exit first. GPUs 0-2 only (never 3).
#   dry   A gene s0 fold0, 2 epochs, 150 batches, lock check skipped, out _dry_smoke
#   grid  [PID ...] wait, then 15 runs, PER_GPU chains per GPU, then 101 lock check
# Launch: nohup bash script_bash/135_run_medium_input_lowprio.sh grid <PID> > result/gnn_20261030_tolpatch/runs_v2_med/logs/run_med_grid.log 2>&1 &
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$R/result/gnn_20261030_tolpatch; DS=$T/dataset_v2; V=$T/runs_v2_med; L=$V/logs; mkdir -p $L
W=${WORKERS:-4}; NICE=${NICE:-10}; PER_GPU=${PER_GPU:-3}; TAG=_tolpatch; LOCK=$T/protocol_lock_v3_tolpatch_medinput_20261102.json
STAGE=$1; shift
echo "medium input $STAGE start $(date); dataset $DS md5 $(md5sum $DS/samples.tsv | cut -c1-32); lock $LOCK"
job() { g=$1 s=$2 f=$3; t=A_gene_fixedcos30_s${s}${TAG}_fold$f
    CUDA_VISIBLE_DEVICES=$g nice -n $NICE $PYG script_python/81b_gnn_train_v2.py --graph A --split gene --seed $s --fold $f \
        --max-epochs 30 --protocol fixed_cosine --protocol-lock $LOCK --dataset $DS --out-root $V --tag $TAG --workers $W > $L/train_$t.log 2>&1 \
        && echo "done $t gpu$g $(date) $(grep -o 'PASS arm=[a-z_]*\|MISMATCH.*' $L/train_$t.log | head -1)" || echo "TRAIN FAILED $t gpu$g $(date)"; }
case $STAGE in
dry) mkdir -p $T/_dry_smoke; CUDA_VISIBLE_DEVICES=0 $PYG script_python/81b_gnn_train_v2.py --graph A --split gene --seed 0 --fold 0 \
        --max-epochs 2 --min-epochs 1 --max-train-batches 150 --protocol fixed_cosine --protocol-lock none --dataset $DS --out-root $T/_dry_smoke \
        --tag ${TAG}_med_dry_$(date +%H%M%S) --workers $W > $T/_dry_smoke/train_med_dry.log 2>&1 && echo "dry OK" || echo "DRY FAILED"
    grep -E "^medium:" $T/_dry_smoke/train_med_dry.log; tail -2 $T/_dry_smoke/train_med_dry.log;;
grid) for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 120; done; echo "waited PID $p $(date)"; done
    jobs=(); for s in 0 1 2; do for f in 0 1 2 3 4; do jobs+=("$s $f"); done; done; nch=$((3 * PER_GPU))
    for c in $(seq 0 $((nch - 1))); do ( g=$((c % 3)); for k in $(seq $c $nch $((${#jobs[@]} - 1))); do job $g ${jobs[$k]}; done ) & done
    wait; echo "grid end $(date)"
    $PYG script_python/101_check_protocol_v3.py --lock $LOCK --out-root $V > $L/check_protocol_medinput.log 2>&1; tail -3 $L/check_protocol_medinput.log;;
*) echo "usage: $0 dry | grid [PID ...]"; exit 2;;
esac
