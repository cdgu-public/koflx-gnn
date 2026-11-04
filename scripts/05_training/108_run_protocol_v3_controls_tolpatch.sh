#!/bin/bash
# Protocol v3 controls on the solver-tolerance-patched dataset (user decision 2026-10-30 13:18 KST, option A). New file; equivalent of
# script_bash/102_run_protocol_v3_controls.sh with dataset result/gnn_20261030_tolpatch/dataset, lock
# result/gnn_20261030_tolpatch/protocol_lock_v3_tolpatch_20261030.json, out roots result/gnn_20261030_tolpatch/runs_v2[_mlp], tag _tolpatch.
# Same rule as the 107 grid: fixed 30 epochs, cosine 1e-3 -> 1e-5, last-epoch checkpoint, per-epoch ckpt/, --no-medium, gene-disjoint split,
# seeds 0-2, folds 0-4. Comparator = grid arm A_gene_nomed_fixedcos30_s<seed>_tolpatch.
#   relabel  90 --shuffle-graph relabel -> runs_v2/A_gene_nomed_shufrelabel_fixedcos30_s<s>_tolpatch/fold<f>
#   rid16    90 --rxn-embed 16          -> runs_v2/A_gene_nomed_rid16_fixedcos30_s<s>_tolpatch/fold<f>
#   wtonly   90 --wt-only               -> runs_v2/A_gene_nomed_wtonly_fixedcos30_s<s>_tolpatch/fold<f>
#   mlp81    93 --recipe gnn81 --no-medium   -> runs_v2_mlp/A_gene_nomed_mlp81_fixedcos30_s<s>_tolpatch/fold<f>
#   mlphgb   93 --recipe hgblike --no-medium -> runs_v2_mlp/A_gene_nomed_mlphgb_fixedcos30_s<s>_tolpatch/fold<f>
# The MLP controls read rel_nn_dist / dup_in_train from runs_v2/A_gene_nomed_fixedcos30_s0_tolpatch/fold<f>/pred_test.tsv (93 --ref-run), so the
# queue waits for the grid (final line "done protocol v3 tolpatch grid" in $GRID_LOG, or GPUs 0-2 all below $IDLE_MIB MiB on two checks 5 min apart).
# 6 workers (two per GPU 0, 1, 2; never GPU 3) pull from one seed-major queue. Before each job the worker waits until its GPU has < $GPU_BUSY_MIB MiB in use.
# Env: WORKERS (default 6), NICE (default 5), SEEDS (default "0 1 2"), FOLDS (default "0 1 2 3 4"), CONTROLS (default "relabel rid16 wtonly mlp81 mlphgb"),
#      GRID_LOG (default runs_v2/logs/run_protocol_v3_grid.log), WAIT=none skips the grid wait, IDLE_MIB (default 1000), GPU_BUSY_MIB (default 18000).
# Usage: bash script_bash/108_run_protocol_v3_controls_tolpatch.sh run|check
# Launch: nohup bash script_bash/108_run_protocol_v3_controls_tolpatch.sh run > result/gnn_20261030_tolpatch/runs_v2/logs/run_protocol_v3_controls.log 2>&1 &
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$R/result/gnn_20261030_tolpatch; DS=$T/${DSN:-dataset_v2}; V=$T/runs_v2; VM=$T/runs_v2_mlp; L=$V/logs; mkdir -p $L $VM   # dataset_v2 = 14 non-optimal knockouts excluded (user ruling 2026-10-30 17:05); absolute: config.json dataset / out_root / ref_root must equal the lock's
W=${WORKERS:-6}; NICE=${NICE:-5}; LOCK=$T/protocol_lock_v3_tolpatch_20261030.json; E=30; TAG=_tolpatch
SEEDS=${SEEDS:-"0 1 2"}; FOLDS=${FOLDS:-"0 1 2 3 4"}; CONTROLS=${CONTROLS:-"relabel rid16 wtonly mlp81 mlphgb"}
GRID_LOG=${GRID_LOG:-$L/run_protocol_v3_grid.log}; IDLE_MIB=${IDLE_MIB:-1000}; GPU_BUSY_MIB=${GPU_BUSY_MIB:-18000}
REFRUN=A_gene_nomed_fixedcos30_s0${TAG}
STAGE=$1; [ -z "$STAGE" ] && { echo "usage: $0 run|check"; exit 2; }
[ "$STAGE" = check ] && { $PYG script_python/101_check_protocol_v3.py --lock $LOCK; exit $?; }
[ "$STAGE" != run ] && { echo "unknown stage $STAGE"; exit 2; }
[ -s $DS/samples.tsv ] || { echo "dataset missing: $DS/samples.tsv"; exit 3; }
echo "protocol v3 tolpatch controls start $(date); dataset $DS; workers $W nice $NICE; seeds [$SEEDS] folds [$FOLDS] controls [$CONTROLS]; lock $LOCK; grid log $GRID_LOG; uptime: $(uptime)"
mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1; }
all_idle() { for g in 0 1 2; do [ "$(mem $g)" -lt $IDLE_MIB ] || return 1; done; return 0; }
if [ "${WAIT:-grid}" != none ]; then
    idle_streak=0
    until grep -q "done protocol v3 tolpatch grid" $GRID_LOG 2>/dev/null; do
        if all_idle; then idle_streak=$((idle_streak + 1)); else idle_streak=0; fi
        [ $idle_streak -ge 2 ] && { echo "GPUs 0-2 idle on two checks, no grid final line yet; starting $(date)"; break; }
        sleep 300
    done
    grep -q "done protocol v3 tolpatch grid" $GRID_LOG 2>/dev/null && echo "grid final line seen $(date): $(grep 'done protocol v3 tolpatch grid' $GRID_LOG | tail -1)"
fi
for f in $FOLDS; do [ -s $V/$REFRUN/fold$f/pred_test.tsv ] || echo "WARNING: $V/$REFRUN/fold$f/pred_test.tsv missing; MLP controls of fold $f will fail (93 --ref-run)"; done
echo "controls launch $(date); uptime: $(uptime); gpu mem used MiB: $(for g in 0 1 2; do echo -n "gpu$g=$(mem $g) "; done)"
JOBS=()
for s in $SEEDS; do
    for c in relabel rid16 wtonly; do case " $CONTROLS " in *" $c "*) for f in $FOLDS; do JOBS+=("$c $s $f"); done;; esac; done
    for c in mlp81 mlphgb; do case " $CONTROLS " in *" $c "*) for f in $FOLDS; do JOBS+=("$c $s $f"); done;; esac; done
done
N=${#JOBS[@]}; echo "$N jobs queued"
Q=$L/.controls_queue_$$; echo 0 > $Q
next() { ( flock -x 9; i=$(cat $Q); echo $((i + 1)) > $Q; echo $i; ) 9> $Q.lock; }
wait_gpu() { until [ "$(mem $1)" -lt $GPU_BUSY_MIB ]; do sleep 120; done; }
job() { g=$1 c=$2 s=$3 f=$4
    case $c in
      relabel) tok=A_gene_nomed_shufrelabel_fixedcos${E}_s$s$TAG; root=$V
               cmd="script_python/90_gnn_e2e3_v2.py --graph A --split gene --seed $s --fold $f --no-medium --shuffle-graph relabel --workers $W --out-root $V";;
      rid16)   tok=A_gene_nomed_rid16_fixedcos${E}_s$s$TAG; root=$V
               cmd="script_python/90_gnn_e2e3_v2.py --graph A --split gene --seed $s --fold $f --no-medium --rxn-embed 16 --workers $W --out-root $V";;
      wtonly)  tok=A_gene_nomed_wtonly_fixedcos${E}_s$s$TAG; root=$V
               cmd="script_python/90_gnn_e2e3_v2.py --graph A --split gene --seed $s --fold $f --no-medium --wt-only --workers $W --out-root $V";;
      mlp81)   tok=A_gene_nomed_mlp81_fixedcos${E}_s$s$TAG; root=$VM
               cmd="script_python/93_mlp_flat_v2.py --split gene --seed $s --fold $f --no-medium --recipe gnn81 --out-root $VM --ref-root $V --ref-run $REFRUN";;
      mlphgb)  tok=A_gene_nomed_mlphgb_fixedcos${E}_s$s$TAG; root=$VM
               cmd="script_python/93_mlp_flat_v2.py --split gene --seed $s --fold $f --no-medium --recipe hgblike --out-root $VM --ref-root $V --ref-run $REFRUN";;
    esac
    t=${tok}_fold$f; wait_gpu $g
    CUDA_VISIBLE_DEVICES=$g nice -n $NICE $PYG $cmd --max-epochs $E --protocol fixed_cosine --protocol-lock $LOCK --dataset $DS --tag $TAG > $L/train_$t.log 2>&1 \
        || { echo "TRAIN FAILED $t gpu$g $(date)"; return 1; }
    CUDA_VISIBLE_DEVICES=$g nice -n $NICE $PYG script_python/100_monitor_v3.py --run $root/$tok/fold$f --workers $W > $L/monitor_$t.log 2>&1 \
        || echo "MONITOR FAILED $t gpu$g $(date)"
    echo "done $t gpu$g $(date)"; }
worker() { g=$(( $1 % 3 ))
    while :; do i=$(next); [ "$i" -ge "$N" ] && break; job $g ${JOBS[$i]}; done; }
for w in 0 1 2 3 4 5; do worker $w & done
wait
rm -f $Q $Q.lock
$PYG script_python/101_check_protocol_v3.py --lock $LOCK > $L/check_protocol_v3_controls.log 2>&1 || echo "PROTOCOL CHECK FAILED controls $(date)"
tail -1 $L/check_protocol_v3_controls.log
echo "done protocol v3 tolpatch controls $(date)"
