#!/bin/bash
# Protocol v3 grid (graph A' only) on the two common-medium datasets (MOMA under a common medium; user ruling 2026-10-31,
# status.md 17:57 entry item 4; design result/common_medium_20261030/gnn_design_20261031.md). Copy of 107 with the dataset,
# lock, out root and folder tag of each medium; training rule unchanged (81b --protocol fixed_cosine, 30 epochs, --no-medium).
#   T     result/gnn_fixed_<M>_20261030_tolpatch, dataset $T/dataset, lock $T/protocol_lock_v3_common_<M>_20261031.json
#   runs  $T/runs_v2/A_<split>_nomed_fixedcos30_s<seed>_cm<M>/fold<k>     (M = DMEMF12_Gibco11320 | RPMI1640_Gibco11875)
# Usage: bash script_bash/116_run_protocol_v3_common_medium.sh dry|grid|check
#   dry   per medium and split: 2 epochs x 150 batches, fold 0 seed 0, lock none, out $T/_dry_smoke (smoke artefact, never evaluated).
#   grid  A' x {gene, line} x seeds 0-2 x folds 0-4 per medium = 60 runs; 6 chains, two per GPU 0, 1, 2 (NEVER GPU 3), both media
#         interleaved; each job waits until its GPU has < GPU_BUSY_MIB MiB in use (default 18000, as 108). Env WORKERS, NICE.
#   check 101_check_protocol_v3.py --lock <lock of each medium>.
# GRAPH=B (2026-11-01, user ruling: graph B on both common media): graph B with --readout mean_max (own-medium grid arm), lock
#   $T/protocol_lock_v3_common_<M>_AB_20261101.json (117 --with-b, graphs A and B), runs $T/runs_v2/B_<split>_nomed_meanmax_fixedcos30_s<seed>_cm<M>.
# Log lines: "done <tag>" per fold, "TRAIN FAILED" / "MONITOR FAILED", final "done protocol v3 common medium <stage>".
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GR=${GRAPH:-A}; case $GR in A) RO=attn_max; RT=;; B) RO=mean_max; RT=_meanmax;; *) echo "GRAPH must be A or B"; exit 2;; esac
W=${WORKERS:-6}; NICE=${NICE:-5}; GPU_BUSY_MIB=${GPU_BUSY_MIB:-18000}
MEDIA="DMEMF12_Gibco11320 RPMI1640_Gibco11875"
tdir() { echo $R/result/gnn_fixed_$1_20261030_tolpatch; }
lock() { [ $GR = B ] && echo $(tdir $1)/protocol_lock_v3_common_$1_AB_20261101.json || echo $(tdir $1)/protocol_lock_v3_common_$1_20261031.json; }
STAGE=$1; [ -z "$STAGE" ] && { echo "usage: $0 dry|grid|check"; exit 2; }
for M in $MEDIA; do [ -s $(tdir $M)/dataset/samples.tsv ] && [ -s $(lock $M) ] || { echo "dataset or lock missing for $M"; exit 3; }; done
echo "protocol v3 common medium $STAGE graph $GR start $(date); workers $W nice $NICE; uptime: $(uptime)"
mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1; }
job() { g=$1 M=$2 sp=$3 s=$4 f=$5; T=$(tdir $M); V=$T/runs_v2; L=$V/logs; mkdir -p $L
    [ "$g" = 3 ] && { echo "GPU 3 is not allowed"; return 1; }
    tok=${GR}_${sp}_nomed${RT}_fixedcos30_s${s}_cm${M}; t=${tok}_fold$f
    until [ "$(mem $g)" -lt $GPU_BUSY_MIB ]; do sleep 120; done
    CUDA_VISIBLE_DEVICES=$g nice -n $NICE $PYG script_python/81b_gnn_train_v2.py --graph $GR --readout $RO --split $sp --seed $s --fold $f --no-medium \
        --max-epochs 30 --protocol fixed_cosine --protocol-lock $(lock $M) --dataset $T/dataset --out-root $V --tag _cm$M --workers $W \
        > $L/train_$t.log 2>&1 || { echo "TRAIN FAILED $t gpu$g $(date)"; return 1; }
    CUDA_VISIBLE_DEVICES=$g nice -n $NICE $PYG script_python/100_monitor_v3.py --run $V/$tok/fold$f --workers $W > $L/monitor_$t.log 2>&1 \
        || echo "MONITOR FAILED $t gpu$g $(date)"
    echo "done $t gpu$g $(date)"; }
case $STAGE in
  dry)
    for M in $MEDIA; do T=$(tdir $M); mkdir -p $T/_dry_smoke; for sp in gene line; do t=${GR}_${sp}_cm${M}_dry_fold0
        CUDA_VISIBLE_DEVICES=${DRY_GPU:-2} nice -n $NICE $PYG script_python/81b_gnn_train_v2.py --graph $GR --readout $RO --split $sp --seed 0 --fold 0 --no-medium \
            --max-epochs 2 --min-epochs 1 --max-train-batches 150 --protocol fixed_cosine --protocol-lock none --dataset $T/dataset \
            --out-root $T/_dry_smoke --tag _cm${M}_dry --workers $W > $T/_dry_smoke/train_$t.log 2>&1 && echo "dry OK $t $(date)" || echo "DRY FAILED $t $(date)"
        grep -E "^samples|epoch_s|done" $T/_dry_smoke/train_$t.log | cut -c1-300; done; done;;
  grid)
    # chain c: GPU c/PER (PER = CHAINS/3 chains per GPU; env CHAINS 6 default, 3 for graph B: two B jobs on one GPU ran out of memory 2026-11-01 13:44)
    B=(); for s in 0 1 2; do for sp in gene line; do for M in $MEDIA; do B+=("$M $sp $s"); done; done; done
    NC=${CHAINS:-6}; PER=$((NC / 3)); for ((c=0; c<NC; c++)); do { g=$((c / PER)); for ((k=c; k<${#B[@]}; k+=NC)); do set -- ${B[k]}
        for f in 0 1 2 3 4; do job $g $1 $2 $3 $f; done; done; } & done
    wait
    for M in $MEDIA; do $PYG script_python/101_check_protocol_v3.py --lock $(lock $M) > $(tdir $M)/runs_v2/logs/check_protocol_v3_grid_$GR.log 2>&1 \
        || echo "PROTOCOL CHECK FAILED grid $M $(date)"; tail -1 $(tdir $M)/runs_v2/logs/check_protocol_v3_grid_$GR.log; done;;
  check)
    for M in $MEDIA; do $PYG script_python/101_check_protocol_v3.py --lock $(lock $M); done;;
  *) echo "unknown stage $STAGE"; exit 2;;
esac
echo "done protocol v3 common medium $STAGE $(date)"
