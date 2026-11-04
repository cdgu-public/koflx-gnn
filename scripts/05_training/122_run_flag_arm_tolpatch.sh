#!/bin/bash
# Baseline-plus-flag control arm on the solver-tolerance-patched dataset (user ruling 2026-11-01 "바로 시작"; design
# result/gnn_20261030_tolpatch/design_flag_arm_20261101.md). New file; 107 / 108 and their runs are untouched.
# Same rule as the 108 controls: 90_gnn_e2e3_v2.py --protocol fixed_cosine, fixed 30 epochs, cosine 1e-3 -> 1e-5, last-epoch checkpoint,
# --no-medium, graph A', gene-disjoint split, seeds 0-2, folds 0-4 = 15 runs, dataset result/gnn_20261030_tolpatch/dataset_v2, tag _tolpatch.
#   disflag  90 --disabled-flag -> runs_v2_disflag/A_gene_nomed_disflag_fixedcos30_s<s>_tolpatch/fold<f>
# Lock: result/gnn_20261030_tolpatch/protocol_lock_v3_tolpatch_disflag_20261101.json (script_python/121; the 2026-10-30 lock plus this arm).
# Comparators (evaluated afterwards, 85 on a symlink root + 118): grid arm A_gene_nomed_fixedcos30_s<s>_tolpatch and A_gene_nomed_wtonly_... in runs_v2.
# One chain per GPU 0, 1, 2 (CHAINS_PER_GPU, default 1; NEVER GPU 3) pulls from one seed-major queue; before each job the chain waits until
# its GPU has < $GPU_BUSY_MIB MiB in use (default 17000: admits one A' job (~3-7 GB) beside one graph B common-medium job (~16.2 GB) and
# refuses when a second large job is present; orchestrator constraint 2026-11-01 13:50 after two B jobs per GPU ran out of memory at 13:44).
# Env: WORKERS (default 6), NICE (default 5), SEEDS (default "0 1 2"), FOLDS (default "0 1 2 3 4"), GPU_BUSY_MIB (default 17000), GPUS (default "0 1 2"),
#      CHAINS_PER_GPU (default 1).
# Usage: bash script_bash/122_run_flag_arm_tolpatch.sh run|check
# Launch: nohup bash script_bash/122_run_flag_arm_tolpatch.sh run > result/gnn_20261030_tolpatch/runs_v2_disflag/logs/run_flag_arm.log 2>&1 &
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$R/result/gnn_20261030_tolpatch; DS=$T/dataset_v2; V=$T/runs_v2_disflag; L=$V/logs; mkdir -p $L
W=${WORKERS:-6}; NICE=${NICE:-5}; LOCK=$T/protocol_lock_v3_tolpatch_disflag_20261101.json; E=30; TAG=_tolpatch
SEEDS=${SEEDS:-"0 1 2"}; FOLDS=${FOLDS:-"0 1 2 3 4"}; GPU_BUSY_MIB=${GPU_BUSY_MIB:-17000}; GPUS=(${GPUS:-0 1 2}); CPG=${CHAINS_PER_GPU:-${PER_GPU:-1}}
for g in "${GPUS[@]}"; do [ "$g" = 3 ] && { echo "GPU 3 is not allowed"; exit 2; }; done
STAGE=$1; [ -z "$STAGE" ] && { echo "usage: $0 run|check"; exit 2; }
[ "$STAGE" = check ] && { $PYG script_python/101_check_protocol_v3.py --lock $LOCK; exit $?; }
[ "$STAGE" != run ] && { echo "unknown stage $STAGE"; exit 2; }
[ -s $DS/samples.tsv ] || { echo "dataset missing: $DS/samples.tsv"; exit 3; }
[ -s $LOCK ] || { echo "lock missing: $LOCK (run script_python/121_make_lock_flag_arm.py)"; exit 3; }
echo "flag arm start $(date); dataset $DS (samples.tsv md5 $(md5sum $DS/samples.tsv | cut -c1-32)); workers $W nice $NICE; seeds [$SEEDS] folds [$FOLDS]; gpus [${GPUS[*]}] chains per gpu $CPG busy threshold $GPU_BUSY_MIB MiB; lock $LOCK; uptime: $(uptime)"
mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1; }
JOBS=(); for s in $SEEDS; do for f in $FOLDS; do JOBS+=("$s $f"); done; done
N=${#JOBS[@]}; echo "$N jobs queued"
Q=$L/.queue_$$; echo 0 > $Q
next() { ( flock -x 9; i=$(cat $Q); echo $((i + 1)) > $Q; echo $i; ) 9> $Q.lock; }
wait_gpu() { until [ "$(mem $1)" -lt $GPU_BUSY_MIB ]; do sleep 120; done; }
job() { g=$1 s=$2 f=$3
    tok=A_gene_nomed_disflag_fixedcos${E}_s$s$TAG; t=${tok}_fold$f
    [ -e $V/$tok/fold$f ] && { echo "SKIP exists $t"; return 0; }
    wait_gpu $g
    CUDA_VISIBLE_DEVICES=$g nice -n $NICE $PYG script_python/90_gnn_e2e3_v2.py --graph A --split gene --seed $s --fold $f --no-medium --disabled-flag \
        --workers $W --out-root $V --max-epochs $E --protocol fixed_cosine --protocol-lock $LOCK --dataset $DS --tag $TAG > $L/train_$t.log 2>&1 \
        || { echo "TRAIN FAILED $t gpu$g $(date)"; return 1; }
    CUDA_VISIBLE_DEVICES=$g nice -n $NICE $PYG script_python/100_monitor_v3.py --run $V/$tok/fold$f --workers $W > $L/monitor_$t.log 2>&1 \
        || echo "MONITOR FAILED $t gpu$g $(date)"
    echo "done $t gpu$g $(date)"; }
NG=${#GPUS[@]}
worker() { g=${GPUS[$(( $1 % NG ))]}
    while :; do i=$(next); [ "$i" -ge "$N" ] && break; job $g ${JOBS[$i]}; done; }
for ((w = 0; w < CPG * NG; w++)); do worker $w & done   # CHAINS_PER_GPU (alias PER_GPU), default 1 beside the graph B grid (2026-11-01)
wait
rm -f $Q $Q.lock
$PYG script_python/101_check_protocol_v3.py --lock $LOCK > $L/check_protocol_v3_disflag.log 2>&1 || echo "PROTOCOL CHECK FAILED disflag $(date)"
tail -1 $L/check_protocol_v3_disflag.log
echo "done flag arm $(date)"
