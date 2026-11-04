#!/bin/bash
# Stopping-rule sensitivity on dataset_v2 (2026-11-02; user ruling 2026-10-31 (3): Methods + supplementary table, not abstract).
# One factor varies against the protocol v3 grid A' gene-disjoint arm (runs_v2/A_gene_nomed_fixedcos30_s{0,1,2}_tolpatch):
#   superseded early stopping on validation BCE (81b --protocol es_valbce: min 10, patience 8, max 50, best-validation checkpoint).
# Same dataset, seeds 0-2, folds 0-4, graph A (default readout), --no-medium, lr/optimizer/batch defaults. New out root runs_v2_es
# (81b refuses existing fold folders). GPUs 0-2 only (never 3), PER_GPU chains per GPU.
#   dry   A gene s0 fold0, 2 epochs, 150 batches, out _dry_smoke
#   grid  15 runs
# Launch: nohup bash script_bash/133_run_stopping_sensitivity.sh grid > result/gnn_20261030_tolpatch/runs_v2_es/logs/run_es_grid.log 2>&1 &
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$R/result/gnn_20261030_tolpatch; DS=$T/dataset_v2; V=$T/runs_v2_es; L=$V/logs; mkdir -p $L
W=${WORKERS:-6}; NICE=${NICE:-5}; PER_GPU=${PER_GPU:-2}; TAG=_tolpatch_es
ES="--protocol es_valbce --min-epochs 10 --patience 8 --max-epochs 50 --protocol-lock none"
echo "stopping sensitivity $1 start $(date); dataset $DS md5 $(md5sum $DS/samples.tsv | cut -c1-32); $ES"
job() { g=$1 s=$2 f=$3; t=A_gene_nomed_s${s}${TAG}_fold$f
    CUDA_VISIBLE_DEVICES=$g nice -n $NICE $PYG script_python/81b_gnn_train_v2.py --graph A --split gene --seed $s --fold $f --no-medium \
        $ES --dataset $DS --out-root $V --tag $TAG --workers $W > $L/train_$t.log 2>&1 && echo "done $t gpu$g $(date)" || echo "TRAIN FAILED $t gpu$g $(date)"; }
case $1 in
dry) mkdir -p $T/_dry_smoke; CUDA_VISIBLE_DEVICES=0 $PYG script_python/81b_gnn_train_v2.py --graph A --split gene --seed 0 --fold 0 --no-medium \
        --protocol es_valbce --min-epochs 1 --max-epochs 2 --max-train-batches 150 --protocol-lock none --dataset $DS --out-root $T/_dry_smoke \
        --tag ${TAG}_dry_$(date +%H%M%S) --workers $W > $T/_dry_smoke/train_es_dry.log 2>&1 && echo "dry OK" || echo "DRY FAILED"; tail -3 $T/_dry_smoke/train_es_dry.log;;
grid) jobs=(); for s in 0 1 2; do for f in 0 1 2 3 4; do jobs+=("$s $f"); done; done
    nch=$((3 * PER_GPU))
    for c in $(seq 0 $((nch - 1))); do ( g=$((c % 3)); for k in $(seq $c $nch $((${#jobs[@]} - 1))); do job $g ${jobs[$k]}; done ) & done
    wait; echo "grid end $(date)";;
*) echo "usage: $0 dry|grid"; exit 2;;
esac
