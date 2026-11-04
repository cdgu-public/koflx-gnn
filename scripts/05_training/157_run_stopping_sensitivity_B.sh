#!/bin/bash
# Graph B stopping-rule sensitivity on dataset_v2 (user approval 2026-11-03: GPU 2 only, seed 0, folds 0-4, after the B medium-input run).
# Same superseded early-stopping rule as 133 (81b --protocol es_valbce: min 10, patience 8, max 50, best-validation checkpoint, constant lr).
# Only factor vs runs_v2/B_gene_nomed_meanmax_fixedcos30_s0_tolpatch: the protocol fields. B readout mean_max, --no-medium, gene split.
# One run at a time on GPU 2 (B peak 15.2 GB). Never GPUs 0, 1, 3. Out runs_v2_es_B.
#   dry   B gene s0 fold0, 2 epochs, 150 batches, out _dry_smoke
#   grid  [PID ...] wait, dry, then 5 runs sequentially
# Launch: nohup bash script_bash/157_run_stopping_sensitivity_B.sh grid <PID> > result/gnn_20261030_tolpatch/runs_v2_es_B/logs/run_esB_grid.log 2>&1 &
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$R/result/gnn_20261030_tolpatch; DS=$T/dataset_v2; V=$T/runs_v2_es_B; L=$V/logs; mkdir -p $L
W=${WORKERS:-6}; NICE=${NICE:-5}; G=2; TAG=_tolpatch_es
ES="--protocol es_valbce --min-epochs 10 --patience 8 --max-epochs 50 --protocol-lock none"
dry() { mkdir -p $T/_dry_smoke; CUDA_VISIBLE_DEVICES=$G $PYG script_python/81b_gnn_train_v2.py --graph B --readout mean_max --split gene --seed 0 --fold 0 \
        --no-medium --protocol es_valbce --min-epochs 1 --max-epochs 2 --max-train-batches 150 --protocol-lock none --dataset $DS \
        --out-root $T/_dry_smoke --tag ${TAG}_B_dry_$(date +%H%M%S) --workers $W > $T/_dry_smoke/train_esB_dry.log 2>&1; }
STAGE=$1; shift
echo "stopping sensitivity B $STAGE start $(date); dataset $DS md5 $(md5sum $DS/samples.tsv | cut -c1-32); $ES"
case $STAGE in
dry) dry && echo "dry OK" || echo "DRY FAILED"; tail -2 $T/_dry_smoke/train_esB_dry.log;;
grid) for p in "$@"; do while kill -0 $p 2>/dev/null; do sleep 120; done; echo "waited PID $p $(date)"; done
    dry && echo "dry OK $(date)" || { echo "DRY FAILED $(date)"; exit 1; }
    for f in 0 1 2 3 4; do t=B_gene_nomed_meanmax_s0${TAG}_fold$f
        CUDA_VISIBLE_DEVICES=$G nice -n $NICE $PYG script_python/81b_gnn_train_v2.py --graph B --readout mean_max --split gene --seed 0 --fold $f \
            --no-medium $ES --dataset $DS --out-root $V --tag $TAG --workers $W > $L/train_$t.log 2>&1 \
            && echo "done $t gpu$G $(date)" || echo "TRAIN FAILED $t gpu$G $(date)"
    done; echo "grid end $(date)";;
*) echo "usage: $0 dry | grid [PID ...]"; exit 2;;
esac
