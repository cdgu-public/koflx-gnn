#!/bin/bash
# 155b (2026-11-04): seeds 1-2 (SEEDS env, default "1 2"), lock protocol_lock_v3_tolpatch_medinput_B_seeds012_20261104.json; otherwise as 155.
# Graph B medium-input GNN on dataset_v2 (user approval 2026-11-03: GPU 2 only, seed 0, folds 0-4, before the B stopping-rule run).
# B gene-disjoint, 81b without --no-medium, fixed_cosine 30 epochs, readout mean_max, lock protocol_lock_v3_tolpatch_medinput_B_20261103.json
# (script_python/154), out runs_v2_med_B. Only factor vs runs_v2/B_gene_nomed_meanmax_fixedcos30_s0_tolpatch: no_medium false.
# One run at a time on GPU 2 (B peak 15.2 GB; two per GPU ran out of memory on 2026-11-01). Never GPUs 0, 1, 3.
#   dry   B gene s0 fold0, 2 epochs, 150 batches, lock check skipped, out _dry_smoke
#   grid  5 runs sequentially, then 101 lock check
# Launch: nohup bash script_bash/155_run_medium_input_B.sh grid > result/gnn_20261030_tolpatch/runs_v2_med_B/logs/run_medB_grid.log 2>&1 &
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=$R/result/gnn_20261030_tolpatch; DS=$T/dataset_v2; V=$T/runs_v2_med_B; L=$V/logs; mkdir -p $L
W=${WORKERS:-6}; NICE=${NICE:-5}; G=2; TAG=_tolpatch; LOCK=$T/protocol_lock_v3_tolpatch_medinput_B_seeds012_20261104.json
echo "medium input B $1 start $(date); dataset $DS md5 $(md5sum $DS/samples.tsv | cut -c1-32); lock $LOCK"
case $1 in
dry) mkdir -p $T/_dry_smoke; CUDA_VISIBLE_DEVICES=$G $PYG script_python/81b_gnn_train_v2.py --graph B --readout mean_max --split gene --seed 0 --fold 0 \
        --max-epochs 2 --min-epochs 1 --max-train-batches 150 --protocol fixed_cosine --protocol-lock none --dataset $DS --out-root $T/_dry_smoke \
        --tag ${TAG}_medB_dry_$(date +%H%M%S) --workers $W > $T/_dry_smoke/train_medB_dry.log 2>&1 && echo "dry OK" || echo "DRY FAILED"
    grep -E "^medium:" $T/_dry_smoke/train_medB_dry.log; tail -2 $T/_dry_smoke/train_medB_dry.log;;
grid) for s in ${SEEDS:-1 2}; do for f in 0 1 2 3 4; do t=B_gene_meanmax_fixedcos30_s${s}${TAG}_fold$f
        CUDA_VISIBLE_DEVICES=$G nice -n $NICE $PYG script_python/81b_gnn_train_v2.py --graph B --readout mean_max --split gene --seed $s --fold $f \
            --max-epochs 30 --protocol fixed_cosine --protocol-lock $LOCK --dataset $DS --out-root $V --tag $TAG --workers $W > $L/train_$t.log 2>&1 \
            && echo "done $t gpu$G $(date) $(grep -o 'PASS arm=[A-Za-z_]*\|MISMATCH.*' $L/train_$t.log | head -1)" || echo "TRAIN FAILED $t gpu$G $(date)"
    done; done; echo "grid end $(date)"
    $PYG script_python/101_check_protocol_v3.py --lock $LOCK --out-root $V > $L/check_protocol_medinputB_seeds12.log 2>&1; tail -3 $L/check_protocol_medinputB_seeds12.log;;
*) echo "usage: $0 dry | grid"; exit 2;;
esac
