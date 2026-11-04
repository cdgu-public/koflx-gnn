#!/bin/bash
# External validation set of non-colorectal cell lines (approved 2026-10-30), predicted with the protocol v3 tolpatch
# cell-line-disjoint grid models (A_line / B_line x seeds 0-2 x folds 0-4; runs in result/gnn_20261030_tolpatch/runs_v2_sel_grid).
# Inference only (script 99), dataset result/external_20261030_tolpatch/dataset. One GPU, never GPU 3.
# Output: result/external_20261030_tolpatch/pred_v3/<run>/fold<k>/pred_external.tsv (script 99 refuses to overwrite).
# Usage: 113_run_external_predict_tolpatch.sh <gpu> [smoke]
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
G=${1:?gpu}; [ "$G" = 3 ] && { echo "GPU 3 is not allowed"; exit 1; }
DS=result/external_20261030_tolpatch/dataset; M=result/gnn_20261030_tolpatch/runs_v2_sel_grid
O=result/external_20261030_tolpatch/pred_v3; [ "$2" = smoke ] && { O=${O}_smoke; SUB="--subset 20"; }
mkdir -p $O/logs
for run in $(ls $M | grep -E '^[AB]_line_'); do
    CUDA_VISIBLE_DEVICES=$G nice -n 5 $PYG script_python/99_gnn_predict_external.py --model-root $M/$run --dataset $DS \
        --out $O/$run --device cuda --workers 4 $SUB > $O/logs/$run.log 2>&1 || echo "PREDICT FAILED $run $(date)"
    echo "done $run $(date)"
done
echo "done external predict $(date)"
