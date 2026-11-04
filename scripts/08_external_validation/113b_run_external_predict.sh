#!/bin/bash
# Design result/external_plan_20261103/design_external_gi_normal_20261103.md section 6(f): script 113 with the dataset and output
# folder as variables. Models of record (user ruling): cell-line-disjoint tolpatch grid A_line_nomed_fixedcos30_s{0,1,2}_tolpatch and
# B_line_nomed_meanmax_fixedcos30_s{0,1,2}_tolpatch (5 folds each, result/gnn_20261030_tolpatch/runs_v2_sel_grid). Inference only (script 99).
# Defaults: DS=result/external_gi_normal_20261103/dataset, O=result/external_gi_normal_20261103/pred_v3 (99 refuses to overwrite).
# One GPU, never GPU 3. Usage: [DS=... O=...] bash script_bash/113b_run_external_predict.sh <gpu> [smoke]
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
G=${1:?gpu}; [ "$G" = 3 ] && { echo "GPU 3 is not allowed"; exit 1; }
case "$G" in 0|1|2) ;; *) echo "GPU must be 0, 1 or 2"; exit 1;; esac
DS=${DS:-result/external_gi_normal_20261103/dataset}; M=result/gnn_20261030_tolpatch/runs_v2_sel_grid
O=${O:-result/external_gi_normal_20261103/pred_v3}; [ "$2" = smoke ] && { O=${O}_smoke; SUB="--subset 20"; }
RUNS="A_line_nomed_fixedcos30_s0_tolpatch A_line_nomed_fixedcos30_s1_tolpatch A_line_nomed_fixedcos30_s2_tolpatch
B_line_nomed_meanmax_fixedcos30_s0_tolpatch B_line_nomed_meanmax_fixedcos30_s1_tolpatch B_line_nomed_meanmax_fixedcos30_s2_tolpatch"
mkdir -p $O/logs
echo "113b start $(date); dataset $DS; models $M; out $O; gpu $G"
for run in $RUNS; do
    [ -d $M/$run ] || { echo "MISSING model run $run"; exit 1; }
    CUDA_VISIBLE_DEVICES=$G nice -n 5 $PYG script_python/99_gnn_predict_external.py --model-root $M/$run --dataset $DS \
        --out $O/$run --device cuda --workers 4 $SUB > $O/logs/$run.log 2>&1 || echo "PREDICT FAILED $run $(date)"
    echo "done $run $(date)"
done
echo "done external predict $(date); files: $(ls $O/*/fold*/pred_external.tsv 2>/dev/null | wc -l) (expected 30)"
