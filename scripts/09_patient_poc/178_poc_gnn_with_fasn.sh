#!/bin/bash
# Patient POC GNN predictions with the FASN knockout of TCGA-A6-2678-11A re-solved (user 2026-11-04 "확인해서 다시 넣어야").
# The 58 MOMA row was 'unbounded'; script 104 --genes re-solved it on the same shipped reference at FeasibilityTol 1e-8
# (result/patient_poc_20261103_fasn_resolve_20261104), merged by 177 into result/patient_poc_20261103_tolpatch_fasn/moma.
# Then the branch T steps of 145 unchanged: 64d (same line map, overrides, scaffold) -> 99 on the 6 model runs -> 146.
# New outputs only: $P/dataset_fasn, $P/pred_v3_fasn, $P/gnn_ensemble_fasn.tsv.
# Usage: GPU=2 nohup bash script_bash/178_poc_gnn_with_fasn.sh > result/patient_poc_20261103/178_chain.log 2>&1 &
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
P=result/patient_poc_20261103; GPU=${GPU:-2}
PY=/home/cdgu/anaconda3/envs/crcgem/bin/python; PYG=/home/cdgu/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1
LINEMAP=data/tcga_xena_20261103/processed_pcU_20261103/lines.tsv
LINES=$(tail -n +2 $LINEMAP | cut -f1)
say() { echo "[$(date '+%F %T')] $*"; }
die() { say "FAILED: $*"; exit 1; }
[ $(echo "$LINES" | wc -l) = 6 ] || die "expected 6 samples in $LINEMAP"
$PY script_python/64d_gnn_build_dataset_tolpatch.py --moma-root ${P}_tolpatch_fasn/moma --out $P/dataset_fasn --line-map $LINEMAP \
  --overrides $P/line_requirement_caps/lower_bound_overrides.tsv --dependency depmap-csv \
  --scaffold-from result/gnn_20261030_tolpatch/dataset --lines $LINES > $P/64d_fasn.log 2>&1 || die 64d
[ $(ls $P/dataset_fasn/lines/*.npz | wc -l) = 6 ] || die "64d outputs"
say "64d done"
M=result/gnn_20261030_tolpatch/runs_v2_sel_grid; mkdir -p $P/pred_v3_fasn/logs
for run in A_line_nomed_fixedcos30_s{0,1,2}_tolpatch B_line_nomed_meanmax_fixedcos30_s{0,1,2}_tolpatch; do
  CUDA_VISIBLE_DEVICES=$GPU nice -n 5 $PYG script_python/99_gnn_predict_external.py --model-root $M/$run --dataset $P/dataset_fasn \
    --out $P/pred_v3_fasn/$run --device cuda --workers 4 > $P/pred_v3_fasn/logs/$run.log 2>&1 || die "99 $run"
done
[ $(ls $P/pred_v3_fasn/*/fold*/pred_external.tsv | wc -l) = 30 ] || die "99: not 30 prediction files"
$PY script_python/146_patient_gnn_ensemble.py --pred-root $P/pred_v3_fasn --out $P/gnn_ensemble_fasn.tsv > $P/146_fasn.log 2>&1 || die 146
say "ALL DONE"
