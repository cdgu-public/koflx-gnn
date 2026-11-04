#!/bin/bash
# 85 evaluation + 118 comparison (--graph B) of the stopping-rule BMRG runs (early stopping on validation loss), seeds 0-2 (2026-11-04; runs script_bash/157 seed 0 and 157b seeds 1-2).
# Same as script_bash/150b with the stopping-rule dirs (runs_v2_es_B) instead of runs_v2_med_B:
# symlink root holding runs_v2/B_gene_nomed_meanmax_fixedcos30_s{0,1,2}_tolpatch and runs_v2_es_B/B_gene_nomed_meanmax_s{0,1,2}_tolpatch_es,
# --samples dataset_v2/samples.tsv, --flat-root flat_v2_expr_dsv2/fluxonly; 118 baseline 'nomed_meanmax_fixedcos30_tolpatch' --graph B.
# CPU only. Refuses to overwrite. Usage: bash script_bash/150b_eval_medium_input_B.sh
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=""
T=$R/result/gnn_20261030_tolpatch; S=$T/runs_v2_esB_eval_seeds012; D=$(date +%Y%m%d_%H%M); O=$T/eval_stoppingB_nomedHGB_seeds012_$D
[ -e $S ] || { mkdir -p $S; for s in 0 1 2; do ln -s $T/runs_v2/B_gene_nomed_meanmax_fixedcos30_s${s}_tolpatch $S/
    ln -s $T/runs_v2_es_B/B_gene_nomed_meanmax_s${s}_tolpatch_es $S/; done; }
[ -e $O ] && { echo "exists: $O"; exit 3; }
nice -n 5 $PYG script_python/85_eval_gnn_v2.py $S --samples $T/dataset_v2/samples.tsv --flat-root $T/flat_v2_expr_dsv2/fluxonly --out $O \
    > $O.log 2>&1 || { echo "EVAL FAILED $(date)"; exit 1; }
b=nomed_meanmax_fixedcos30_tolpatch
$PYG script_python/118_compare_controls_all_seeds.py $O/per_fold_seed.tsv $b $O/stoppingB_vs_${b}_allseeds_$D.tsv --graph B >> $O/118.log 2>&1 \
    || echo "118 FAILED $b"
echo "done stopping-rule B eval $O $(date)"
