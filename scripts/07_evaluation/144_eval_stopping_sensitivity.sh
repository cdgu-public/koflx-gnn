#!/bin/bash
# 85 evaluation + 118 comparison of the stopping-rule sensitivity runs (2026-11-03; ruling 2026-10-31 (3); runs script_bash/133):
# symlink root holding the grid A' gene-disjoint baseline (fixed 30 epochs, cosine) and the early-stopping runs (seeds 0-2), --samples dataset_v2/samples.tsv,
# --flat-root flat_v2_expr_dsv2/fluxonly (HGB no medium, as the controls eval); 118 with baseline 'nomed_fixedcos30_tolpatch'
# only. CPU only. Refuses to overwrite. Usage: bash script_bash/144_eval_stopping_sensitivity.sh
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=""
T=$R/result/gnn_20261030_tolpatch; S=$T/runs_v2_es_eval; D=$(date +%Y%m%d_%H%M); O=$T/eval_stopping_nomedHGB_$D
[ -e $S ] || { mkdir -p $S; for s in 0 1 2; do ln -s $T/runs_v2/A_gene_nomed_fixedcos30_s${s}_tolpatch $S/
    ln -s $T/runs_v2_es/A_gene_nomed_s${s}_tolpatch_es $S/; done; }
[ -e $O ] && { echo "exists: $O"; exit 3; }
nice -n 5 $PYG script_python/85_eval_gnn_v2.py $S --samples $T/dataset_v2/samples.tsv --flat-root $T/flat_v2_expr_dsv2/fluxonly --out $O \
    > $O.log 2>&1 || { echo "EVAL FAILED $(date)"; exit 1; }
for b in nomed_fixedcos30_tolpatch; do
    $PYG script_python/118_compare_controls_all_seeds.py $O/per_fold_seed.tsv $b $O/stopping_vs_${b}_allseeds_$D.tsv >> $O/118.log 2>&1 \
        || echo "118 FAILED $b"; done
echo "done stopping eval $O $(date)"
