#!/bin/bash
# 85 evaluation + 118 comparisons of the disabled-reaction flag arm (design result/gnn_20261030_tolpatch/design_flag_arm_20261101.md,
# lines 55-58): symlink root holding baseline, WT-only and disflag runs (seeds 0-2), --samples dataset_v2/samples.tsv,
# --flat-root flat_v2_expr_dsv2/fluxonly (HGB no medium, as the controls eval); 118 with baseline 'nomed_fixedcos30_tolpatch'
# and with 'nomed_wtonly_fixedcos30_tolpatch'. CPU only. Refuses to overwrite. Usage: bash script_bash/123_eval_flag_arm.sh
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=""
T=$R/result/gnn_20261030_tolpatch; S=$T/runs_v2_flag_eval; D=$(date +%Y%m%d_%H%M); O=$T/eval_flag_nomedHGB_$D
[ -e $S ] || { mkdir -p $S; for s in 0 1 2; do for v in nomed nomed_wtonly; do ln -s $T/runs_v2/A_gene_${v}_fixedcos30_s${s}_tolpatch $S/; done
    ln -s $T/runs_v2_disflag/A_gene_nomed_disflag_fixedcos30_s${s}_tolpatch $S/; done; }
[ -e $O ] && { echo "exists: $O"; exit 3; }
nice -n 5 $PYG script_python/85_eval_gnn_v2.py $S --samples $T/dataset_v2/samples.tsv --flat-root $T/flat_v2_expr_dsv2/fluxonly --out $O \
    > $O.log 2>&1 || { echo "EVAL FAILED $(date)"; exit 1; }
for b in nomed_fixedcos30_tolpatch nomed_wtonly_fixedcos30_tolpatch; do
    $PYG script_python/118_compare_controls_all_seeds.py $O/per_fold_seed.tsv $b $O/flag_vs_${b}_allseeds_$D.tsv >> $O/118.log 2>&1 \
        || echo "118 FAILED $b"; done
echo "done flag eval $O $(date)"
