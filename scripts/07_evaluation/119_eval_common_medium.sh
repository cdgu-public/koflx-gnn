#!/bin/bash
# 85 evaluation of the protocol v3 common-medium grid per medium (design result/common_medium_20261030/gnn_design_20261031.md (3a), line 86).
#   85 <T>/runs_v2 --samples <T>/dataset/samples.tsv --flat-root <T>/flat_v2_expr/fluxonly --out <T>/eval_v2_nomedHGB_<date>
# CPU only; both media in parallel. Usage: bash script_bash/119_eval_common_medium.sh
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=""
D=$(date +%Y%m%d_%H%M)
for M in DMEMF12_Gibco11320 RPMI1640_Gibco11875; do T=$R/result/gnn_fixed_${M}_20261030_tolpatch; O=$T/eval_v2_nomedHGB_$D
    [ -e $O ] && { echo "exists: $O"; exit 3; }
    nice -n 5 $PYG script_python/85_eval_gnn_v2.py $T/runs_v2 --samples $T/dataset/samples.tsv --flat-root $T/flat_v2_expr/fluxonly --out $O \
        > $T/eval_v2_nomedHGB_$D.log 2>&1 && echo "done eval $M $O $(date)" || echo "EVAL FAILED $M $(date)" &
done
wait; echo "done common medium eval $(date)"
