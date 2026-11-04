#!/bin/bash
# MOMA under a common medium: after the solver-tolerance patch of result/moma_fixed_<A>_20261029 (script_bash/105 fixed, 104) into
# result/moma_fixed_<A>_20261030_tolpatch/, apply the non-optimal exclusion rule (user ruling 2026-10-30 17:05, as dataset_v2), build the
# dataset, run the no-medium gradient-boosting readout proposed in review_gem.md 9.2.5 (90_flat_expression_v2 --variant fluxonly), then the
# pre-specified MOMA-level comparison (111) against result/gnn_20261030_tolpatch/dataset_v2. New folders only.
#   per medium A: result/gnn_fixed_<A>_20261030_tolpatch/{exclude_nonoptimal.tsv, dataset/, flat_v2_expr/fluxonly/, *.log}
#   comparison:   result/common_medium_20261030/
# Usage: bash script_bash/112_common_medium_tolpatch_chain.sh [steps "1 2 3 4"]   (1 summaries + exclude files, 2 datasets, 3 flat fluxonly, 4 comparison 111)
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
STEPS=${1:-"1 2 3 4"}; MEDIA="DMEMF12_Gibco11320 RPMI1640_Gibco11875"
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }
echo "common-medium chain start $(date); steps [$STEPS]"
for A in $MEDIA; do
    M=result/moma_fixed_${A}_20261030_tolpatch; G=result/gnn_fixed_${A}_20261030_tolpatch; mkdir -p $G
    n=$(ls $M/*/summary.tsv 2>/dev/null | wc -l); [ "$n" = 59 ] || { echo "STOP: $A patch incomplete ($n/59)"; exit 3; }
    if has 1; then
        [ -s $M/patch_summary_total.tsv ] || $PYG script_python/104b_summarize_tolpatch.py --root $M --expect 59 > $M/patch_summary.log 2>&1 && echo "step 1 $A summary ok"
        if [ ! -s $G/exclude_nonoptimal.tsv ]; then
            { echo "# knockouts excluded from the GNN input of the common-medium dataset ($A): trace rows whose patched re-solve (script 104, FeasibilityTol 1e-8, Presolve -1 then 0) was non-optimal and whose shipped row was kept (summary.tsv patch_status kept_shipped_resolve_*); rule = dataset_v2 rule (user ruling 2026-10-30 17:05: ambiguous -> drop); date $(date '+%Y-%m-%d %H:%M')"
              echo -e "line\tgene\treason"
              for f in $M/ACH-*/summary.tsv; do l=$(basename $(dirname $f)); grep -v "^#" $f | awk -F'\t' -v l=$l 'NR>1 && $9 ~ /kept/ {print l"\t"$1"\tpatch re-solve non-optimal ("$9"); numerical failure of the QP under the patch condition; excluded by the dataset_v2 rule"}'; done; } > $G/exclude_nonoptimal.tsv
            echo "step 1 $A exclude file: $(grep -vc '^#\|^line' $G/exclude_nonoptimal.tsv) knockouts"
        fi
    fi
    if has 2; then
        if [ -s $G/dataset/samples.tsv ]; then echo "step 2 $A skip"; else
        $PYG script_python/64d_gnn_build_dataset_tolpatch.py --moma-root $M --out $G/dataset --line-map result/moma_fixed_${A}_20261029/line_medium_map.fixed.tsv \
            --overrides result/moma_fixed_${A}_20261029/line_requirement_caps/lower_bound_overrides.tsv --exclude $G/exclude_nonoptimal.tsv > $G/dataset_build.log 2>&1 \
            && echo "step 2 $A ok: $(tail -1 $G/dataset_build.log | cut -c1-160)" || { echo "STEP 2 $A FAILED"; exit 4; }; fi
    fi
    if has 3; then
        if [ -d $G/flat_v2_expr/fluxonly ]; then echo "step 3 $A skip"; else
        NS=$($PYG -c "
import numpy as np, os; D='$G/dataset'; n=0
for f in sorted(os.listdir(D+'/lines')):
    z=np.load(D+'/lines/'+f); th=1e-6/float(z['mu_ref']); V=np.where(np.abs(z['V'])>th,z['V'],0); vw=np.where(np.abs(z['vwt'])>th,z['vwt'],0); n+=int((np.abs(V-vw).max(1)>th).sum())
print(n)")
        echo "step 3 $A: fluxome-change sample count $NS"
        OMP_NUM_THREADS=8 nice -n 5 $PYG script_python/90_flat_expression_v2.py --variant fluxonly --dataset $G/dataset --out $G/flat_v2_expr --n-samples $NS \
            --rel-root $G/none --rel-run none > $G/flat_v2_expr.log 2>&1 && echo "step 3 $A ok" || echo "STEP 3 $A FAILED"; fi
    fi
done
if has 4; then
    $PYG script_python/111_compare_common_medium.py --own result/gnn_20261030_tolpatch/dataset_v2 --own-moma result/moma_20261030_tolpatch \
        --common result/gnn_fixed_DMEMF12_Gibco11320_20261030_tolpatch/dataset result/gnn_fixed_RPMI1640_Gibco11875_20261030_tolpatch/dataset \
        --common-tag DMEMF12_Gibco11320 RPMI1640_Gibco11875 \
        --common-moma result/moma_fixed_DMEMF12_Gibco11320_20261030_tolpatch result/moma_fixed_RPMI1640_Gibco11875_20261030_tolpatch \
        --out result/common_medium_20261030 > result/common_medium_20261030.log 2>&1 && echo "step 4 ok" || echo "STEP 4 FAILED"
    tail -4 result/common_medium_20261030.log | cut -c1-250
fi
echo "common-medium chain done $(date)"
