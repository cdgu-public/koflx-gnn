#!/bin/bash
# Everything derived from the MOMA screens, rebuilt from the solver-tolerance-patched roots (user decision 2026-10-30 13:18 KST, option A),
# into NEW folders; nothing under result/gnn_20261028 or result/external_20261030 is written.
# Steps (each refuses to overwrite; a step whose output exists is skipped, so the script can be re-run after a failure):
#   1 patch summary tables           104b -> result/moma_20261030_tolpatch/patch_summary_*.tsv and result/oob_20261030_tolpatch/moma/patch_summary_*.tsv
#   2 colorectal dataset             64d  -> result/gnn_20261030_tolpatch/dataset (scaffold copied from result/gnn_20261028/dataset)
#   3 dataset diff old vs new        106  -> result/gnn_20261030_tolpatch/dataset_diff (sample set, splits, strata)
#   4 external dataset               64d  -> result/external_20261030_tolpatch/dataset (scaffold copied from the new colorectal dataset)
#   5 gradient-boosting baseline     86   -> result/gnn_20261030_tolpatch/flat_v2 (with medium; growth probe)       [CPU, OMP 8]
#   6 same without medium            90_flat_expression_v2 --variant fluxonly -> result/gnn_20261030_tolpatch/flat_v2_expr/fluxonly  [CPU, OMP 8]
#       (rel_nn_dist strata inside its meta.json come from --rel-run A_gene_nomed_fixedcos30_s0_tolpatch and are absent until the grid has run; 85 does not need them)
#   7 external comparators           86b  -> result/external_20261030_tolpatch/flat (refit on the new colorectal dataset, applied to the new external dataset)
#   8 GNN dry test                   107 dry (one short 81b job on GPU 0, lock check skipped)
# Usage: bash script_bash/109_rebuild_derived_tolpatch.sh [steps, default "1 2 3 4 5 6 7 8"]
# Launch: nohup bash script_bash/109_rebuild_derived_tolpatch.sh > result/gnn_20261030_tolpatch/rebuild_derived.log 2>&1 &
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
PYG=~/anaconda3/envs/crcgem_gnn/bin/python; export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
T=result/gnn_20261030_tolpatch; DSN=${DSN:-dataset_v2}; SFX=${SFX:-_dsv2}; EXC=${EXC:-$T/exclude_nonoptimal_20261030.tsv}; DS=$T/$DSN; XT=result/external_20261030_tolpatch; mkdir -p $T $XT   # DSN/SFX/EXC (2026-10-30 17:05 ruling): dataset_v2 = dataset minus the 14 excluded knockouts; derived folders carry SFX
STEPS=${1:-"1 2 3 4 5 6 7 8"}
echo "rebuild derived (tolpatch) start $(date); steps [$STEPS]; uptime: $(uptime)"
has() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }
n_done() { ls $1/*/summary.tsv 2>/dev/null | wc -l; }
if has 1; then
    [ "$(n_done result/moma_20261030_tolpatch)" = 59 ] || { echo "STOP: colorectal patch incomplete ($(n_done result/moma_20261030_tolpatch)/59)"; exit 3; }
    [ "$(n_done result/oob_20261030_tolpatch/moma)" = 7 ] || { echo "STOP: external patch incomplete ($(n_done result/oob_20261030_tolpatch/moma)/7)"; exit 3; }
    $PYG script_python/104b_summarize_tolpatch.py --root result/moma_20261030_tolpatch --expect 59 > result/moma_20261030_tolpatch/patch_summary.log 2>&1 && echo "step 1 crc summary ok" || echo "STEP 1 FAILED (crc)"
    $PYG script_python/104b_summarize_tolpatch.py --root result/oob_20261030_tolpatch/moma --expect 7 > result/oob_20261030_tolpatch/moma/patch_summary.log 2>&1 && echo "step 1 external summary ok" || echo "STEP 1 FAILED (external)"
fi
if has 2; then
    if [ -s $DS/samples.tsv ]; then echo "step 2 skip (dataset exists)"; else
    $PYG script_python/64d_gnn_build_dataset_tolpatch.py --moma-root result/moma_20261030_tolpatch --out $DS ${EXC:+--exclude $EXC} > $T/${DSN}_build.log 2>&1 && echo "step 2 ok: $(tail -1 $T/${DSN}_build.log | cut -c1-150)" || { echo "STEP 2 FAILED"; exit 4; }; fi
fi
if has 3; then
    if [ -s $T/dataset_diff$SFX/dataset_diff_summary.tsv ]; then echo "step 3 skip"; else
    $PYG script_python/106_compare_datasets_tolpatch.py --old result/gnn_20261028/dataset --new $DS --out $T/dataset_diff$SFX > $T/dataset_diff$SFX.log 2>&1 && echo "step 3 ok" || echo "STEP 3 FAILED"; fi
fi
if has 4; then
    if [ -s $XT/dataset/samples.tsv ]; then echo "step 4 skip"; else
    $PYG script_python/64d_gnn_build_dataset_tolpatch.py --moma-root result/oob_20261030_tolpatch/moma --out $XT/dataset --line-map data/depmap/oob_20261028/oob_lines.tsv \
        --overrides result/oob_20261028/line_requirement_caps/lower_bound_overrides.tsv --dependency depmap-csv --scaffold-from $DS > $XT/dataset_build.log 2>&1 \
        && echo "step 4 ok: $(tail -1 $XT/dataset_build.log | cut -c1-150)" || echo "STEP 4 FAILED"; fi
fi
if has 5; then
    if [ -s $T/flat_v2$SFX/line/fold4.tsv ]; then echo "step 5 skip"; else
    OMP_NUM_THREADS=8 nice -n 5 $PYG script_python/86_flat_baselines_v2.py --dataset $DS --out $T/flat_v2$SFX > $T/flat_v2$SFX.log 2>&1 && echo "step 5 ok" || echo "STEP 5 FAILED"; fi
fi
if has 6; then
    if [ -d $T/flat_v2_expr$SFX/fluxonly ]; then echo "step 6 skip"; else
    NS=$($PYG -c "
import numpy as np, os; D='$DS'; n=0
for f in sorted(os.listdir(D+'/lines')):
    z=np.load(D+'/lines/'+f); th=1e-6/float(z['mu_ref']); V=np.where(np.abs(z['V'])>th,z['V'],0); vw=np.where(np.abs(z['vwt'])>th,z['vwt'],0); n+=int((np.abs(V-vw).max(1)>th).sum())
print(n)")
    echo "step 6: fluxome-change sample count $NS"
    OMP_NUM_THREADS=8 nice -n 5 $PYG script_python/90_flat_expression_v2.py --variant fluxonly --dataset $DS --out $T/flat_v2_expr$SFX --n-samples $NS \
        --rel-root $T/runs_v2 --rel-run A_gene_nomed_fixedcos30_s0_tolpatch > $T/flat_v2_expr$SFX.log 2>&1 && echo "step 6 ok" || echo "STEP 6 FAILED"; fi
fi
if has 7; then
    if [ -s $XT/flat$SFX/line/fold4.tsv ]; then echo "step 7 skip"; else
    OMP_NUM_THREADS=8 nice -n 5 $PYG script_python/86b_flat_baselines_external.py --train-dataset $DS --external-dataset $XT/dataset --out $XT/flat$SFX > $XT/flat${SFX}_build.log 2>&1 && echo "step 7 ok" || echo "STEP 7 FAILED"; fi
fi
if has 8; then
    bash script_bash/107_run_protocol_v3_tolpatch.sh dry
fi
echo "rebuild derived (tolpatch) done $(date)"
