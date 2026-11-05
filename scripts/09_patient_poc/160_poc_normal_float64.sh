#!/bin/bash
# Patient POC normals rerun with float64 storage (user 2026-11-03 ~18:40 "rerun" (translated); cause = verify_poc_distance_20261103.md item 1).
# Only factor vs the 152 run: storage dtype of the OptGP samples (110 --sample-dtype float64) and of the 142 knockout fluxes
# (--flux-dtype float64). Same models, assembly, caps, seeds 0 1, thinning 100, n 100, gene list, sample ranges. New output roots.
#   (1) 110 per normal (3 in parallel, 4 processes each) -> $P/optgp_f64/<n>
#   (2) 161: float32(new) == old samples? max|Sv| float64 vs float32 -> $P/optgp_f64/161_<n>.log
#   (3) 142 on the float64 seed-0 samples, 6 ranges x 3 normals in parallel -> $P/moma_normal_f64
#   (4) 158 -> $P/moma_normal_f64_141; 141 --tol 1e-7 -> $P/distance/poc_normal_optgp142_f64; 148, 159
# Usage: P=result/patient_poc_20261103 nohup nice -n 10 bash script_bash/160_poc_normal_float64.sh > result/patient_poc_20261103/160_chain.log 2>&1 &
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
: ${P:?}
PY=$HOME/anaconda3/envs/crcgem/bin/python; export PYTHONNOUSERSITE=1; A=RPMI1640_Gibco11875
say() { echo "[$(date '+%F %T')] $*"; }
die() { say "FAILED: $*"; echo "$(date '+%F %T') 160 $*" >> $P/chain_failed.txt; exit 1; }
NOR="TCGA-A6-2671-11A TCGA-A6-2675-11A TCGA-A6-2678-11A"; G=$P/genes/poc_genes_cut10.txt; D=$P/distance
[ "$(grep -vc '^#' $G)" = 18 ] || die "gene list $G is not 18 genes"
mkdir -p $P/optgp_f64
for n in $NOR; do
  $PY script_python/110_optgp_timing.py --line $n --assembly $A --processes 4 --thinning 100 --n 100 --seeds 0 1 --ko-genes \
    --threads 2 --models-root $P/models --caps-dir $P/line_requirement_caps --sample-dtype float64 --out $P/optgp_f64/$n > $P/optgp_f64/110_$n.log 2>&1 &
done; wait
for n in $NOR; do [ -s $P/optgp_f64/$n/samples_WT_thin100_seed0.parquet ] || die "110 $n: no float64 samples"
  $PY script_python/161_check_optgp_float64.py $P/models $n $P/optgp/$n/samples_WT_thin100_seed0.parquet \
    $P/optgp_f64/$n/samples_WT_thin100_seed0.parquet > $P/optgp_f64/161_$n.log 2>&1 || die "161 $n"; done
say "110 + 161 done: $(grep -h 'exactly' $P/optgp_f64/161_*.log | tr '\n' ' ')"
for n in $NOR; do for r in 0-16 17-33 34-50 51-66 67-83 84-99; do
  $PY script_python/142_moma_from_samples.py --line $n --assembly $A --models-root $P/models --caps-dir $P/line_requirement_caps \
    --samples $P/optgp_f64/$n/samples_WT_thin100_seed0.parquet --genes $G --sample-idx $r --flux-dtype float64 \
    --out $P/moma_normal_f64 --no-overwrite > $P/moma_normal_f64_${n}_$r.log 2>&1 &
done; done; wait
for n in $NOR; do nf=$(ls $P/moma_normal_f64/$n/flux_*.parquet 2>/dev/null | wc -l); [ "$nf" = 100 ] || die "142 $n: $nf of 100"; done
say "142 done"
$PY script_python/158_layout_142_for_141.py --in-root $P/moma_normal_f64 --out-root $P/moma_normal_f64_141 > $P/158_f64.log 2>&1 || die 158
$PY script_python/141_fluxome_distance.py --moma-root $P/moma_normal_f64_141 --set poc_normal_optgp142_f64 --tol 1e-7 --out-root $D \
  > $D/141_normal142_f64.log 2>&1 || die 141
$PY script_python/148_check_fluxome_distance.py $D/poc_normal_optgp142_f64 > $D/148_normal142_f64.log 2>&1 || die 148
$PY script_python/159_repro_error_by_objective.py $D/poc_normal_optgp142_f64 > $D/159_normal142_f64.log 2>&1 || die 159
say "ALL DONE"
