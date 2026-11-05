#!/bin/bash
# Post-processing of the gated MOMA run (2026-11-04): 172 checker -> 173 layout for 141 -> 141 --tol 1e-7 -> 148, 159 -> 174 percentile.
# Uses finished 171 ranges only. Argument $1 = suffix for every output name ('' for the complete run, e.g. _partial_HHMM at a deadline).
# Outputs: $O/gates_per_model$1.tsv, $O/{tumor,normal}_141$1/, $P/distance/poc_{tumor,normal}_gated_20261104$1/,
#          $P/distance/poc_percentile_gated_20261104$1.tsv. Nothing existing is overwritten (173/141 refuse or suffix).
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
P=result/patient_poc_20261103; O=$P/moma_gated_20261104; D=$P/distance; SFX=$1
PY=$HOME/anaconda3/envs/crcgem/bin/python; export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=
say() { echo "[$(date '+%F %T')] $*"; }
nice -n 10 $PY script_python/172_check_moma_gated.py $O --tag "$SFX" > $O/172$SFX.log 2>&1 || { say "172 FAILED"; exit 1; }; say "172 ok"
for s in tumor normal; do
  nice -n 10 $PY script_python/173_layout_171_for_141.py $O/$s $O/${s}_141$SFX $s > $O/173_$s$SFX.log 2>&1 || { say "173 $s FAILED"; exit 1; }
done; say "173 ok"
for s in tumor normal; do
  nm=poc_${s}_gated_20261104$SFX
  nice -n 10 $PY script_python/141_fluxome_distance.py --moma-root $O/${s}_141$SFX --set $nm --tol 1e-7 --out-root $D > $D/141_$nm.log 2>&1 &
done; wait
for s in tumor normal; do
  nm=poc_${s}_gated_20261104$SFX
  grep -q Traceback $D/141_$nm.log && { say "141 $s FAILED"; exit 1; }
  nice -n 10 $PY script_python/148_check_fluxome_distance.py $D/$nm > $D/148_$nm.log 2>&1 || say "148 FAILED $nm"
  nice -n 10 $PY script_python/159_repro_error_by_objective.py $D/$nm > $D/159_$nm.log 2>&1 || say "159 FAILED $nm"
done; say "141/148/159 done"
nice -n 10 $PY script_python/174_percentile_gated.py --moma-tumor $O/tumor_141$SFX --moma-normal $O/normal_141$SFX \
  --dist-tumor $D/poc_tumor_gated_20261104$SFX --dist-normal $D/poc_normal_gated_20261104$SFX \
  --out $D/poc_percentile_gated_20261104$SFX.tsv > $D/174_percentile_gated_20261104$SFX.log 2>&1 || { say "174 FAILED"; exit 1; }
say "post done"
