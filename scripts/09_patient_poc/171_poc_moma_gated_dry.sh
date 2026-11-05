#!/bin/bash
# Dry test of script_python/171_moma_gated.py (2026-11-04): tumor TCGA-A6-2678-01A (pFBA recomputed at FeasibilityTol 1e-9) and
# normal TCGA-A6-2678-11A OptGP float64 sample 0; 18 genes; Method 0 and 1; -> $O/dry/{tumor,normal}. CPU only, nice 10, Threads 1.
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
P=result/patient_poc_20261103; O=$P/moma_gated_20261104; PY=$HOME/anaconda3/envs/crcgem/bin/python
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=
A=RPMI1640_Gibco11875; G=$P/genes/poc_genes_cut10.txt; mkdir -p $O/dry
nice -n 10 $PY script_python/171_moma_gated.py --line TCGA-A6-2678-01A --assembly $A --models-root $P/models --caps-dir $P/line_requirement_caps \
  --tumor-pfba --compare-ref $P/moma/TCGA-A6-2678-01A/reference.parquet --genes $G --out $O/dry/tumor --no-overwrite > $O/dry/171_tumor_2678.log 2>&1 &
nice -n 10 $PY script_python/171_moma_gated.py --line TCGA-A6-2678-11A --assembly $A --models-root $P/models --caps-dir $P/line_requirement_caps \
  --samples $P/optgp_f64/TCGA-A6-2678-11A/samples_WT_thin100_seed0.parquet --sample-idx 0-0 --genes $G --out $O/dry/normal --no-overwrite \
  > $O/dry/171_normal_2678_s0.log 2>&1 &
wait; echo "dry done $(date '+%F %T')"
