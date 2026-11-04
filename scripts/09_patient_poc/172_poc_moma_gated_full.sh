#!/bin/bash
# Full gated MOMA run (script_python/171_moma_gated.py), 2026-11-04, after the dry test (script_bash/171_poc_moma_gated_dry.sh).
# Jobs in priority order (user ruling via coordinator 07:3x): tumors TCGA-A6-{2671,2675,2678}-01A (pFBA recomputed at 1e-9), then normal
# OptGP float64 samples 0-99 of 2678-11A, 2671-11A, 2675-11A in ranges of 2 samples; 24 jobs in parallel, Threads 1 each, nice 10, CPU only.
# Output $O/{tumor,normal}/<model>/, one log per job in $O/logs/. Afterwards script_bash/173_poc_gated_post.sh (checker, layout, 141, 148, 159, 174).
# Usage: nohup bash script_bash/172_poc_moma_gated_full.sh > result/patient_poc_20261103/moma_gated_20261104/full_chain.log 2>&1 &
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
export P=result/patient_poc_20261103 O=result/patient_poc_20261103/moma_gated_20261104 PY=/home/cdgu/anaconda3/envs/crcgem/bin/python
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= A=RPMI1640_Gibco11875 G=result/patient_poc_20261103/genes/poc_genes_cut10.txt
mkdir -p $O/logs
[ "$(grep -vc '^#' $G)" = 18 ] || { echo "gene list not 18"; exit 1; }
job() {   # $1 = model, $2 = range or 'tumor'
  if [ "$2" = tumor ]; then
    nice -n 10 $PY script_python/171_moma_gated.py --line $1 --assembly $A --models-root $P/models --caps-dir $P/line_requirement_caps \
      --tumor-pfba --compare-ref $P/moma/$1/reference.parquet --genes $G --out $O/tumor --no-overwrite > $O/logs/171_$1.log 2>&1
  else
    nice -n 10 $PY script_python/171_moma_gated.py --line $1 --assembly $A --models-root $P/models --caps-dir $P/line_requirement_caps \
      --samples $P/optgp_f64/$1/samples_WT_thin100_seed0.parquet --sample-idx $2 --genes $G --out $O/normal --no-overwrite > $O/logs/171_$1_$2.log 2>&1
  fi
  echo "[$(date '+%F %T')] job $1 $2 exit $?"
}
export -f job
{ for t in TCGA-A6-2671-01A TCGA-A6-2675-01A TCGA-A6-2678-01A; do echo "$t tumor"; done
  for n in TCGA-A6-2678-11A TCGA-A6-2671-11A TCGA-A6-2675-11A; do for s in $(seq 0 2 98); do echo "$n $s-$((s+1))"; done; done
} | xargs -P 24 -L 1 bash -c 'job "$0" "$1"'
echo "[$(date '+%F %T')] all 171 jobs finished"
