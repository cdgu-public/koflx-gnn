#!/bin/bash
# Patient POC: tumor models with OptGP references (user 2026-11-04 ~11:4x "환자 Cancer 도 OptGP로 돌려서 데이터 같이 보여줘").
# Only factor changed vs the normal side of 160 + 172 + 173: the model (TCGA-A6-{2671,2675,2678}-01A instead of -11A).
# Same assembly RPMI1640_Gibco11875, caps, 110 args (processes 4, thinning 100, n 100, seeds 0 1, threads 2, float64),
# same 18-gene list, 171 on seed-0 samples in ranges of 2 (FeasibilityTol 1e-9, both checks at their defaults), 141 --tol 1e-7.
# Parallelism 36 instead of 24 (171 Threads 1 each; does not change values).
# NOTE on labels: 172/173/174 use the directory/side name "normal" for the OptGP-sample layout; in this run those rows are TUMOR
# models (-01A) with OptGP references. Written to NOTE_side_label.txt in every output root.
# Outputs (new roots, nothing existing touched): $P/optgp_f64_tumor/<t>, $O/normal/<t> (171), $O/normal_141,
#   $D/poc_tumor_optgp_gated_20261104/, $D/poc_percentile_tumor_optgp_gated_20261104.tsv (tumor pFBA rows = existing run, for comparison)
# Usage: nohup bash script_bash/175_poc_tumor_optgp_gated.sh > result/patient_poc_20261103/tumor_optgp_gated_20261104/chain.log 2>&1 &
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
export P=result/patient_poc_20261103; export O=$P/tumor_optgp_gated_20261104; D=$P/distance
export PY=/home/cdgu/anaconda3/envs/crcgem/bin/python PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= A=RPMI1640_Gibco11875
export G=$P/genes/poc_genes_cut10.txt
TUM="TCGA-A6-2671-01A TCGA-A6-2675-01A TCGA-A6-2678-01A"
say() { echo "[$(date '+%F %T')] $*"; }
die() { say "FAILED: $*"; exit 1; }
[ "$(grep -vc '^#' $G)" = 18 ] || die "gene list not 18"
mkdir -p $O/logs $P/optgp_f64_tumor
NOTE="side label 'normal' in this root = OptGP-sample layout; models here are tumor (-01A) with OptGP references (175, 2026-11-04)"
echo "$NOTE" > $O/NOTE_side_label.txt
# (1) OptGP, 3 tumors in parallel
for t in $TUM; do
  nice -n 10 $PY script_python/110_optgp_timing.py --line $t --assembly $A --processes 4 --thinning 100 --n 100 --seeds 0 1 --ko-genes \
    --threads 2 --models-root $P/models --caps-dir $P/line_requirement_caps --sample-dtype float64 --out $P/optgp_f64_tumor/$t \
    > $P/optgp_f64_tumor/110_$t.log 2>&1 &
done; wait
for t in $TUM; do [ -s $P/optgp_f64_tumor/$t/samples_WT_thin100_seed0.parquet ] || die "110 $t: no samples"; done
say "110 done"
# (2) 171 on seed-0 samples
job() {
  nice -n 10 $PY script_python/171_moma_gated.py --line $1 --assembly $A --models-root $P/models --caps-dir $P/line_requirement_caps \
    --samples $P/optgp_f64_tumor/$1/samples_WT_thin100_seed0.parquet --sample-idx $2 --genes $G --out $O/normal --no-overwrite \
    > $O/logs/171_$1_$2.log 2>&1
  echo "[$(date '+%F %T')] job $1 $2 exit $?"
}
export -f job
job TCGA-A6-2678-01A 0-1 | grep -q "exit 0" || die "171 dry range 2678-01A 0-1"
say "171 dry range ok"
{ for t in $TUM; do for s in $(seq 0 2 98); do [ "$t $s" = "TCGA-A6-2678-01A 0" ] || echo "$t $s-$((s+1))"; done; done; } \
  | xargs -P 36 -L 1 bash -c 'job "$0" "$1"'
say "171 all ranges finished"
# (3) post: checker, layout, 141, 148, 174 (tumor pFBA side = existing gated run)
nice -n 10 $PY script_python/172_check_moma_gated.py $O > $O/172.log 2>&1 || die 172
nice -n 10 $PY script_python/173_layout_171_for_141.py $O/normal $O/normal_141 normal > $O/173.log 2>&1 || die 173
echo "$NOTE" > $O/normal_141/NOTE_side_label.txt
nm=poc_tumor_optgp_gated_20261104
nice -n 10 $PY script_python/141_fluxome_distance.py --moma-root $O/normal_141 --set $nm --tol 1e-7 --out-root $D > $D/141_$nm.log 2>&1
grep -q Traceback $D/141_$nm.log && die 141
echo "$NOTE" > $D/$nm/NOTE_side_label.txt
nice -n 10 $PY script_python/148_check_fluxome_distance.py $D/$nm > $D/148_$nm.log 2>&1 || say "148 FAILED"
nice -n 10 $PY script_python/174_percentile_gated.py --moma-tumor $P/moma_gated_20261104/tumor_141 --moma-normal $O/normal_141 \
  --dist-tumor $D/poc_tumor_gated_20261104 --dist-normal $D/$nm \
  --out $D/poc_percentile_tumor_optgp_gated_20261104.tsv > $D/174_percentile_tumor_optgp_gated_20261104.log 2>&1 || die 174
say "ALL DONE"
