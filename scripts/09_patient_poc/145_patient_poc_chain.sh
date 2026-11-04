#!/usr/bin/env bash
# Patient POC chain (2026-11-03). Spec: result/external_plan_20261103/design_poc_addendum_20261103.md section 9 (b)-(f), plus the user
# scope update 2026-11-03 (coordinator relay): 104 tolpatch on all 6 models (tumor and normal), 64d dataset as the external pipeline
# (settings of result/external_20261030_tolpatch/dataset/build_condition.json), GNN prediction with the 113 loop (runs of record
# A_line_nomed_fixedcos30_s{0,1,2}_tolpatch, B_line_nomed_meanmax_fixedcos30_s{0,1,2}_tolpatch; 15 models each; GPU $GPU, never 3).
# Two branches after the context models exist:
#   N (normal reference): 110 OptGP n=100 per normal, then STOP at the 142 gate (gene list pending, user 2026-11-03 ~08:00;
#                         143 and the cut rule are not run by this chain)
#   T (pFBA reference):   wait 77b (58 on 6 models) -> 104 on 6 models ->
#                         64d -> 99 predict -> 146 ensemble
# Every step is verified (file counts / exit codes) before the next; failures go to $P/chain_failed.txt and stop that branch.
# Usage: P=result/patient_poc_20261103 EXPR=... LINEMAP=... GPU=0 nohup nice -n 10 bash script_bash/145_patient_poc_chain.sh \
#          > result/patient_poc_20261103_chain.log 2>&1 &
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
: ${P:?} ${EXPR:?} ${LINEMAP:?}; GPU=${GPU:-0}; [ "$GPU" = 3 ] && { echo "GPU 3 is not allowed"; exit 1; }
PY=/home/cdgu/anaconda3/envs/crcgem/bin/python; PYG=/home/cdgu/anaconda3/envs/crcgem_gnn/bin/python
export PYTHONNOUSERSITE=1; A=RPMI1640_Gibco11875; mkdir -p $P
say() { echo "[$(date '+%F %T')] $*"; }
die() { say "FAILED: $*"; echo "$(date '+%F %T') $*" >> $P/chain_failed.txt; exit 1; }
LINES=$(tail -n +2 $LINEMAP | cut -f1); TUM=$(echo "$LINES" | grep -- '-01[A-Z]$'); NOR=$(echo "$LINES" | grep -- '-11[A-Z]$')
[ $(echo "$TUM" | wc -l) = 3 ] && [ $(echo "$NOR" | wc -l) = 3 ] || die "expected 3 tumor + 3 normal samples in $LINEMAP"

# ---- (b) context models + 58 on all 6 ----
O=$P EXPR=$EXPR LINEMAP=$LINEMAP nice -n 10 bash script_bash/77b_run_external_context_moma.sh 6 > ${P}_77b.log 2>&1 &
B=$!; say "77b PID $B"
until [ $(ls $P/models/*/*.repaired.xml 2>/dev/null | wc -l) = 6 ] || ! kill -0 $B 2>/dev/null; do sleep 60; done
[ $(ls $P/models/*/*.repaired.xml 2>/dev/null | wc -l) = 6 ] || die "77b: 6 repaired models not produced ($(cat $P/failed.txt 2>/dev/null))"
until grep -q "caps done" ${P}_77b.log || ! kill -0 $B 2>/dev/null; do sleep 60; done
grep -q "caps done" ${P}_77b.log || die "77b: 62 caps not done"
say "context models + caps done; gapfill.json: $(cat $P/line_requirement_caps/gapfill.json | tr -d '\n' | cut -c1-300)"

branch_N() {
  # (d) 143 / gene list NOT run here: user 2026-11-03 ~08:00, drug-target genes = top 10 by drug usage, ruling pending; the
  #     coordinator writes the genes file. 142 (and the P2 tumor check, which needs genes) are GATED (see end of branch N).
  # (e) OptGP per normal, one at a time (8 cores)
  for n in $NOR; do
    $PY script_python/110_optgp_timing.py --line $n --assembly $A --processes 4 --thinning 100 --n 100 --seeds 0 1 --ko-genes \
      --threads 2 --models-root $P/models --caps-dir $P/line_requirement_caps --out $P/optgp/$n > $P/optgp_$n.log 2>&1 \
      || { echo -e "$n\t110 exit $?\t$(tail -n 2 $P/optgp_$n.log | tr '\n' ' ')" >> $P/p5_optgp_faults.tsv; say "110 FAILED $n (P5 fault logged)"; continue; }
    $PY -c "import pandas as pd,sys; s=pd.read_parquet('$P/optgp/$n/samples_WT_thin100_seed0.parquet'); sys.exit(0 if len(s)==100 else 1)" \
      || { echo -e "$n\tsamples_WT_thin100_seed0 missing or != 100 rows" >> $P/p5_optgp_faults.tsv; continue; }
    say "110 done $n: $(grep -o '"rhat_median": [0-9.na]*' $P/optgp/$n/results.json | head -2 | tr '\n' ' ')"
  done
  OK=$(for n in $NOR; do [ -s $P/optgp/$n/samples_WT_thin100_seed0.parquet ] && echo $n; done)
  [ -n "$OK" ] || die "110: no normal model has samples"
  # GATE: 142 dry test, cut rule and full run are NOT run by this chain (gene list pending, user 2026-11-03 ~08:00).
  # The removed stage (dry test, equivalence, cut rule, 6-worker full run, P2 tumor check) runs separately once the genes file exists.
  say "branch N stopped at the 142 gate (genes file pending); OptGP samples ready for: $OK"
}

branch_T() {
  wait $B 2>/dev/null; while kill -0 $B 2>/dev/null; do sleep 60; done
  for l in $LINES; do grep -q '"status": "done"' $P/moma/$l/meta.json 2>/dev/null || die "58 not done for $l"; done
  say "58 done on 6 models"
  # 104 tolpatch on all 6 (parameters of 105 run_external)
  mkdir -p ${P}_tolpatch/moma
  for l in $LINES; do
    mkdir -p ${P}_tolpatch/moma/$l
    $PY script_python/104_moma_tolpatch_line.py $l --threads 2 --set Presolve=-1 --set OptimalityTol=1e-6 --assembly $A \
      --src-root $P/moma --models-root $P/models --caps-dir $P/line_requirement_caps --out-root ${P}_tolpatch/moma \
      > ${P}_tolpatch/moma/$l/run.log 2>&1 &
  done; wait
  [ $(ls ${P}_tolpatch/moma/*/summary.tsv 2>/dev/null | wc -l) = 6 ] || die "104: not 6 summaries"
  say "104 done on 6 models"
  # 64d as the external pipeline (build_condition.json of result/external_20261030_tolpatch/dataset)
  $PY script_python/64d_gnn_build_dataset_tolpatch.py --moma-root ${P}_tolpatch/moma --out $P/dataset --line-map $LINEMAP \
    --overrides $P/line_requirement_caps/lower_bound_overrides.tsv --dependency depmap-csv \
    --scaffold-from result/gnn_20261030_tolpatch/dataset --lines $LINES > $P/64d.log 2>&1 || die "64d"
  [ -s $P/dataset/samples.tsv ] && [ $(ls $P/dataset/lines/*.npz | wc -l) = 6 ] || die "64d outputs"
  say "64d done"
  # 99 predict, the 113 loop with DS/O changed
  M=result/gnn_20261030_tolpatch/runs_v2_sel_grid; mkdir -p $P/pred_v3/logs
  for run in A_line_nomed_fixedcos30_s{0,1,2}_tolpatch B_line_nomed_meanmax_fixedcos30_s{0,1,2}_tolpatch; do
    CUDA_VISIBLE_DEVICES=$GPU nice -n 5 $PYG script_python/99_gnn_predict_external.py --model-root $M/$run --dataset $P/dataset \
      --out $P/pred_v3/$run --device cuda --workers 4 > $P/pred_v3/logs/$run.log 2>&1 || die "99 $run"
  done
  [ $(ls $P/pred_v3/*/fold*/pred_external.tsv | wc -l) = 30 ] || die "99: not 30 prediction files"
  $PY script_python/146_patient_gnn_ensemble.py --pred-root $P/pred_v3 --out $P/gnn_ensemble.tsv > $P/146.log 2>&1 || die "146"
  say "branch T done"
}

branch_N > $P/chain_branch_N.log 2>&1 &
NP=$!
branch_T > $P/chain_branch_T.log 2>&1 &
TP=$!
say "branch N PID $NP, branch T PID $TP"; wait $NP $TP
say "ALL DONE; failures: $(cat $P/chain_failed.txt 2>/dev/null | wc -l)"
