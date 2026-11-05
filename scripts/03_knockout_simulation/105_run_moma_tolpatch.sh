#!/usr/bin/env bash
# Solver-tolerance patch of the MOMA screens (script_python/104_moma_tolpatch_line.py; user decision 2026-10-30 13:18 KST, option A).
# Two sets, each into its own NEW root (production roots result/moma_20261027 and result/oob_20261028/moma are read only):
#   crc       59 done colorectal cell lines of result/moma_20261027      -> result/moma_20261030_tolpatch/<line>/
#   external  7 cell lines of result/oob_20261028/moma (assembly from result/oob_20261028/lines_assembly.tsv)
#                                                                        -> result/oob_20261030_tolpatch/moma/<line>/
# Idempotent: a cell line whose <out>/<line>/summary.tsv exists is skipped (104 refuses to overwrite). Failures -> <root>/failed.txt.
# CPU only. Host 2026-10-30 13:20: 24 fixed-medium MOMA jobs (58b, 1 thread each) running, load ~27/48 -> default 8 parallel x 2 threads.
# Usage: bash script_bash/105_run_moma_tolpatch.sh crc|external|all [N parallel]
# Launch: nohup bash script_bash/105_run_moma_tolpatch.sh all 8 > result/moma_20261030_tolpatch.launch.log 2>&1 &
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
export PYTHONNOUSERSITE=1 R
P=$HOME/anaconda3/envs/crcgem/bin/python; export P
SET=${1:-all}; N=${2:-8}; T=${THREADS:-2}; export T
# Solver conditions of the re-solves (measured 2026-10-30 13:30-13:45, ACH-000651): the production path is FeasibilityTol 1e-7,
# OptimalityTol 1e-7, Presolve 0, Method 0 (cobra 0.29.1 / optlang defaults, NOT Gurobi's 1e-6 / Presolve -1); at FeasibilityTol 1e-8
# with Presolve 0 the first trace QP had not finished after 4 min (killed). The re-solves therefore use the condition under which
# review_tolerance.md measured the stable calls: FeasibilityTol 1e-8 with Gurobi automatic presolve (Presolve -1) and OptimalityTol 1e-6
# (verifier DEFAULTS, 01_rebuild_qp.py:82), every other parameter as on the production path. Override with EXTRA (space-separated PARAM=VALUE).
EXTRA=${EXTRA:-"Presolve=-1 OptimalityTol=1e-6"}; SETARGS=""; for kv in $EXTRA; do SETARGS="$SETARGS --set $kv"; done; export SETARGS
echo "tolpatch $SET start $(date); parallel $N threads $T; extra solver params [$EXTRA]; uptime: $(uptime)"
run_crc() {
    O=$R/result/moma_20261030_tolpatch; mkdir -p $O; export O
    for d in $R/result/moma_20261027/ACH-*/; do l=$(basename $d)
        $P -c "import json,sys; sys.exit(0 if json.load(open('$d/meta.json')).get('status')=='done' else 1)" 2>/dev/null && echo $l; done |
    xargs -P $N -I{} bash -c '
        [ -s $O/{}/summary.tsv ] && { echo "skip {} (done)"; exit 0; }
        mkdir -p $O/{}
        $P $R/script_python/104_moma_tolpatch_line.py {} --threads $T $SETARGS > $O/{}/run.log 2>&1 || echo "PATCH FAILED {} $(date)" | tee -a $O/failed.txt
        echo "done {} $(date): $(tail -1 $O/{}/run.log | cut -c1-200)"'
    echo "crc set finished $(date); summaries: $(ls $O/*/summary.tsv 2>/dev/null | wc -l)/59"
}
run_external() {
    O=$R/result/oob_20261030_tolpatch/moma; mkdir -p $O; export O
    xargs -P $N -L1 bash -c '
        l=$0; a=$1
        [ -s $O/$l/summary.tsv ] && { echo "skip $l (done)"; exit 0; }
        mkdir -p $O/$l
        $P $R/script_python/104_moma_tolpatch_line.py $l --threads $T $SETARGS --assembly $a --src-root result/oob_20261028/moma \
            --models-root result/oob_20261028/models --caps-dir result/oob_20261028/line_requirement_caps \
            --out-root result/oob_20261030_tolpatch/moma > $O/$l/run.log 2>&1 || echo "PATCH FAILED $l $(date)" | tee -a $O/failed.txt
        echo "done $l $(date): $(tail -1 $O/$l/run.log | cut -c1-200)"' < $R/result/oob_20261028/lines_assembly.tsv
    echo "external set finished $(date); summaries: $(ls $O/*/summary.tsv 2>/dev/null | wc -l)/7"
}
redo_pre_fallback() {
    # 2026-10-30 14:05: 104 gained a Presolve-0 retry for solver errors and non-optimal statuses (fallback_policy in meta.json counts).
    # Cell lines patched by an earlier version that had >= 1 non-optimal trace re-solve are redone (folder moved to _superseded_pre_fallback/).
    for O in $R/result/moma_20261030_tolpatch $R/result/oob_20261030_tolpatch/moma; do
        for d in $O/ACH-*/; do l=$(basename $d); [ -s $d/summary.tsv ] || continue
            $P -c "
import json,sys; c=json.load(open('$d/meta.json'))['tolpatch']['counts']
sys.exit(0 if (c.get('fallback_policy')!='presolve0_retry_on_error_or_nonoptimal_then_keep_shipped' and c.get('n_trace_nonoptimal_new',0)>0) else 1)" \
            && { mkdir -p $O/_superseded_pre_fallback; mv $d $O/_superseded_pre_fallback/${l}_$(date +%H%M%S); echo "redo $l (pre-fallback version, non-optimal re-solves)"; }
        done
    done
}
run_fixed() {
    # MOMA under a common medium (script 58b runs, result/moma_fixed_<A>_20261029): same patch into result/moma_fixed_<A>_20261030_tolpatch/
    A=${ASSEMBLY:?set ASSEMBLY=DMEMF12_Gibco11320 or RPMI1640_Gibco11875}; SRC=result/moma_fixed_${A}_20261029; O=$R/result/moma_fixed_${A}_20261030_tolpatch; mkdir -p $O; export O A SRC
    for d in $R/$SRC/ACH-*/; do l=$(basename $d)
        $P -c "import json,sys; sys.exit(0 if json.load(open('$d/meta.json')).get('status')=='done' else 1)" 2>/dev/null && echo $l; done |
    xargs -P $N -I{} bash -c '
        [ -s $O/{}/summary.tsv ] && { echo "skip {} (done)"; exit 0; }
        mkdir -p $O/{}
        $P $R/script_python/104_moma_tolpatch_line.py {} --threads $T $SETARGS --assembly $A --src-root $SRC --caps-dir $SRC/line_requirement_caps \
            --out-root result/moma_fixed_${A}_20261030_tolpatch > $O/{}/run.log 2>&1 || echo "PATCH FAILED {} $(date)" | tee -a $O/failed.txt
        echo "done {} $(date): $(tail -1 $O/{}/run.log | cut -c1-200)"'
    echo "fixed $A set finished $(date); summaries: $(ls $O/*/summary.tsv 2>/dev/null | wc -l)/59"
}
case $SET in
  crc) run_crc;;
  external) run_external;;
  fixed) run_fixed;;
  redo) redo_pre_fallback; run_crc; run_external;;
  all) run_crc; run_external;;
  *) echo "usage: $0 crc|external|all [N]"; exit 2;;
esac
echo "tolpatch $SET ALL DONE $(date)"
