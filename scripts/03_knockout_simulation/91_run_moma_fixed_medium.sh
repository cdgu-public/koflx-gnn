#!/usr/bin/env bash
# Fixed-medium MOMA for all 59 CRC lines (user decision 2026-10-29: one medium for every line, DMEM/F-12 or RPMI-1640,
# per-line requirement caps kept (script 62), 16 lines in parallel). Scripts 58/62/64 unchanged; uses 58b and 64b.
# Steps: (1) line map copy with every assembly_id = ASSEMBLY -> (2) script 62 per-line caps into the run folder
#        (LRC_LINE_MAP, LRC_OUT) -> (3) 58b per line, --assembly ASSEMBLY --caps-dir <run>/line_requirement_caps
#        --out-root <run>, 1 Gurobi thread, N in parallel -> (4) 64b dataset into <run>/dataset (GNN_MOMA_ROOT, GNN_DATASET_OUT).
# Never overwrites: refuses to start if the run folder already has any <line>/meta.json unless RESUME=1 (then finished
# lines are skipped, as runner 77 does).
# Usage: bash script_bash/91_run_moma_fixed_medium.sh <ASSEMBLY> [N parallel=16] [LINE ...]
#   ASSEMBLY in {DMEMF12_Gibco11320, RPMI1640_Gibco11875} (any assembly_id of exchange_lower_bounds_by_assembly.tsv works)
#   SMOKE_KO=5 bash script_bash/91_run_moma_fixed_medium.sh DMEMF12_Gibco11320 1 ACH-000007   -> smoke run into <run>_smoke
# Output root: result/moma_fixed_<ASSEMBLY>_20261029/  (log run.log, per-line run.log, failed.txt, ALL DONE line)
set -u
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM
A=${1:?assembly id}; N=${2:-16}; shift; shift || true
PY58=$R/script_python/58b_moma_line_fixed.py; PY62=$R/script_python/62_line_requirement_caps.py; PY64=$R/script_python/64b_gnn_build_dataset_paths.py
PYC=$HOME/anaconda3/envs/crcgem/bin/python
grep -q "^${A}	" $R/data/medium/uptake_caps/exchange_lower_bounds_by_assembly.tsv || { echo "unknown assembly $A"; exit 1; }
O=$R/result/moma_fixed_${A}_20261029; [ -n "${SMOKE_KO:-}" ] && O=${O}_smoke
if ls $O/ACH-*/meta.json >/dev/null 2>&1 && [ "${RESUME:-0}" != 1 ]; then echo "$O already holds results; set RESUME=1 to skip finished lines"; exit 1; fi
mkdir -p $O/line_requirement_caps
export PYTHONNOUSERSITE=1 R O A PY58 PYC
echo "START $(date) assembly $A parallel $N smoke_ko ${SMOKE_KO:-0}" | tee -a $O/run.log
# (1) line map with one assembly for all lines
awk -F'\t' -v a="$A" 'BEGIN{OFS="\t"} NR==1{for(i=1;i<=NF;i++) if($i=="assembly_id") c=i; print; next} {$c=a; print}' \
    $R/data/medium/line_medium_map.tsv > $O/line_medium_map.fixed.tsv
LINES=("$@"); [ ${#LINES[@]} -eq 0 ] && mapfile -t LINES < <(tail -n +2 $R/data/medium/line_medium_map.tsv | cut -f1)
# (2) per-line requirement caps on the fixed medium (24 s/line, N processes)
if [ ! -s $O/line_requirement_caps/lower_bound_overrides.tsv ]; then
    LRC_LINE_MAP=$O/line_medium_map.fixed.tsv LRC_OUT=$O/line_requirement_caps $PYC $PY62 $N "${LINES[@]}" > $O/caps62.log 2>&1 \
        || { echo "script 62 FAILED, see $O/caps62.log" | tee -a $O/run.log; exit 1; }
fi
echo "caps done $(date)" | tee -a $O/run.log
# (3) MOMA per line, 1 thread each, N parallel; finished lines skipped
printf '%s\n' "${LINES[@]}" | xargs -P "$N" -I{} bash -c '
l={}; mkdir -p $O/$l
grep -q "\"status\"" $O/$l/meta.json 2>/dev/null && { echo "skip $l (done)"; exit 0; }
cd $R && $PYC $PY58 $l --threads 1 --assembly $A --caps-dir ${O#$R/}/line_requirement_caps --out-root ${O#$R/} \
    ${SMOKE_KO:+--max-ko $SMOKE_KO} > $O/$l/run.log 2>&1 || echo "MOMA FAILED $l $(date)" >> $O/failed.txt
echo "moma done $l $(date)" >> $O/run.log'
echo "MOMA ALL DONE $(date)" | tee -a $O/run.log
# (4) dataset (skipped in smoke runs)
if [ -z "${SMOKE_KO:-}" ]; then
    cd $R && GNN_MOMA_ROOT=$O GNN_DATASET_OUT=$O/dataset $PYC $PY64 > $O/dataset_build.log 2>&1 && echo "DATASET DONE $(date)" | tee -a $O/run.log \
        || echo "DATASET FAILED, see $O/dataset_build.log" | tee -a $O/run.log
fi
echo "ALL DONE $(date)" | tee -a $O/run.log
