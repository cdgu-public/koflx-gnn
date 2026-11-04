#!/usr/bin/env bash
# Design result/external_plan_20261103/design_external_gi_normal_20261103.md section 6(c): solver-tolerance patch (script 104) of the
# 2026-11-03 external MOMA screens; body of 105 run_external with the roots as variables. Same solver settings as 105
# (FeasibilityTol 1e-8 default of 104, Presolve -1, OptimalityTol 1e-6, 2 threads).
# Sets (each line: src root <set>/moma, models <set>/models, caps <set>/line_requirement_caps, assembly from <set>/lines_assembly.tsv):
#   result/external_gi_normal_20261103   (6 GI-tract cancer cell lines + Hs 729)
#   result/external_normal_bph1_20261103 (BPH-1, RPMI1640_Gibco11875 proxy, user ruling 2026-11-03 01:3x)
# All patched cell lines go to ONE root OUT (default result/external_gi_normal_20261103_tolpatch/moma), so one dataset and one evaluation cover them.
# No overwrite: a cell line whose OUT/<line>/summary.tsv exists is skipped (104 refuses to overwrite). Failures -> OUT/failed.txt. CPU only, nice 10.
# Usage: bash script_bash/105b_run_external_tolpatch.sh [N parallel]
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R
export PYTHONNOUSERSITE=1 R
P=/home/cdgu/anaconda3/envs/crcgem/bin/python; export P
N=${1:-6}; T=${THREADS:-2}; export T
OUT=${OUT:-result/external_gi_normal_20261103_tolpatch/moma}; export OUT
SETS=${SETS:-"result/external_gi_normal_20261103 result/external_normal_bph1_20261103"}
EXTRA=${EXTRA:-"Presolve=-1 OptimalityTol=1e-6"}; SETARGS=""; for kv in $EXTRA; do SETARGS="$SETARGS --set $kv"; done; export SETARGS
mkdir -p $R/$OUT
echo "105b tolpatch start $(date); parallel $N threads $T; extra [$EXTRA]; sets [$SETS]; out $OUT; uptime: $(uptime)"
for S in $SETS; do awk -v s=$S 'NF>=2 && $1 ~ /^ACH-/ {print $1, $2, s}' $R/$S/lines_assembly.tsv; done |
xargs -P $N -L1 bash -c '
    l=$0; a=$1; s=$2
    [ -s $R/$OUT/$l/summary.tsv ] && { echo "skip $l (done)"; exit 0; }
    mkdir -p $R/$OUT/$l
    nice -n 10 $P $R/script_python/104_moma_tolpatch_line.py $l --threads $T $SETARGS --assembly $a --src-root $s/moma \
        --models-root $s/models --caps-dir $s/line_requirement_caps --out-root $OUT > $R/$OUT/$l/run.log 2>&1 \
        || echo "PATCH FAILED $l $(date)" | tee -a $R/$OUT/failed.txt
    echo "done $l ($s) $(date): $(tail -1 $R/$OUT/$l/run.log | cut -c1-200)"'
echo "105b finished $(date); summaries: $(ls $R/$OUT/*/summary.tsv 2>/dev/null | wc -l)"
