#!/usr/bin/env bash
# Per line: wait for its tINIT SBML (script 55), task check/repair (script 56), pFBA + MOMA (script 58).
# Idempotent: skips steps whose outputs exist. Usage: <N parallel>
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM
export PYTHONNOUSERSITE=1 R
P=/home/cdgu/anaconda3/envs/crcgem/bin/python
export P
xargs -P "${1:-20}" -I{} bash -c '
d=$R/result/tinit_localT2_20261027/models/{}; o=$R/result/moma_20261027/{}; mkdir -p $o
while [ ! -s $d/{}.xml ]; do sleep 30; done
[ -s $d/{}.repaired.xml ] || $P $R/script_python/56_task_check_repair.py {} > $d/repair.log 2>&1
[ -s $d/{}.repaired.xml ] || { echo "REPAIR FAILED {}" >> $R/result/moma_20261027/failed.txt; exit 0; }
grep -q "\"status\"" $o/meta.json 2>/dev/null || $P $R/script_python/58_moma_line.py {} > $o/run.log 2>&1 || echo "MOMA FAILED {}" >> $R/result/moma_20261027/failed.txt
' < $R/result/tinit_localT2_20261027/lines.txt
echo "ALL DONE $(date); meta: $(ls $R/result/moma_20261027/*/meta.json 2>/dev/null | wc -l)"
