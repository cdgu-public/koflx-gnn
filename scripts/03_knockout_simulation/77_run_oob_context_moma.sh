#!/usr/bin/env bash
# Out-of-bag / POC lines (script 76; user 2026-10-28), the CRC pipeline unchanged, in separate result folders:
# tINIT (22; localT2 thresholds of the 59 CRC lines) -> essential-task repair (56) -> per-line requirement caps
# at mu (62, own output folder) -> pFBA + MOMA (58).
# Usage: bash script_bash/77_run_oob_context_moma.sh [N parallel]
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM
O=$R/result/oob_20261028; mkdir -p $O/models $O/moma $O/line_requirement_caps
export PYTHONNOUSERSITE=1 R O
tail -n +2 $R/data/depmap/oob_20261028/oob_lines.tsv | cut -f1,7 > $O/lines_assembly.tsv
xargs -P "${1:-7}" -L1 bash -c '
l=$0; d=$O/models/$l; mkdir -p $d; cd $d; printf "Threads 2\n" > gurobi.env
[ -s $d/$l.xml ] || /home/cdgu/anaconda3/envs/crcgem_extraction/bin/python $R/script_python/22_tinit_localT2.py \
  --expression $R/data/depmap/oob_20261028/oob_expression.ENSG.parquet \
  --threshold-expression $R/data/depmap/crc_trainset/crc_expression.ENSG.parquet \
  --scaffold $R/data/scaffold/Human-GEM.xml --solver gurobi --lines $l --out $d > $d/run.log 2>&1 || { echo "TINIT FAILED $l" >> $O/failed.txt; exit 0; }
cd $R
[ -s $d/$l.repaired.xml ] || /home/cdgu/anaconda3/envs/crcgem/bin/python script_python/56_task_check_repair.py $l $O/models > $d/repair.log 2>&1 || echo "REPAIR FAILED $l" >> $O/failed.txt
echo "repaired $l $(date)"
' < $O/lines_assembly.tsv
cd $R
LRC_OUT=$O/line_requirement_caps LRC_MODELS=$O/models LRC_LINE_MAP=$R/data/depmap/oob_20261028/oob_lines.tsv \
  /home/cdgu/anaconda3/envs/crcgem/bin/python script_python/62_line_requirement_caps.py 7 $(cut -f1 $O/lines_assembly.tsv) > $O/line_requirement_caps/run.log 2>&1 || { echo "CAPS FAILED" >> $O/failed.txt; exit 1; }
echo "caps done $(date)"
xargs -P "${1:-7}" -L1 bash -c '
l=$0; a=$1; mkdir -p $O/moma/$l
grep -q "\"status\"" $O/moma/$l/meta.json 2>/dev/null || /home/cdgu/anaconda3/envs/crcgem/bin/python script_python/58_moma_line.py $l --threads 2 \
  --models-root result/oob_20261028/models --out-root result/oob_20261028/moma --assembly $a \
  --caps-dir result/oob_20261028/line_requirement_caps > $O/moma/$l/run.log 2>&1 || echo "MOMA FAILED $l" >> $O/failed.txt
echo "moma done $l $(date)"
' < $O/lines_assembly.tsv
echo "ALL DONE $(date)"
