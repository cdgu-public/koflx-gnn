#!/usr/bin/env bash
# Further external lines (2026-11-03; copy of 77 with EXPR/LINEMAP/O variables; design result/external_plan_20261103 section 6(b)). CRC pipeline unchanged:
# tINIT (22; localT2 thresholds of the 59 CRC lines) -> essential-task repair (56) -> per-line requirement caps
# at mu (62, own output folder) -> pFBA + MOMA (58).
# Usage: O=<result dir> EXPR=<expression.ENSG.parquet> LINEMAP=<lines.tsv> bash script_bash/77b_run_external_context_moma.sh [N parallel]
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM
: ${O:?} ${EXPR:?} ${LINEMAP:?}; mkdir -p $O/models $O/moma $O/line_requirement_caps
export PYTHONNOUSERSITE=1 R O EXPR LINEMAP
tail -n +2 $LINEMAP | awk -F"\t" '{print $1"\t"$8}' > $O/lines_assembly.tsv
xargs -P "${1:-7}" -L1 bash -c '
l=$0; d=$O/models/$l; mkdir -p $d; cd $d; printf "Threads 2\n" > gurobi.env
[ -s $d/$l.xml ] || $HOME/anaconda3/envs/crcgem_extraction/bin/python $R/script_python/22_tinit_localT2.py \
  --expression $EXPR \
  --threshold-expression $R/data/depmap/crc_trainset/crc_expression.ENSG.parquet \
  --scaffold $R/data/scaffold/Human-GEM.xml --solver gurobi --lines $l --out $d > $d/run.log 2>&1 || { echo "TINIT FAILED $l" >> $O/failed.txt; exit 0; }
cd $R
[ -s $d/$l.repaired.xml ] || $HOME/anaconda3/envs/crcgem/bin/python script_python/56_task_check_repair.py $l $O/models > $d/repair.log 2>&1 || echo "REPAIR FAILED $l" >> $O/failed.txt
echo "repaired $l $(date)"
' < $O/lines_assembly.tsv
cd $R
LRC_OUT=$O/line_requirement_caps LRC_MODELS=$O/models LRC_LINE_MAP=$LINEMAP \
  $HOME/anaconda3/envs/crcgem/bin/python script_python/62_line_requirement_caps.py 7 $(cut -f1 $O/lines_assembly.tsv) > $O/line_requirement_caps/run.log 2>&1 || { echo "CAPS FAILED" >> $O/failed.txt; exit 1; }
echo "caps done $(date)"
xargs -P "${1:-7}" -L1 bash -c '
l=$0; a=$1; mkdir -p $O/moma/$l
grep -q "\"status\"" $O/moma/$l/meta.json 2>/dev/null || $HOME/anaconda3/envs/crcgem/bin/python script_python/58_moma_line.py $l --threads 2 \
  --models-root $O/models --out-root $O/moma --assembly $a \
  --caps-dir $O/line_requirement_caps > $O/moma/$l/run.log 2>&1 || echo "MOMA FAILED $l" >> $O/failed.txt
echo "moma done $l $(date)"
' < $O/lines_assembly.tsv
echo "ALL DONE $(date)"
