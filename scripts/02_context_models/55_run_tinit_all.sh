#!/usr/bin/env bash
# tINIT (script 22: localT2 25/75; biomass protected) for every line, one process per line, each in
# its own folder, run from that folder so a gurobi.env there limits Gurobi to 2 threads (30 processes
# x 48 threads oversubscribed the host on 2026-10-27 07:2x). Essential tasks are checked and repaired
# afterwards (script 56), not forced inside the MILP (no incumbent after 500 s with 1,429 forced).
# Skips lines whose SBML exists. Usage: <N parallel>
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM
O=$R/result/tinit_localT2_20261027
export PYTHONNOUSERSITE=1
xargs -P "${1:-22}" -I{} bash -c 'd='"$O"'/models/{}; [ -s $d/{}.xml ] && exit 0; mkdir -p $d; cd $d; printf "Threads 2\n" > gurobi.env; $HOME/anaconda3/envs/crcgem_extraction/bin/python '"$R"'/script_python/22_tinit_localT2.py --expression '"$R"'/data/depmap/crc_trainset/crc_expression.ENSG.parquet --scaffold '"$R"'/data/scaffold/Human-GEM.xml --solver gurobi --lines {} --out $d > $d/run.log 2>&1' < "$O/lines.txt"
echo "ALL DONE $(date); models: $(ls $O/models/*/*.xml 2>/dev/null | wc -l)"
