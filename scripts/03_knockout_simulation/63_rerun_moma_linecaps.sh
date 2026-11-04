#!/bin/bash
# Rerun MOMA (script 58) for the lines whose caps changed under the per-line requirement rule (script 62).
# Usage: bash script_bash/63_rerun_moma_linecaps.sh <N parallel> <line>...   Threads 1 per line.
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; N=$1; shift
printf '%s\n' "$@" | xargs -P "$N" -I{} bash -c "mkdir -p $R/result/moma_20261027/{}; cd $R; PYTHONNOUSERSITE=1 ~/anaconda3/envs/crcgem/bin/python script_python/58_moma_line.py {} --threads 1 > result/moma_20261027/{}/run.log 2>&1 || echo 'MOMA FAILED {}' >> result/moma_20261027/failed.txt"
echo "ALL DONE $(date)"
