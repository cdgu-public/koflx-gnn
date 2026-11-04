#!/usr/bin/env bash
# Session-independent completion of script_python/45_universal_exchange_checks.py.
# Waits for the in-session run to end; reruns only if its JSON output is missing.
cd /data/workspace/cdgu/Winery/seeding/CRC_GEM
OUT=result/medium_checks_20261026
while pgrep -f "script_python/45_universal_exchange_checks.py" >/dev/null; do sleep 60; done
if [ ! -s $OUT/universal_exchange_checks.json ]; then
  echo "rerun start $(date)"
  PYTHONNOUSERSITE=1 /home/cdgu/anaconda3/envs/crcgem/bin/python script_python/45_universal_exchange_checks.py 2>&1 \
    | grep -v -i "warn\|license\|Set param\|WLS" > $OUT/universal_exchange_checks.log
fi
echo "done $(date); json: $(ls -l $OUT/universal_exchange_checks.json 2>&1)"
