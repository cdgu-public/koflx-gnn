#!/bin/bash
# Waits for the graph B common-medium grid runner (116 GRAPH=B, PID $1) to exit, then runs 119 (85 eval per medium; covers A and B
# runs in runs_v2). Ruled 2026-11-01 (graph B on both common media) and 2026-11-02 ("run experiments at once"). CPU only.
# Usage: nohup setsid bash script_bash/127_eval_after_grid_B.sh <runner PID> > result/common_medium_20261030/eval_after_B_20261102.log 2>&1 &
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; cd $R; L=$R/result/common_medium_20261030/gnn_grid_B_20261101.log
while kill -0 $1 2>/dev/null; do sleep 300; done
echo "runner exited $(date); done B folds $(grep -c '^done B_' $L); FAILED $(grep -c FAILED $L)"; tail -3 $L
grep -q FAILED $L && { echo "grid had failures, eval not started"; exit 1; }
bash script_bash/119_eval_common_medium.sh
