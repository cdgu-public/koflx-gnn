#!/usr/bin/env bash
# Serum-rescue sensitivity (D12 option D, scope section 5; precedent Cheng et al. 2023 [ref: PMID 37205704]):
# MOMA on a few lines with cholesterol and fatty-acid uptake opened on top of the line's OWN medium and caps
# (default), or on top of a fixed medium (FIXED=<assembly>, needs the caps folder of runner 91 for that assembly).
# Exchange ids read from Human-GEM.xml with cobra (2026-10-29, scratchpad serum_exchange_ids.log):
#   cholesterol MAR09285 (MAM01450e), palmitate MAR00611, oleate MAR00650, stearate MAR00639, arachidonate MAR00568,
#   linoleate MAR09035, linolenate MAR09036 (already capped in every assembly), LDL MAR09051 / HDL MAR09050 (lumped
#   lipoprotein pools; LDL alone gives biomass 13.96 -> never opened).
# Measured (same log): as the sole organic input, each fatty acid gives ATP 1000 and cholesterol gives ATP 936.5,
# so none may be opened at -1000 (the rule that closed H2S and sulfite); they are CAPPED.
# Cap (proposal, user may override with CAPS="MAR09285:-x,..."): the assembly's linolenate requirement cap
# (basis EFA_requirement_x_mu, MAR09036 in exchange_lower_bounds_by_assembly.tsv) applied to each added species, i.e.
# a serum lipid supply on the scale of the essential-fatty-acid requirement at mu. Not a measured FBS concentration.
# Usage: bash script_bash/92_run_moma_serum_variant.sh [N parallel=5] LINE [LINE ...]
#   FIXED=DMEMF12_Gibco11320 bash script_bash/92_run_moma_serum_variant.sh 5 ACH-000007 ...   (on the fixed medium)
#   SMOKE_KO=5 ... -> output root gets _smoke
# Output root: result/moma_serum_variant_20261029/<own|FIXED>/<line>/ ; never overwrites a finished line.
set -u
R=/data/workspace/cdgu/Winery/seeding/CRC_GEM; N=${1:-5}; shift || true
[ $# -ge 1 ] || { echo "give at least one line"; exit 1; }
PY58=$R/script_python/58b_moma_line_fixed.py; PYC=/home/cdgu/anaconda3/envs/crcgem/bin/python
LBT=$R/data/medium/uptake_caps/exchange_lower_bounds_by_assembly.tsv
SUB=${FIXED:-own}; O=$R/result/moma_serum_variant_20261029/$SUB; [ -n "${SMOKE_KO:-}" ] && O=${O}_smoke
mkdir -p $O; export PYTHONNOUSERSITE=1 R O PY58 PYC LBT
echo "START $(date) mode $SUB parallel $N lines $*" | tee -a $O/run.log
printf '%s\n' "$@" | xargs -P "$N" -I{} bash -c '
l={}; mkdir -p $O/$l
grep -q "\"status\"" $O/$l/meta.json 2>/dev/null && { echo "skip $l (done)"; exit 0; }
if [ -n "${FIXED:-}" ]; then a=$FIXED; CAPDIR=result/moma_fixed_${FIXED}_20261029/line_requirement_caps; ASM="--assembly $FIXED";
else a=$(awk -F"\t" -v l=$l "NR>1 && \$1==l{for(i=1;i<=NF;i++) if(h[i]==\"assembly_id\") print \$i} NR==1{for(i=1;i<=NF;i++) h[i]=\$i}" $R/data/medium/line_medium_map.tsv);
     CAPDIR=data/medium/uptake_caps/line_requirement_caps; ASM=""; fi
[ -s $R/$CAPDIR/lower_bound_overrides.tsv ] || { echo "caps folder $CAPDIR missing (run runner 91 for $FIXED first)" >> $O/failed.txt; exit 1; }
efa=$(awk -F"\t" -v a="$a" "\$1==a && \$2==\"MAR09036\"{print \$4}" $LBT)
caps=${CAPS:-"MAR09285:$efa,MAR00611:$efa,MAR00650:$efa,MAR00639:$efa,MAR00568:$efa"}
echo "$l assembly $a open_exchanges $caps" >> $O/run.log
cd $R && $PYC $PY58 $l --threads 1 $ASM --caps-dir $CAPDIR --out-root ${O#$R/} --open-exchanges "$caps" \
    ${SMOKE_KO:+--max-ko $SMOKE_KO} > $O/$l/run.log 2>&1 || echo "MOMA FAILED $l $(date)" >> $O/failed.txt
echo "moma done $l $(date)" >> $O/run.log'
echo "ALL DONE $(date)" | tee -a $O/run.log
