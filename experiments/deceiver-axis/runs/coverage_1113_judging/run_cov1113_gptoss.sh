#!/bin/bash
# Post-judging analysis for the gpt-oss set gfx02 = Koby's coverage arms of 2026-11-13 (Goodfire floors gf_floor_p50 / p75 / p95,
# 12 seeds each; PLAN "Koby's coverage batch of 2026-11-13: judging"): run_gfar_analysis.sh unchanged (family = every judged
# Goodfire arm in verdicts_gf*/ + the three random add arms, October unsteered arm), then the dose-rule read as for gfx01.
set -e
N=/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis; P=$N/gi_v2/pilot; TAG=gfx02
PY=/home/jansen05/self-concept-experiments/.venv-313/bin/python; S=experiments/deceiver-axis/scripts
bash $N/gfar_vs_gi_v2/steer/run_gfar_analysis.sh gptoss $TAG $TAG gf_floor_p50 gf_floor_p75 gf_floor_p95
cd /home/jansen05/self-concept-experiments; export OPENSSL_CONF=/dev/null
FAMILY=$(ls $P/analysis_root_gfar | grep '^gf_' | tr '\n' ' ')
echo "== dose rule (hidden pass over all episodes, no code): $FAMILY"
$PY $S/dose_rule_read.py $P/analysis_root_gfar --tag main --verdicts $P/verdicts_$TAG/variants_${TAG}_all/as_judged.json --analysis $P/analysis_${TAG}_all_as_judged.json --arms $FAMILY --out $P/dose_rule_read_$TAG.json 2>&1 | grep -v SyntaxWarning | tee $P/dose_rule_read_$TAG.md
echo "done $(date)"
