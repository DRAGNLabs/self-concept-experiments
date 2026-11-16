#!/bin/bash
# Detached judging chain for Koby's coverage batch of 2026-11-13 (PLAN "Koby's coverage batch of 2026-11-13: judging", 2026-11-16):
# five blind sets one after another, each proxy -> hidden (-> hidden_imp on Qwen), Sonnet rubric-judge through judge_runner.py,
# 6 judges at once and never more. A stage that the runner stops early (plan limit, three failures in a row) stops the chain;
# batches with rubric field-value notes or a failed retry do not (they are read at verify time). Markers in cov1113_chain.log:
# CHAINSTART / SETSTART / STAGEDONE <set> <kind> rc / SETDONE / CHAINSTOP / CHAINDONE. Rerunning the script resumes (finished batches are skipped).
cd /home/jansen05/self-concept-experiments || exit 1; export OPENSSL_CONF=/dev/null
N=/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis
PY=/home/jansen05/self-concept-experiments/.venv-313/bin/python; L=$N/gfar_vs_gi_v2/cov1113_chain.log
echo "CHAINSTART $(date) pid $$ host $(hostname)" >> $L
for sv in "gptoss_gfx02 $N/gi_v2/pilot/verdicts_gfx02" "gemma_g10 $N/gemma4_31b_gi_v2/rh/hard108/verdicts_ctl_g10_s0-15" \
          "muse_gfx03 $N/muse_glimmer/gi_v2/pilot/verdicts_gfx03" "qwen_gix01 $N/qwen38_27b_gi_v2/rh/qwen_hard61/verdicts_gix01" \
          "qwen_gfx04 $N/qwen38_27b_gi_v2/rh/qwen_hard61/verdicts_gfx04"; do set -- $sv; NAME=$1; V=$2; H=$V/headless_run_$NAME.log
  echo "SETSTART $NAME $(date)" >> $L
  for K in proxy hidden hidden_imp; do T=$V/JUDGE_TASK_$K.md
    if [ ! -f $T ] || ! grep -q . <(ls $(dirname $V)/dossiers$([ $K = proxy ] || echo _$K)_${V##*/verdicts_}/batches/batch_*.list 2>/dev/null); then echo "SKIP $NAME $K (no task file or no batches) $(date)" >> $L; continue; fi
    $PY experiments/deceiver-axis/scripts/judge_runner.py --task $T --batches all --jobs 6 --stop-util ${STOP_UTIL:-0.95} --log $V/judge_runner_log_$K.jsonl >> $H 2>&1; rc=$?
    echo "STAGEDONE $NAME $K rc=$rc $(date) | $(grep 'SUMMARY' $H | tail -1 | cut -c1-600)" >> $L
    grep 'SUMMARY' $H | tail -1 | grep -q 'stopped early: no' || { echo "CHAINSTOP $NAME $K $(date)" >> $L; exit 1; }
  done
  echo "SETDONE $NAME $(date)" >> $L
done
echo "CHAINDONE $(date)" >> $L
