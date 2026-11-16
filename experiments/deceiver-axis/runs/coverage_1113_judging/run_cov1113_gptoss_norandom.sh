#!/bin/bash
# gpt-oss set gfx02, Goodfire family WITHOUT the three isotropic random add arms (user decision 2026-11-16 ~1:41 pm, PLAN amendment
# 1:42 pm; made after the 16-contrast files analysis_gfx02_* were read, so both are kept and reported). Same verdict variants and
# episodes as run_cov1113_gptoss.sh; own analysis root (analysis_root_gfar_nr) without the random arm links; Holm over the 13
# Goodfire contrasts. Outputs analysis_gfx02nr_{all,proxy}_<variant>.json, dose_rule_read_gfx02nr.{json,md}.
set -e
N=/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis; P=$N/gi_v2/pilot; TAG=gfx02; V=$P/verdicts_$TAG
PY=/home/jansen05/self-concept-experiments/.venv-313/bin/python; S=experiments/deceiver-axis/scripts
cd /home/jansen05/self-concept-experiments; export OPENSSL_CONF=/dev/null
FAMILY=$(ls $P/analysis_root_gfar | grep '^gf_' | tr '\n' ' ')
R=$P/analysis_root_gfar_nr; mkdir -p $R
ln -sfn "$(readlink $P/analysis_root_gfar/unsteered)" $R/unsteered
for a in $FAMILY; do ln -sfn ../$a $R/$a; done; ls $R | tr '\n' ' '; echo
for v in as_judged as_judged_borderline_up as_judged_weak_down; do
  $PY $S/main_analysis.py $R --tag main --contrasts $FAMILY --verdicts $V/variants_${TAG}_all/$v.json --out $P/analysis_${TAG}nr_all_$v.json > $P/analysis_${TAG}nr_all_$v.log 2>&1
  $PY $S/main_analysis.py $R --tag main --contrasts $FAMILY --verdicts $V/variants_${TAG}_proxy/$v.json --out $P/analysis_${TAG}nr_proxy_$v.json > $P/analysis_${TAG}nr_proxy_$v.log 2>&1
done
$PY $S/dose_rule_read.py $R --tag main --verdicts $V/variants_${TAG}_all/as_judged.json --analysis $P/analysis_${TAG}nr_all_as_judged.json --arms $FAMILY --out $P/dose_rule_read_${TAG}nr.json 2>&1 | grep -v SyntaxWarning | tee $P/dose_rule_read_${TAG}nr.md
grep -v SyntaxWarning $P/analysis_${TAG}nr_all_as_judged.log | grep -A16 "== judged_hack" | cut -c1-200
echo "done $(date)"
