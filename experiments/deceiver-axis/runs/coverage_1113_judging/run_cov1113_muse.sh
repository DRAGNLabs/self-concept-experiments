#!/bin/bash
# Post-judging analysis for the Muse set gfx03 = Koby's coverage arms of 2026-11-13 (Goodfire floors gf_floor_p50 / p75 / p95 and
# doses gf_m0.04 / gf_p0.04, 12 seeds each; PLAN "Koby's coverage batch of 2026-11-13: judging"). Copy of run_m01_analysis.sh:
# the gfar Holm family = every judged Goodfire arm of the model (verdicts_gf*/ and verdicts_ctl_m01_s0-11/), against the October
# unsteered arm. The three isotropic random add arms are NOT contrasts (user decision 2026-11-16 ~1:41 pm, PLAN amendment 1:42 pm,
# made before any Muse verdict of this set was collected). Own analysis root (analysis_root_gfar15) so earlier reads stay
# as written. Dose rule on hidden pass over all episodes.
set -e
TAG=gfx03; PREFIX=gfx03
cd /home/jansen05/self-concept-experiments; export OPENSSL_CONF=/dev/null
N=/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis
PY=/home/jansen05/self-concept-experiments/.venv-313/bin/python; S=experiments/deceiver-axis/scripts; VB=$N/gi_v2/pilot/verdicts_main
P=$N/muse_glimmer/gi_v2/pilot; V0=$P/verdicts_main
ALL_DETAIL="$V0/verdicts_main.detail.json $V0/verdicts_hidden_main.detail.json $V0/verdicts_p016.detail.json $V0/verdicts_hidden_p016.detail.json $V0/ref_all.detail.json"
PROXY_DETAIL="$V0/verdicts_main.detail.json $V0/verdicts_p016.detail.json $V0/ref_proxy.detail.json"; RANDOMS="random_plus random_plus_1730 random_plus_1731"
V=$P/verdicts_$TAG
echo "== verify  $(date)"
$PY $VB/verify_batches.py $P/dossiers_$TAG $V/proxy_batches ${PREFIX}_batch_ | tail -3
$PY $VB/verify_batches.py $P/dossiers_hidden_$TAG $V/hidden_batches hidden_${PREFIX}_batch_ | tail -3
echo "== collect"
$PY $S/collect_verdicts.py --verdicts $V/proxy_batches --dossier-sets $P/dossiers_$TAG --out $V/verdicts_$TAG.json 2>&1 | tee $V/collect_$TAG.log
$PY $S/collect_verdicts.py --verdicts $V/hidden_batches --dossier-sets $P/dossiers_hidden_$TAG --out $V/verdicts_hidden_$TAG.json 2>&1 | tee $V/collect_hidden_$TAG.log
FAMILY=""; GF_DETAIL=""; GF_PROXY=""
for d in $P/verdicts_gf*/ $P/verdicts_ctl_m01_s0-11/; do t=$(basename $d); t=${t#verdicts_}; [ -f $d/verdicts_$t.detail.json ] || continue
  GF_DETAIL="$GF_DETAIL $d/verdicts_$t.detail.json"; [ -f $d/verdicts_hidden_$t.detail.json ] && GF_DETAIL="$GF_DETAIL $d/verdicts_hidden_$t.detail.json"; GF_PROXY="$GF_PROXY $d/verdicts_$t.detail.json"
  FAMILY="$FAMILY $($PY -c "import json,sys; print(' '.join(sorted({k.split('/')[0] for k in json.load(open(sys.argv[1]))} - {'unsteered_rerun'})))" $d/verdicts_$t.detail.json)"
done
FAMILY=$(echo $FAMILY | tr ' ' '\n' | sort -u | tr '\n' ' ')
echo "gfar Holm family: $FAMILY"
echo "== variants"
$PY $S/verdict_variants.py --detail $ALL_DETAIL $GF_DETAIL --out-dir $V/variants_${TAG}_all 2>&1 | tee $V/variants_${TAG}_all.log | grep -E '^as_judged ' | cut -c1-600
$PY $S/verdict_variants.py --detail $PROXY_DETAIL $GF_PROXY --out-dir $V/variants_${TAG}_proxy 2>&1 | tee $V/variants_${TAG}_proxy.log | grep -E '^as_judged ' | cut -c1-600
echo "== gfar family against the October unsteered arm"
R=$P/analysis_root_gfar15; mkdir -p $R
for l in unsteered; do ln -sfn "$(readlink $P/analysis_root_main/$l)" $R/$l; done
for a in $FAMILY; do ln -sfn ../$a $R/$a; done; rm -f $R/gi_cap_p25 $R/unsteered_rerun; ls $R | tr '\n' ' '; echo
for v in as_judged as_judged_borderline_up as_judged_weak_down; do
  $PY $S/main_analysis.py $R --tag main --contrasts $FAMILY --verdicts $V/variants_${TAG}_all/$v.json --out $P/analysis_${TAG}_all_$v.json > $P/analysis_${TAG}_all_$v.log 2>&1
  $PY $S/main_analysis.py $R --tag main --contrasts $FAMILY --verdicts $V/variants_${TAG}_proxy/$v.json --out $P/analysis_${TAG}_proxy_$v.json > $P/analysis_${TAG}_proxy_$v.log 2>&1
done
$PY $S/pilot_summary.py $R --tag main --out $P/summary_gfar15.json > $P/pilot_summary_gfar15.log 2>&1
echo "== dose rule (hidden pass over all episodes, no code)"
$PY $S/dose_rule_read.py $R --tag main --verdicts $V/variants_${TAG}_all/as_judged.json --analysis $P/analysis_${TAG}_all_as_judged.json --arms $FAMILY --out $P/dose_rule_read_$TAG.json 2>&1 | grep -v SyntaxWarning | tee $P/dose_rule_read_$TAG.md
echo "== as_judged (all)"; grep -v SyntaxWarning $P/analysis_${TAG}_all_as_judged.log | grep -E "^==|unsteered|gf_|random|^arm" | cut -c1-220
echo "done $(date)"
