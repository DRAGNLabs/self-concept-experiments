#!/bin/bash
# Post-judging analysis for a Goodfire-axis (gfar) blind set on gpt-oss or Muse (set built by build_ladder_set.sh): verify,
# collect, three verdict variants over the model's reference record (unsteered + random controls from the main run) plus
# every judged gfar set of the model, then main_analysis on analysis_root_gfar with the Holm family = every judged gfar arm
# (kept apart from the gi_v2 family; random arms raw). Adapted from run_ladder_analysis.sh, 2026-11-10.
# Usage: run_gfar_analysis.sh <gptoss|muse> <TAG> <PREFIX> <ARM...>   (TAG / PREFIX / arms as in the build)
set -e
MODEL=$1; TAG=$2; PREFIX=$3; shift 3; ARMS="$@"
cd /home/jansen05/self-concept-experiments; export OPENSSL_CONF=/dev/null
N=/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis
PY=/home/jansen05/self-concept-experiments/.venv-313/bin/python; S=experiments/deceiver-axis/scripts; VB=$N/gi_v2/pilot/verdicts_main
case $MODEL in
  gptoss) P=$N/gi_v2/pilot; V0=$P/verdicts_main
          ALL_DETAIL="$V0/verdicts_mixed.detail.json $V0/verdicts_hidden_mixed.detail.json $V0/ref_proxy.detail.json $V0/ref_hidden.detail.json"
          PROXY_DETAIL="$V0/verdicts_mixed.detail.json $V0/ref_proxy.detail.json"; RANDOMS="random_a0.07 random_a0.07_s1730 random_a0.07_s1731";;
  muse)   P=$N/muse_glimmer/gi_v2/pilot; V0=$P/verdicts_main
          ALL_DETAIL="$V0/verdicts_main.detail.json $V0/verdicts_hidden_main.detail.json $V0/verdicts_p016.detail.json $V0/verdicts_hidden_p016.detail.json $V0/ref_all.detail.json"
          PROXY_DETAIL="$V0/verdicts_main.detail.json $V0/verdicts_p016.detail.json $V0/ref_proxy.detail.json"; RANDOMS="random_plus random_plus_1730 random_plus_1731";;
  *) echo "model gptoss|muse"; exit 1;;
esac
V=$P/verdicts_$TAG
echo "== verify  $(date)"
$PY $VB/verify_batches.py $P/dossiers_$TAG $V/proxy_batches ${PREFIX}_batch_ | tail -3
$PY $VB/verify_batches.py $P/dossiers_hidden_$TAG $V/hidden_batches hidden_${PREFIX}_batch_ | tail -3
echo "== collect"
$PY $S/collect_verdicts.py --verdicts $V/proxy_batches --dossier-sets $P/dossiers_$TAG --out $V/verdicts_$TAG.json 2>&1 | tee $V/collect_$TAG.log
$PY $S/collect_verdicts.py --verdicts $V/hidden_batches --dossier-sets $P/dossiers_hidden_$TAG --out $V/verdicts_hidden_$TAG.json 2>&1 | tee $V/collect_hidden_$TAG.log
# every judged gfar set of the model (verdicts_gf*/ with collected detail files), this one included
FAMILY=""; GF_DETAIL=""; GF_PROXY=""
for d in $P/verdicts_gf*/; do t=$(basename $d); t=${t#verdicts_}; [ -f $d/verdicts_$t.detail.json ] || continue
  GF_DETAIL="$GF_DETAIL $d/verdicts_$t.detail.json $d/verdicts_hidden_$t.detail.json"; GF_PROXY="$GF_PROXY $d/verdicts_$t.detail.json"
  FAMILY="$FAMILY $($PY -c "import json,sys; print(' '.join(sorted({k.split('/')[0] for k in json.load(open(sys.argv[1]))})))" $d/verdicts_$t.detail.json)"
done
echo "gfar Holm family:$FAMILY"
echo "== variants"
$PY $S/verdict_variants.py --detail $ALL_DETAIL $GF_DETAIL --out-dir $V/variants_${TAG}_all 2>&1 | tee $V/variants_${TAG}_all.log | grep -E '^as_judged ' | cut -c1-400
$PY $S/verdict_variants.py --detail $PROXY_DETAIL $GF_PROXY --out-dir $V/variants_${TAG}_proxy 2>&1 | tee $V/variants_${TAG}_proxy.log | grep -E '^as_judged ' | cut -c1-400
echo "== analysis root"
R=$P/analysis_root_gfar; mkdir -p $R
for l in unsteered $RANDOMS; do ln -sfn "$(readlink $P/analysis_root_main/$l)" $R/$l; done
for a in $FAMILY; do ln -sfn ../$a $R/$a; done; rm -f $R/gi_cap_p25; ls $R | tr '\n' ' '; echo
for v in as_judged as_judged_borderline_up as_judged_weak_down; do
  $PY $S/main_analysis.py $R --tag main --contrasts $FAMILY $RANDOMS --verdicts $V/variants_${TAG}_all/$v.json --out $P/analysis_${TAG}_all_$v.json > $P/analysis_${TAG}_all_$v.log 2>&1
  $PY $S/main_analysis.py $R --tag main --contrasts $FAMILY $RANDOMS --verdicts $V/variants_${TAG}_proxy/$v.json --out $P/analysis_${TAG}_proxy_$v.json > $P/analysis_${TAG}_proxy_$v.log 2>&1
done
$PY $S/pilot_summary.py $R --tag main --out $P/summary_gfar.json > $P/pilot_summary_gfar.log 2>&1
echo "== as_judged (all), Holm over$FAMILY + randoms"; grep -v SyntaxWarning $P/analysis_${TAG}_all_as_judged.log | grep -E "^==|unsteered|gf_|random|^arm" | cut -c1-200
echo "done $(date)"
