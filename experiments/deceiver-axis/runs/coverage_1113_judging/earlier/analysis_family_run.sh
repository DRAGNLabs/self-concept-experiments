#!/bin/bash
# Per-arm Gemma sets (PLAN 2026-11-11 "Gemma arms judged one at a time as they arrive"): analysis of a family against
# Koby's B200 unsteered arm with the 2026-11-02 `main` verdicts for unsteered.
# Usage: analysis_family_run.sh <NAME> <ARM>:<TAG> [<ARM>:<TAG> ...]    e.g. analysis_family_run.sh floor1 floor_p95_L20-38:ctl_g01_s0-15
NAME=$1; shift
export OPENSSL_CONF=/dev/null
P=/home/jansen05/self-concept-experiments/.venv-313/bin/python
SC=/home/jansen05/self-concept-experiments/experiments/deceiver-axis/scripts
O=/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis/gemma4_31b_gi_v2/rh/hard108
cd $O || exit 1
A=analysis_$NAME; RT=analysis_root_$NAME; mkdir -p $A $RT
ln -sfn "$(readlink analysis_root_frozen19/unsteered)" $RT/unsteered
ALL=(verdicts_main/verdicts_main.detail.json verdicts_main/verdicts_hidden_main.detail.json); PROXY=(verdicts_main/verdicts_main.detail.json)
C=""
for at in "$@"; do ARM=${at%%:*}; TAG=${at#*:}
  ln -sfn $O/$ARM $RT/$ARM; C="$C $ARM"
  ALL+=(verdicts_$TAG/verdicts_$TAG.detail.json verdicts_$TAG/verdicts_hidden_$TAG.detail.json); PROXY+=(verdicts_$TAG/verdicts_$TAG.detail.json)
done
$P $SC/verdict_variants.py --detail "${ALL[@]}" --out-dir $A/variants_all > $A/variants_all.log 2>&1 || { echo FAILED variants; exit 1; }
$P $SC/verdict_variants.py --detail "${PROXY[@]}" --out-dir $A/variants_proxy > $A/variants_proxy.log 2>&1 || { echo FAILED variants; exit 1; }
for fam in all proxy; do for var in as_judged as_judged_borderline_up as_judged_weak_down; do
  nice -n 10 $P $SC/main_analysis.py $RT --tag pilot --reps 20000 --verdicts $A/variants_$fam/$var.json --out $A/analysis_${fam}_${var}.json --contrasts $C > $A/analysis_${fam}_${var}.log 2>&1 || echo "FAILED $fam $var"
done; done
nice -n 10 $P $SC/pilot_summary.py $RT --tag pilot --out $A/summary_$NAME.json > $A/summary_$NAME.log 2>&1 || echo "FAILED summary"
nice -n 10 $P $SC/dose_rule_read.py $RT --tag pilot --verdicts $A/variants_all/as_judged.json --analysis $A/analysis_all_as_judged.json --arms $C --out $A/dose_rule_read.json > $A/dose_rule_read.log 2>&1 || echo "FAILED dose rule"
echo ANALYSIS_DONE $NAME $(date)
