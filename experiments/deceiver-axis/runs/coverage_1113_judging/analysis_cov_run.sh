#!/bin/bash
# Koby's coverage arms of 2026-11-13 on Qwen3.8-27B qwen_hard61 (PLAN "Koby's coverage batch of 2026-11-13: judging"), judged in
# two blind sets: gfx04 (Goodfire floors p50 / p75 / p95, Goodfire cap p50, Goodfire +-0.16) and gix01 (gi_v2 cap p50).
# (1) collect both sets; (2) Goodfire family = gfar_L32_a+-0.08 + gfar_cap_p25 (verdicts_gfdose, verdicts_gfcap) + the six new
# arms, 9 contrasts, against Koby's unsteered arm (verdicts_main) -> analysis_gfar_gfx04; (3) gi_v2 family = the eleven contrasts
# of analysis_floor + cap_p50_L22-42, 12 contrasts -> analysis_gicap50. New arm directories are the cov1113/ symlinks to the share
# copy. From analysis_gfar_run.sh and analysis_floor_run.sh; dose-rule read added (hidden pass over all episodes, no code).
export OPENSSL_CONF=/dev/null
P=/home/jansen05/self-concept-experiments/.venv-313/bin/python
SC=/home/jansen05/self-concept-experiments/experiments/deceiver-axis/scripts
SH=/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis
Q=/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis/qwen38_27b_gi_v2/rh/qwen_hard61
cd $Q || exit 1
# TAGS / PART (both|gfar|gi) let one family run as soon as its set is judged (added 2026-11-16 2:07 pm)
for TAG in ${TAGS:-gfx04 gix01}; do V=verdicts_$TAG
  for s in proxy hidden hidden_imp; do ls $V/${s}_batches/*batch_*.jsonl >/dev/null 2>&1 || continue
    mkdir -p $V/collect_$s; for f in $V/${s}_batches/*batch_*.jsonl; do ln -sf $Q/$f $V/collect_$s/; done; done
  $P $SC/collect_verdicts.py --verdicts $V/collect_proxy --dossier-sets $Q/dossiers_$TAG --out $V/verdicts_$TAG.json || exit 1
  $P $SC/collect_verdicts.py --verdicts $V/collect_hidden --dossier-sets $Q/dossiers_hidden_$TAG --out $V/verdicts_hidden_$TAG.json || exit 1
  if [ -d $V/collect_hidden_imp ]; then $P $SC/collect_verdicts.py --verdicts $V/collect_hidden_imp --dossier-sets $Q/dossiers_hidden_imp_$TAG --out $V/verdicts_hidden_imp_$TAG.json || exit 1; fi
done
VM=verdicts_main
run() { # run <analysis dir> <root> <variants prefix> <summary name> <contrasts...>; uses arrays ALL and PROXY
  A=$1; RT=$2; VP=$3; SN=$4; shift 4; C="$@"; mkdir -p $A
  $P $SC/verdict_variants.py --detail "${ALL[@]}" --out-dir ${VP}_all > $A/variants_all.log 2>&1 || { echo FAILED variants; exit 1; }
  $P $SC/verdict_variants.py --detail "${PROXY[@]}" --out-dir ${VP}_proxy > $A/variants_proxy.log 2>&1 || { echo FAILED variants; exit 1; }
  for fam in all proxy; do for var in as_judged as_judged_borderline_up as_judged_weak_down; do
    nice -n 10 $P $SC/main_analysis.py $RT --tag pilot --reps 20000 --verdicts ${VP}_${fam}/$var.json --out $A/analysis_${fam}_${var}.json --contrasts $C > $A/analysis_${fam}_${var}.log 2>&1 || echo "FAILED $fam $var"
  done; done
  nice -n 10 $P $SC/pilot_summary.py $RT --tag pilot --out $A/$SN.json > $A/$SN.log 2>&1 || echo "FAILED summary"
  nice -n 10 $P $SC/dose_rule_read.py $RT --tag pilot --verdicts ${VP}_all/as_judged.json --analysis $A/analysis_all_as_judged.json --arms $C --out $A/dose_rule_read.json 2>&1 | grep -v SyntaxWarning > $A/dose_rule_read.md || echo "FAILED dose rule"
}
# (2) Goodfire family
if [ "${PART:-both}" != gi ]; then
ALL=($VM/verdicts_main.detail.json $VM/verdicts_hidden_main.detail.json $VM/verdicts_hidden_imp_main.detail.json); PROXY=($VM/verdicts_main.detail.json)
for t in gfdose gfcap gfx04; do d=verdicts_$t
  ALL+=($d/verdicts_$t.detail.json $d/verdicts_hidden_$t.detail.json); PROXY+=($d/verdicts_$t.detail.json)
  [ -f $d/verdicts_hidden_imp_$t.detail.json ] && ALL+=($d/verdicts_hidden_imp_$t.detail.json)
done
OLD="gfar_L32_a-0.08 gfar_L32_a0.08 gfar_cap_p25_L22-42"; NEW="gfar_L32_a-0.16 gfar_L32_a0.16 gfar_cap_p50_L22-42 gfar_floor_p50_L22-42 gfar_floor_p75_L22-42 gfar_floor_p95_L22-42"
RT=analysis_root_gfar9; mkdir -p $RT; ln -sfn $SH/qwen38_27b_gi_v2/rh/qwen_hard61/unsteered $RT/unsteered
for a in $OLD; do ln -sfn $Q/$a $RT/$a; done; for a in $NEW; do ln -sfn $Q/cov1113/$a $RT/$a; done; ls $RT | tr '\n' ' '; echo
run analysis_gfar_gfx04 $RT verdicts_gfx04/variants_gfx04 summary_gfar gfar_L32_a-0.16 gfar_L32_a-0.08 gfar_L32_a0.08 gfar_L32_a0.16 gfar_cap_p25_L22-42 gfar_cap_p50_L22-42 gfar_floor_p50_L22-42 gfar_floor_p75_L22-42 gfar_floor_p95_L22-42
echo GFAR_DONE $(date)
fi
# (3) gi_v2 family
if [ "${PART:-both}" != gfar ]; then
V=verdicts_floor; VA=verdicts_a024; VG=verdicts_gix01
ALL=($VM/verdicts_main.detail.json $VM/verdicts_hidden_main.detail.json $VM/verdicts_hidden_imp_main.detail.json $VA/verdicts_a024.detail.json $VA/verdicts_hidden_a024.detail.json $V/verdicts_floor.detail.json $V/verdicts_hidden_floor.detail.json $VG/verdicts_gix01.detail.json $VG/verdicts_hidden_gix01.detail.json)
[ -f $V/verdicts_hidden_imp_floor.detail.json ] && ALL+=($V/verdicts_hidden_imp_floor.detail.json)
[ -f $VG/verdicts_hidden_imp_gix01.detail.json ] && ALL+=($VG/verdicts_hidden_imp_gix01.detail.json)
PROXY=($VM/verdicts_main.detail.json $VA/verdicts_a024.detail.json $V/verdicts_floor.detail.json $VG/verdicts_gix01.detail.json)
RT=analysis_root_gi12; mkdir -p $RT
for l in analysis_root_floor/*; do ln -sfn "$(readlink $l)" $RT/$(basename $l); done; ln -sfn $Q/cov1113/cap_p50_L22-42 $RT/cap_p50_L22-42; ls $RT | tr '\n' ' '; echo
run analysis_gicap50 $RT verdicts_gix01/variants_gix01 summary_gicap50 L32_a-0.32 L32_a-0.16 L32_a-0.08 L32_a0.08 L32_a0.16 L32_a0.24 L32_a0.32 cap_p25_L22-42 cap_p50_L22-42 floor_p50_L22-42 floor_p75_L22-42 floor_p95_L22-42
fi
echo ANALYSIS_DONE $(date)
