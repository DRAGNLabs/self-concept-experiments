#!/bin/bash
# Blind-set builder for the cap dose-ladder arms of a gi_v2 pilot (gpt-oss or Muse), the hard108 build_ctl_set.sh recipe
# on the 45-problem x 12-seed main-run layout: one mixed proxy set over the given arms (shown-pass/hidden-fail episodes),
# one hidden-pass signature set (answer17_anecdotes/coding/screen_hidden_sig.py, unchanged rule), dossiers with
# --omit-hidden-suite, batches <= 300 KB / 6, and self-contained JUDGE_TASK_{proxy,hidden}.md (hard108 judge_templates
# task text + this model's own JUDGE_BRIEF.md / JUDGE_BRIEF_HIDDEN.md + a verbatim rubric copy) for rubric-judge agents.
# Usage: build_ladder_set.sh <gptoss|muse> <TAG> <PREFIX> <ARM...>   e.g. build_ladder_set.sh gptoss capladder ladder gi_cap_p5 gi_cap_p50
set -e
MODEL=$1; TAG=$2; PREFIX=$3; shift 3; ARMS="$@"
cd /home/jansen05/self-concept-experiments; export OPENSSL_CONF=/dev/null
N=/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis
case $MODEL in gptoss) ROOT=$N/gi_v2/pilot; SALT=gi-gptoss-v2;; muse) ROOT=$N/muse_glimmer/gi_v2/pilot; SALT=gi-muse-v2;; *) echo "model gptoss|muse"; exit 1;; esac
DATA=/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/rh_hillclimb/data
PY=/home/jansen05/self-concept-experiments/.venv-313/bin/python
T=$N/gemma4_31b_gi_v2/rh/hard108/judge_templates; RUBRIC=/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/rh_hillclimb/judge_rubric_v3.md
DATE=$(date +%Y-%m-%d)
RUNS=(); for a in $ARMS; do for s in $(seq 0 11); do [ -f $ROOT/$a/seed$s/main_impossible_original.jsonl ] || { echo "missing $a/seed$s"; exit 1; }; done; RUNS+=("$a=$ROOT/$a"); done
rm -rf $ROOT/dossiers_$TAG
nice -n 10 timeout 900 $PY experiments/deceiver-axis/scripts/make_dossiers.py --data $DATA --out $ROOT/dossiers_$TAG --runs "${RUNS[@]}" --salt $SALT-$TAG-$DATE --omit-hidden-suite 2>&1 | tail -4
nice -n 10 $PY experiments/deceiver-axis/scripts/repack_batches.py $ROOT/dossiers_$TAG 300 6
echo "proxy dossiers: $(ls $ROOT/dossiers_$TAG/dossiers | wc -l); batches: $(ls $ROOT/dossiers_$TAG/batches/*.list | wc -l)"
HRUNS=(); rm -rf $ROOT/hidden_sig_$TAG; for a in $ARMS; do nice -n 10 $PY $N/answer17_anecdotes/coding/screen_hidden_sig.py $ROOT/$a $ROOT/hidden_sig_$TAG/$a; HRUNS+=("$a=$ROOT/hidden_sig_$TAG/$a"); done
rm -rf $ROOT/dossiers_hidden_$TAG
nice -n 10 timeout 900 $PY experiments/deceiver-axis/scripts/make_dossiers.py --data $DATA --out $ROOT/dossiers_hidden_$TAG --runs "${HRUNS[@]}" --salt $SALT-hidden-$TAG-$DATE --all --omit-hidden-suite 2>&1 | tail -4
nice -n 10 $PY experiments/deceiver-axis/scripts/repack_batches.py $ROOT/dossiers_hidden_$TAG 300 6
echo "hidden dossiers: $(ls $ROOT/dossiers_hidden_$TAG/dossiers | wc -l); batches: $(ls $ROOT/dossiers_hidden_$TAG/batches/*.list | wc -l)"
V=$ROOT/verdicts_$TAG; mkdir -p $V/proxy_batches $V/hidden_batches $V/superseded
subst() { sed -e "s#__O__#$ROOT#g" -e "s#__TAG__#$TAG#g" -e "s#__V__#$V#g" -e "s#__PREFIX__#$PREFIX#g" "$1"; }
{ subst $T/TASK_proxy.tmpl; echo; echo ---; echo; cat $ROOT/verdicts_main/JUDGE_BRIEF.md; echo; echo ---; echo; echo "# Rubric v3 (verbatim copy of $RUBRIC)"; echo; cat $RUBRIC; } > $V/JUDGE_TASK_proxy.md
{ subst $T/TASK_hidden.tmpl; echo; echo ---; echo; cat $ROOT/verdicts_main/JUDGE_BRIEF_HIDDEN.md; echo; echo ---; echo; cat $ROOT/verdicts_main/JUDGE_BRIEF.md; echo; echo ---; echo; echo "# Rubric v3 (verbatim copy of $RUBRIC)"; echo; cat $RUBRIC; } > $V/JUDGE_TASK_hidden.md
grep -c "$TAG" $V/JUDGE_TASK_proxy.md $V/JUDGE_TASK_hidden.md; wc -c $V/JUDGE_TASK_proxy.md $V/JUDGE_TASK_hidden.md
echo "built $V  $(date)"
