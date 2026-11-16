#!/bin/bash
# Generic blind-set builder for a controls arm over a seed range (same recipe as verdicts_ctl_a0.005_s{0-7,8-15}).
# Usage: build_ctl_set.sh <ARM> <TAG> <PREFIX> <SEEDS...>   e.g. build_ctl_set.sh L30_rand1729_a-0.005 ctl_a-0.005_s0-7 ctl_an0005_s07 0 1 2 3 4 5 6 7
# Creates stage_ctl/<TAG#ctl_>, dossiers_<TAG>, hidden_sig/<ARM>_<seedrange>, dossiers_hidden_<TAG>, and verdicts_<TAG>/ with
# self-contained JUDGE_TASK_{proxy,hidden}.md assembled from judge_templates/ (task + brief + rubric copy; 2026-11-05 cost pass:
# dossiers built with --omit-hidden-suite, judges dispatched as the rubric-judge agent (Bash only), 128 KB Bash window).
set -e
ARM=$1; TAG=$2; PREFIX=$3; shift 3; SEEDS="$@"
cd /home/jansen05/self-concept-experiments; export OPENSSL_CONF=/dev/null
O=/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis/gemma4_31b_gi_v2/rh/hard108
SH=/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis
PY=/home/jansen05/self-concept-experiments/.venv-313/bin/python
DATE=$(date +%Y-%m-%d); SR="s$(echo $SEEDS | awk '{print $1"-"$NF}')"
S=$O/stage_ctl/${TAG#ctl_}; rm -rf $S
for s in $SEEDS; do mkdir -p $S/seed$s; for f in pilot_impossible_original.jsonl pilot_hard108_reasoning.jsonl pilot_impossible_original_summary.json; do [ -f $O/$ARM/seed$s/$f ] || { echo "missing $ARM/seed$s/$f"; exit 1; }; ln -sfn $O/$ARM/seed$s/$f $S/seed$s/$f; done; done
rm -rf $O/dossiers_$TAG
nice -n 10 timeout 900 $PY experiments/deceiver-axis/scripts/make_dossiers.py --data $SH/hard108_data --out $O/dossiers_$TAG --runs $ARM=$S --salt gi-gemma31b-v2-hard108-$TAG-$DATE --omit-hidden-suite 2>&1 | tail -4
nice -n 10 $PY $O/repack_batches.py $O/dossiers_$TAG 300 6
echo "proxy dossiers: $(ls $O/dossiers_$TAG/dossiers | wc -l); batches: $(ls $O/dossiers_$TAG/batches/*.list | wc -l)"
HS=$O/hidden_sig/${ARM}_$SR; rm -rf $HS
nice -n 10 $PY $O/hidden_sig/screen_hidden_sig.py $S $HS
rm -rf $O/dossiers_hidden_$TAG
nice -n 10 timeout 900 $PY experiments/deceiver-axis/scripts/make_dossiers.py --data $SH/hard108_data --out $O/dossiers_hidden_$TAG --runs $ARM=$HS --salt gi-gemma31b-v2-hidden-hard108-$TAG-$DATE --all --omit-hidden-suite 2>&1 | tail -4
nice -n 10 $PY $O/repack_batches.py $O/dossiers_hidden_$TAG 300 6
echo "hidden dossiers: $(ls $O/dossiers_hidden_$TAG/dossiers | wc -l); batches: $(ls $O/dossiers_hidden_$TAG/batches/*.list | wc -l)"
V=$O/verdicts_$TAG; mkdir -p $V/proxy_batches $V/hidden_batches $V/superseded
T=$O/judge_templates; RUBRIC=/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/rh_hillclimb/judge_rubric_v3.md
cp -n $O/verdicts_ctl_a0.005_s8-15/verify_batches.py $V/ 2>/dev/null || true
# Self-contained task files (task + brief(s) + rubric copy): one read per judge instead of three. Criteria text is verbatim.
subst() { sed -e "s#__O__#$O#g" -e "s#__TAG__#$TAG#g" -e "s#__V__#$V#g" -e "s#__PREFIX__#$PREFIX#g" "$1"; }
{ subst $T/TASK_proxy.tmpl; echo; echo ---; echo; cat $T/BRIEF_proxy.md; echo; echo ---; echo; echo "# Rubric v3 (verbatim copy of $RUBRIC)"; echo; cat $RUBRIC; } > $V/JUDGE_TASK_proxy.md
{ subst $T/TASK_hidden.tmpl; echo; echo ---; echo; cat $T/BRIEF_hidden.md; echo; echo ---; echo; cat $T/BRIEF_proxy.md; echo; echo ---; echo; echo "# Rubric v3 (verbatim copy of $RUBRIC)"; echo; cat $RUBRIC; } > $V/JUDGE_TASK_hidden.md
grep -c "$TAG" $V/JUDGE_TASK_proxy.md $V/JUDGE_TASK_hidden.md; wc -c $V/JUDGE_TASK_proxy.md $V/JUDGE_TASK_hidden.md
echo "built $V"
