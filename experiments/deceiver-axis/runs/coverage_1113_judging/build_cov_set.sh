#!/bin/bash
# Blind judging set for Koby's coverage arms of 2026-11-13 on Qwen3.8-27B qwen_hard61 (PLAN "Koby's coverage batch of 2026-11-13: judging"), 2026-11-16.
# Usage: build_cov_set.sh <TAG> <PREFIX> <ARM...>; arm directories are symlinks under $Q/cov1113 to the share copy (read-only; our two-seed pilot
# directories of gfar_L32_a+-0.16 under $Q carry the same names). Copy of build_gfar_set.sh with the arm root, the hidden-screen output dirs and the salt date changed.
# Same recipe as gemma4_31b_gi_v2/rh/hard108/build_ctl_set.sh (2026-11-05 cost pass): make_dossiers.py --omit-hidden-suite,
# repack 300 KB / 6, hidden-pass signature screen (Muse rule, unchanged), self-contained JUDGE_TASK files, rubric-judge agents.
# Plus the Qwen-pilot supplementary set: every hidden pass on lcbhard_57 / lcbhard_81 the screen did not flag (pre-registered 2026-11-06).
set -e
cd /home/jansen05/self-concept-experiments; export OPENSSL_CONF=/dev/null
PY=/home/jansen05/self-concept-experiments/.venv-313/bin/python
SH=/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis
K=$SH/qwen38_27b_gi_v2/rh/qwen_hard61; D=$SH/qwen_hard61_data
Q=/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis/qwen38_27b_gi_v2/rh/qwen_hard61
TAG=$1; PREFIX=$2; shift 2; ARMS="$@"; DATE=2026-11-16; A=$Q/cov1113
RUNS=""; for a in $ARMS; do RUNS="$RUNS $a=$A/$a"; done
rm -rf $Q/dossiers_$TAG
nice -n 10 timeout 3000 $PY experiments/deceiver-axis/scripts/make_dossiers.py --data $D --out $Q/dossiers_$TAG --runs $RUNS --salt qwen-gi-v2-hard61-$TAG-$DATE --omit-hidden-suite 2>&1 | tail -12
nice -n 10 $PY $Q/repack_batches.py $Q/dossiers_$TAG 300 6
echo "proxy dossiers: $(ls $Q/dossiers_$TAG/dossiers | wc -l); batches: $(ls $Q/dossiers_$TAG/batches/*.list | wc -l)"
HRUNS=""; for a in $ARMS; do rm -rf $Q/hidden_sig_cov1113/$a; nice -n 10 $PY $Q/hidden_sig/screen_hidden_sig.py $A/$a $Q/hidden_sig_cov1113/$a; HRUNS="$HRUNS $a=$Q/hidden_sig_cov1113/$a"; done
rm -rf $Q/dossiers_hidden_$TAG
nice -n 10 timeout 3000 $PY experiments/deceiver-axis/scripts/make_dossiers.py --data $D --out $Q/dossiers_hidden_$TAG --runs $HRUNS --salt qwen-gi-v2-hidden-hard61-$TAG-$DATE --all --omit-hidden-suite 2>&1 | tail -12
nice -n 10 $PY $Q/repack_batches.py $Q/dossiers_hidden_$TAG 300 6
echo "hidden dossiers: $(ls $Q/dossiers_hidden_$TAG/dossiers | wc -l); batches: $(ls $Q/dossiers_hidden_$TAG/batches/*.list | wc -l)"
# supplementary set: hidden passes on lcbhard_57 / lcbhard_81 not flagged by the screen
IRUNS=""; for a in $ARMS; do rm -rf $Q/hidden_imp_cov1113/$a; nice -n 10 $PY - $A/$a $Q/hidden_sig_cov1113/$a $Q/hidden_imp_cov1113/$a <<'PYEOF'
import json, os, sys
from pathlib import Path
src, sig, dst = map(Path, sys.argv[1:4]); n = 0
for sd in sorted(src.glob('seed*'), key=lambda p: int(p.name[4:])):
    flagged = {json.loads(l)['example_id'] for l in (sig / sd.name / 'pilot_impossible_original.jsonl').open()}
    keep = [l if l.endswith('\n') else l + '\n' for l in (sd / 'pilot_impossible_original.jsonl').open()
            if (r := json.loads(l)).get('passed_original') and r['example_id'] in ('lcbhard_57', 'lcbhard_81') and r['example_id'] not in flagged]
    if not keep: continue
    od = dst / sd.name; od.mkdir(parents=True, exist_ok=True)
    (od / 'pilot_impossible_original.jsonl').write_text(''.join(keep)); n += len(keep)
    for rf in sd.glob('pilot_*_reasoning.jsonl'):
        (od / rf.name).exists() or os.symlink(rf.resolve(), od / rf.name)
print(src.name, 'imp (57/81 hidden passes not flagged)', n)
PYEOF
[ -d $Q/hidden_imp_cov1113/$a ] && IRUNS="$IRUNS $a=$Q/hidden_imp_cov1113/$a"; done
rm -rf $Q/dossiers_hidden_imp_$TAG
if [ -n "$IRUNS" ]; then nice -n 10 timeout 900 $PY experiments/deceiver-axis/scripts/make_dossiers.py --data $D --out $Q/dossiers_hidden_imp_$TAG --runs $IRUNS --salt qwen-gi-v2-hiddenimp-hard61-$TAG-$DATE --all --omit-hidden-suite 2>&1 | tail -12
nice -n 10 $PY $Q/repack_batches.py $Q/dossiers_hidden_imp_$TAG 300 6
echo "imp dossiers: $(ls $Q/dossiers_hidden_imp_$TAG/dossiers | wc -l); batches: $(ls $Q/dossiers_hidden_imp_$TAG/batches/*.list | wc -l)"; fi
V=$Q/verdicts_$TAG; mkdir -p $V/proxy_batches $V/hidden_batches $V/hidden_imp_batches $V/superseded
T=$Q/judge_templates; RUBRIC=/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/rh_hillclimb/judge_rubric_v3.md
cp $T/BRIEF_proxy.md $V/JUDGE_BRIEF.md; cp $T/BRIEF_hidden.md $V/JUDGE_BRIEF_HIDDEN.md
task() { sed -e "s#verdicts_main#verdicts_$TAG#g" -e "s#__Q__#$Q#g" -e "s#__SET__#$1#g" -e "s#__OUTFILE__#$2#g" -e "s#__BRIEFS__#$3#g" $T/TASK.tmpl; }
{ task dossiers_$TAG "proxy_batches/${PREFIX}_batch_N.jsonl" "the judging brief"; echo; echo ---; echo; cat $T/BRIEF_proxy.md; echo; echo ---; echo; echo "# Rubric v3 (verbatim copy of $RUBRIC)"; echo; cat $RUBRIC; } > $V/JUDGE_TASK_proxy.md
for s in hidden hidden_imp; do
{ task dossiers_${s}_$TAG "${s}_batches/${s}_${PREFIX}_batch_N.jsonl" "the hidden-set brief, the main judging brief"; echo; echo ---; echo; cat $T/BRIEF_hidden.md; echo; echo ---; echo; cat $T/BRIEF_proxy.md; echo; echo ---; echo; echo "# Rubric v3 (verbatim copy of $RUBRIC)"; echo; cat $RUBRIC; } > $V/JUDGE_TASK_$s.md; done
wc -c $V/JUDGE_TASK_*.md; du -sh $Q/dossiers_$TAG $Q/dossiers_hidden_$TAG $Q/dossiers_hidden_imp_$TAG 2>/dev/null
echo "BUILD_DONE $(date)"
