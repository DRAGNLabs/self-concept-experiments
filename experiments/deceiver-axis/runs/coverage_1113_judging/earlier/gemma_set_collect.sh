#!/bin/bash
# Verify + collect one per-arm Gemma blind set built by build_ctl_set.sh. Usage: gemma_set_collect.sh <TAG> <PREFIX>   e.g. ctl_g01_s0-15 g01_s015
set -e
TAG=$1; PREFIX=$2
export OPENSSL_CONF=/dev/null
P=/home/jansen05/self-concept-experiments/.venv-313/bin/python
SC=/home/jansen05/self-concept-experiments/experiments/deceiver-axis/scripts
O=/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis/gemma4_31b_gi_v2/rh/hard108
V=$O/verdicts_$TAG
$P $V/verify_batches.py $O/dossiers_$TAG $V/proxy_batches ${PREFIX}_batch_ --quiet | tail -1
$P $V/verify_batches.py $O/dossiers_hidden_$TAG $V/hidden_batches hidden_${PREFIX}_batch_ --quiet | tail -1
$P $SC/collect_verdicts.py --verdicts $V/proxy_batches --dossier-sets $O/dossiers_$TAG --out $V/verdicts_$TAG.json
$P $SC/collect_verdicts.py --verdicts $V/hidden_batches --dossier-sets $O/dossiers_hidden_$TAG --out $V/verdicts_hidden_$TAG.json
