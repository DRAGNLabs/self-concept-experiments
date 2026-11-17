# Goodfire-axis general-question probes, gpt-oss and Muse (2026-11-17)

Plan and reading rules: PLAN "Goodfire-axis general-question probes on gpt-oss" and "Muse: Goodfire plus doses up to +0.32".
Result: FINDINGS entry of 2026-11-17. Data root
`R = ~/nobackup/autodelete/self-concept-experiments/schemer-axis/gfar_vs_gi_v2/probes`.

- `gptoss_settings.tsv`, `gptoss_sets.tsv`, `muse_settings_plus.tsv`, `muse_sets.tsv`: the `SETTINGS` and `SETS` files of the
  `slurm/neutral_probe.sbatch` arrays (gpt-oss 15478868 low-effort set and 15479189, Muse 15479559); submissions, picker output
  and placement in `gfar_vs_gi_v2/steer/jobs.txt` and `R/{gptoss,muse}/gpu_pick_2026-11-17.txt`.
- `gptoss_analysis.{md,json}`, `muse_analysis.{md,json}`: output of `probe_blind_labels.py analyze` (100,000 draws, seed 0).

Commands (repo root, `PY = OPENSSL_CONF=/dev/null PYTHONPATH=src .venv-313/bin/python`, `S = experiments/deceiver-axis/scripts`,
`A = ~/nobackup/autodelete/self-concept-experiments/schemer-axis/gi_v2/anecdotes`, Muse `A = .../muse_glimmer/gi_v2/anecdotes`):

    $PY $S/probe_blind_labels.py build --out $R/gptoss/blind --sets probes_low probes_high probes_striking \
        --arm unsteered=$A --arm gi_m0.4=$A ... --arm gi_p0.4=$A --arm gf_m0.4=$R/gptoss ... --arm gf_p0.4=$R/gptoss \
        --asks-deception probes_striking:10,14,19,21
    $PY $S/probe_blind_labels.py usable --dir $R/gptoss/blind
    python3 $S/judge_runner.py --task $R/gptoss/blind/JUDGE_TASK.md --batches all --agent probe-labeler --fields probe --jobs 6
    $PY $S/probe_blind_labels.py analyze --dir $R/gptoss/blind --family gi_v2=gi_ --family goodfire=gf_ \
        --out $R/gptoss/blind/analysis.json > $R/gptoss/blind/analysis.md
    $PY $S/probe_labels_chart.py --analysis "gpt-oss-120b=$R/gptoss/blind/analysis.json" \
        "Muse-Glimmer-30B=$R/muse/blind/analysis.json" --out <figures>/probes/probe_labels_gptoss_muse_2026-11-17.png

Muse: the same with `$R/muse/blind` and the arms `unsteered`, `gi_p0.16`, `gi_p0.32` (10-30 files), `gf_p0.16`, `gf_p0.32`.
The gpt-oss analysis took about 14 minutes on one core of a login node (132 contrasts of 100,000 draws), the Muse one under
two minutes.
