# Sandbagging under an incentive on a second benchmark (MMLU-Pro), 2026-11-18

Plan and reading rules: PLAN "Sandbagging under an incentive: a second benchmark, MMLU-Pro" (commit 0ee8cb2, before any run;
stage-1 reads 38cb9d3, b84aabf, 8ee2195; gpt-oss, Qwen and Gemma on the user's word 371f115, 2ba02e2). Result for Muse and
gpt-oss: FINDINGS "Sandbagging on a second benchmark (MMLU-Pro), Muse and gpt-oss". Gemma and Qwen stage 2 were still
running when this was written. `N` = the schemer-axis data root, `S` = `$N/gfar_vs_gi_v2/sandbag`, `M` = `$S/mmlu_pro`.

```
python3 $S/build_arms.py --benchmark mmlu_pro stage1 gptoss muse gemma qwen  # $M/<model>/arms_stage1.tsv
python3 $S/build_arms.py --benchmark mmlu_pro stage2 <model> b --seeds 1 2 3 4 5 --out arms_stage2.tsv [--skip gf_cap_p25]
python3 $S/build_arms.py --benchmark mmlu_pro stage2 <model> b --seeds 1 2 3 4 5 --out arms_stage2_gfcapp25.tsv \
  --only gf_cap_p25 [--split-variants]
# from the repo root, with the per-model MODEL / CHAT_KWARGS / TP / ENGINE_KWARGS of the GSM8K runs
export TASKS=mmlu_pro ARMS=$M/<model>/arms_stage2.tsv ROOT=$M/<model>/stage2 ARMS_PER_TASK=<5 gpt-oss | 3 Muse | 1>
# Gemma and Qwen also: EXTRA_ARGS="--batch-size 308" (one batch per pass; sampling seeds are per item)
sbatch --array=<...> --time <...> -p cs,cs2 --qos cs --gpus-per-node=<1 | 2 Gemma> --mem-per-cpu=<10G|4G|6G|5G> \
  --no-requeue --export=ALL experiments/deceiver-axis/slurm/capability_eval.sbatch
A=experiments/deceiver-axis/scripts
python $A/sandbag_analysis.py stage1 --root $M/<model>/stage1 --task mmlu_pro --out $M/<model>/stage1.json
python $A/sandbag_analysis.py analyze --root $M/<model>/stage2 --task mmlu_pro --out $M/<model>/analysis.json \
  --md $M/<model>/analysis.md
python $A/sandbag_chart.py --analysis "Muse-Glimmer-30B=$M/muse/analysis.json" "gpt-oss-120b=$M/gptoss/analysis.json" \
  --benchmark "MMLU-Pro" --out $M/sandbag_gap_mmlu_pro_muse_gptoss_2026-11-18.png
```

- `<model>_stage1.json`: unsteered accuracy per notice and the gap per framing (308 items, seed 0), all four models.
- `<model>_analysis.{json,md}` (Muse, gpt-oss): per arm the accuracies, the gap with its bootstrap interval, the usable
  flag, the contrast with unsteered (sign-flip p, Holm within the axis family), gi_v2 minus Goodfire at equal α, the
  outcome classes under the incentive notice and the share of answers that mention the consequence.
- `<model>_arms_*.tsv`, `build_arms.py`: the arm lines as run. `gpu_pick_2026-11-18.txt`: picker output and decisions.
- Notices: `data/sandbagging/mmlu_<a|b>_<control|incentive>.txt`.
- Jobs: stage 1 15507029 (gpt-oss), 15507037 (Muse), 15507041 (Qwen), 15507045 (Gemma). Muse stage 2 15507387 (task 1
  cancelled by me after its second arm: limit undersized), 15507474 (Goodfire cap p25), 15510430 (gi_v2 cap p25).
  gpt-oss stage 2 15509177, 15509214. Gemma stage 2 15510137, 15510322, 15510333. Qwen stage 2 15510529, 15510539.
