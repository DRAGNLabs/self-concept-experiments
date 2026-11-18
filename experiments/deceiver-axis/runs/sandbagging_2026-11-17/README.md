# Sandbagging under an incentive, 2026-11-17

Plan and reading rules: PLAN "Sandbagging under an incentive" (commit 73e5aaf, before any run; stage-1 result and framing
choice in 586ffc1, before stage 2 was read). Result: FINDINGS "Sandbagging under an incentive, gi_v2 against the Goodfire
axis on gpt-oss and Muse". `N` = the schemer-axis data root, `S` = `$N/gfar_vs_gi_v2/sandbag`.

```
python3 $S/build_arms.py stage1                 # <model>/arms_stage1.tsv: unsteered, framings A and B, seed 0
python3 $S/build_arms.py stage2 <gptoss|muse> b # <model>/arms_stage2.tsv: unsteered + the steering arms, seeds 1 2 3
# from the repo root; Muse also exports MODEL=meta-models/Muse-Glimmer-30B CHAT_KWARGS='{"reasoning_strength":"low"}'
export TASKS=gsm8k ARMS=$S/<model>/arms_stage2.tsv ROOT=$S/<model>/stage2 ARMS_PER_TASK=<5 gpt-oss | 3 Muse>
sbatch --array=<0-2 | 0-4> --time <00:45:00 | 00:50:00> -p cs,cs2 --qos cs --gpus-per-node=1 --mem-per-cpu=<10G | 4G> \
  --no-requeue --export=ALL experiments/deceiver-axis/slurm/capability_eval.sbatch
A=experiments/deceiver-axis/scripts
python $A/sandbag_analysis.py stage1 --root $S/<model>/stage1 --out $S/<model>/stage1.json
python $A/sandbag_analysis.py analyze --root $S/<model>/stage2 --out $S/<model>/analysis.json --md $S/<model>/analysis.md
python $A/sandbag_chart.py --analysis "gpt-oss-120b=$S/gptoss/analysis.json" "Muse-Glimmer-30B=$S/muse/analysis.json" \
  --out $S/sandbag_gap_2026-11-17.png
```

- `<model>_stage1.json`: unsteered accuracy per notice and the gap per framing (300 items, seed 0).
- `<model>_analysis.{json,md}`: per arm the accuracies, the gap with its bootstrap interval, the usable flag, the contrast
  with unsteered (sign-flip p, Holm within the axis family), gi_v2 minus Goodfire at equal α, and under the incentive
  notice the outcome classes and the share of answers that mention the consequence.
- `<model>_arms_stage{1,2}.tsv`, `build_arms.py`: the arm lines as run. `gpu_pick_2026-11-17.txt`: picker output and
  scheduling decisions, including the cancelled Muse submission and the rerun of Muse Goodfire cap p25 (15486740).
- Jobs: 15485765, 15485768 (stage 1); 15485848 (gpt-oss stage 2); 15485936 and 15486740 (Muse stage 2).
  All on `cs` A100 80 GB cards except 15486740, which ran on a `cs2` H100 (Muse Goodfire cap p25: `control` seed 3 and the
  three `incentive` passes).

## Ten seeds, Gemma and Qwen (added the same evening)

Plan: PLAN "Sandbagging under an incentive: Gemma, Qwen and more arms" (commits e1a85d4, 5f2d41b before submission; 63f2b9f
stage-1 read; acf3049 Qwen on the user's word). Result for gpt-oss, Muse and Qwen: FINDINGS "Sandbagging under an incentive
at ten seeds, with Qwen added". Gemma's stage 2 was still running when this was written.

```
python3 $S/build_arms.py stage1 gemma qwen
python3 $S/build_arms.py stage2 gptoss b --seeds 4 5 6 7 8 9 10 --out arms_stage2_seeds4-10.tsv
python3 $S/build_arms.py stage2 muse b --seeds 4 5 6 7 8 9 10 --out arms_stage2_seeds4-10.tsv --skip gf_cap_p25
python3 $S/build_arms.py stage2 muse b --seeds 4 5 6 7 8 9 10 --out arms_stage2_seeds4-10_gfcapp25.tsv --only gf_cap_p25 \
  --split-variants
python3 $S/build_arms.py stage2 qwen b --seeds 1 2 3 4 5 6 7 8 9 10 --out arms_stage2.tsv --skip gf_cap_p25
python3 $S/build_arms.py stage2 gemma b --seeds 1 2 3 4 5 --out arms_stage2_seeds1-5.tsv    # and 6 7 8 9 10
# Qwen: MODEL=Qwen/Qwen3.8-27B VLLM_MAX_NUM_SEQS=128, --gpus-per-node=1 --mem-per-cpu=5G, one arm per task, 01:45:00
# Gemma: MODEL=google/gemma-4-31B-it CHAT_KWARGS='{"enable_thinking":true}' TP=2
#        ENGINE_KWARGS='{"attention_backend":"TRITON_ATTN"}', --gpus-per-node=2 --mem-per-cpu=6G, one arm per task
python $A/sandbag_analysis.py analyze --root $S/<model>/stage2 --out $S/<model>/analysis_10seeds.json \
  --md $S/<model>/analysis_10seeds.md          # Qwen: analysis.{json,md}
python $A/sandbag_chart.py --analysis "gpt-oss-120b=$S/gptoss/analysis_10seeds.json" \
  "Muse-Glimmer-30B=$S/muse/analysis_10seeds.json" "Qwen3.8-27B=$S/qwen/analysis.json" \
  --out $S/sandbag_gap_10seeds_gptoss_muse_qwen_2026-11-17.png
```

- `<gptoss|muse>_analysis_10seeds.{json,md}`: seeds 1 to 10 (the three-seed files above are kept as they were).
  `qwen_analysis.{json,md}`: seeds 1 to 10; this run is outside the stage-1 rule (stage-1 gap +8.3 points, bar 10) and
  was made on the user's word. `<gemma|qwen>_stage1.json`: the unsteered stage-1 read.
- `build_arms.py` is the current builder (options `--seeds`, `--out`, `--only`, `--skip`, `--split-variants`; it checks
  each arm line against the steering block recorded with the benchmark arm). The version that built the three-seed files
  is in this directory's history (commit 027a83e).
- Jobs: 15488504 (gpt-oss seeds 4-10), 15488508 and 15488549 (Muse seeds 4-10), 15490866 (Muse Goodfire cap p25,
  `control` seed 10, rerun whole after 15488549_0 was cut by its time limit in that pass), 15488311 and 15488349
  (stage 1, Gemma and Qwen), 15489355 and 15489364 (Qwen stage 2), 15488908, 15488917, 15490785, 15490787, 15490788
  (Gemma stage 2, running). Card type per task is in the FINDINGS run notes; decisions in `gpu_pick_2026-11-17.txt`.
