# Run scripts: judging and analysis of Koby's coverage batch (2026-11-16)

These are the scripts that built, judged and analysed the 16 arms of Koby's coverage batch of 2026-11-13 on 2026-11-16
(PLAN "Koby's coverage batch of 2026-11-13: judging"; FINDINGS entries of 2026-11-16; run log
`gfar_vs_gi_v2/steer/jobs.txt`). They were written next to the data on `nobackup/autodelete` and are copied here
byte for byte (added 2026-11-16, 3:31 pm MST) so that they outlive that storage. `SHA256SUMS` lists the checksums. Every
shell script here is also on the group share
(`gfar_vs_gi_v2_jansen05/`, `qwen38_27b_gi_v2_judging_jansen05/qwen_hard61/`,
`gemma4_31b_gi_v2_judging_jansen05/hard108/`, `gptoss120b/gi_v2/pilot/`), identical when compared on 2026-11-16.

They are records of what ran, not tools. Paths are absolute and point at this account's `nobackup` tree
(`N=/home/jansen05/nobackup/autodelete/self-concept-experiments/schemer-axis`) and at the group share
(`SH=/grphome/grp_mirror/nobackup/autodelete/self-concept-experiments/schemer-axis`). The reusable code they call is in
`../../scripts/` (`make_dossiers.py`, `judge_runner.py`, `verify_batches.py`, `collect_verdicts.py`,
`verdict_variants.py`, `main_analysis.py`, `pilot_summary.py`, `dose_rule_read.py`, `capability_eval_compare.py`).

## Written on 2026-11-16

| file | ran from | what it did |
|---|---|---|
| `build_cov_set.sh` | `$N/qwen38_27b_gi_v2/rh/qwen_hard61/` | Builds a blind Qwen set (proxy, hidden-pass signature, supplementary `lcbhard_57` / `lcbhard_81`) from the `cov1113/` arm links. Run for `gfx04` and `gix01`. |
| `cov1113_chain.sh` | `$N/gfar_vs_gi_v2/` | Detached judging chain over the five sets (Sonnet rubric-judge through `judge_runner.py`, six at once). |
| `read_audit.py` | scratch (`audit.py`) | Lists the judge's commands for every batch whose last row in a `judge_runner_log_*.jsonl` carries a read-audit flag. |
| `run_cov1113_gptoss.sh` | `$N/gfar_vs_gi_v2/steer/` | gpt-oss set `gfx02`: verify, collect and verdict variants through `earlier/run_gfar_analysis.sh`, then the dose-rule read. |
| `run_cov1113_gptoss_norandom.sh` | `$N/gfar_vs_gi_v2/steer/` | gpt-oss Goodfire family of 13 contrasts without the isotropic random arms → `analysis_gfx02nr_*`. |
| `run_cov1113_muse.sh` | `$N/gfar_vs_gi_v2/steer/` | Muse set `gfx03`: verify, collect, variants, Goodfire family of 12 contrasts → `analysis_gfx03_*`. |
| `analysis_cov_run.sh` | `$N/qwen38_27b_gi_v2/rh/qwen_hard61/` | Qwen sets `gfx04` and `gix01`: collect, then the Goodfire family of 9 (`analysis_gfar_gfx04/`) and the gi_v2 family of 12 (`analysis_gicap50/`). |

Two notes on the gpt-oss scripts:

- `run_cov1113_gptoss.sh` ran first and its analysis files (`analysis_gfx02_*`, 16 contrasts with the three random add arms)
  are not used. Its verify, collect and variant steps are still what `run_cov1113_gptoss_norandom.sh` reads.
- The header comment of `run_cov1113_gptoss_norandom.sh` says both families "are kept and reported". That was superseded the
  same afternoon (user, about 2:33 pm): only the Goodfire-only family is reported. The comment is left as it ran.

## Earlier scripts run unchanged on 2026-11-16 (`earlier/`)

| file | original | used for |
|---|---|---|
| `build_ladder_set.sh` | `$N/gi_v2/pilot/` and `$N/muse_glimmer/gi_v2/pilot/` (identical copies, 2026-11-06) | Blind sets `gfx02` (gpt-oss) and `gfx03` (Muse). |
| `build_ctl_set.sh` | `$N/gemma4_31b_gi_v2/rh/hard108/` (2026-11-05) | Blind Gemma set `ctl_g10_s0-15`. |
| `gemma_set_collect.sh` | same directory (2026-11-11) | Verify and collect the Gemma set. |
| `analysis_family_run.sh` | same directory (2026-11-11) | Gemma Goodfire family of 7 contrasts → `analysis_gfar7/`. |
| `run_gfar_analysis.sh` | `$N/gfar_vs_gi_v2/steer/` (2026-11-10) | Called by `run_cov1113_gptoss.sh`. |
| `screen_hidden_sig_main.py` | `$N/answer17_anecdotes/coding/screen_hidden_sig.py` | Hidden-pass signature screen, main-run file layout (gpt-oss, Muse). |
| `screen_hidden_sig_pilot.py` | `hidden_sig/screen_hidden_sig.py` in the Gemma and Qwen roots (identical) | The same screen for the `pilot_` file layout. |

Not copied: the Gemma and Qwen builders call a `repack_batches.py` next to the data, which is `../../scripts/repack_batches.py`
without its two-line empty-set guard; and the judge briefs and task templates (`judge_templates/`, `JUDGE_BRIEF*.md`), which
are prompts and stay with the data and on the share.

## Commands that were typed, not scripted

Arm links (Koby's share copy `C=$SH/coverage_1113_kobyjl`, read-only):

```bash
for a in gf_floor_p50 gf_floor_p75 gf_floor_p95; do ln -s $C/gptoss120b/rh/$a $N/gi_v2/pilot/$a; done
for a in gf_floor_p50 gf_floor_p75 gf_floor_p95 gf_m0.04 gf_p0.04; do ln -s $C/muse/rh/$a $N/muse_glimmer/gi_v2/pilot/$a; done
ln -s $C/gemma4_31b/rh/gfar_floor_p50_L20-38 $N/gemma4_31b_gi_v2/rh/hard108/gfar_floor_p50_L20-38
mkdir $Q/cov1113 && for a in cap_p50_L22-42 gfar_L32_a0.16 gfar_L32_a-0.16 gfar_cap_p50_L22-42 gfar_floor_p50_L22-42 \
  gfar_floor_p75_L22-42 gfar_floor_p95_L22-42; do ln -s $C/qwen38_27b/rh/$a $Q/cov1113/$a; done
```

Set builds (`H=$N/gemma4_31b_gi_v2/rh/hard108`, `Q=$N/qwen38_27b_gi_v2/rh/qwen_hard61`):

```bash
bash $N/gi_v2/pilot/build_ladder_set.sh gptoss gfx02 gfx02 gf_floor_p50 gf_floor_p75 gf_floor_p95
bash $N/muse_glimmer/gi_v2/pilot/build_ladder_set.sh muse gfx03 gfx03 gf_floor_p50 gf_floor_p75 gf_floor_p95 gf_m0.04 gf_p0.04
bash $H/build_ctl_set.sh gfar_floor_p50_L20-38 ctl_g10_s0-15 g10_s015 $(seq 0 15)
bash $Q/build_cov_set.sh gfx04 gfx04 gfar_floor_p50_L22-42 gfar_floor_p75_L22-42 gfar_floor_p95_L22-42 gfar_cap_p50_L22-42 \
  gfar_L32_a0.16 gfar_L32_a-0.16
bash $Q/build_cov_set.sh gix01 gix01 cap_p50_L22-42
```

Judging chain (`R=$N/gfar_vs_gi_v2`, from the repo root):

```bash
setsid nohup bash $R/cov1113_chain.sh > $R/cov1113_chain.nohup 2>&1 < /dev/null &
```

Analyses, in the order they ran:

```bash
bash $N/gfar_vs_gi_v2/steer/run_cov1113_gptoss.sh
cd $H && bash gemma_set_collect.sh ctl_g10_s0-15 g10_s015 && bash analysis_family_run.sh gfar7 \
  gfar_L30_a-0.08:ctl_g07_s0-15 gfar_L30_a-0.04:ctl_g04_s0-15 gfar_L30_a0.04:ctl_g05_s0-15 gfar_L30_a0.08:ctl_g06_s0-15 \
  gfar_cap_p75_L20-38:ctl_g08_s0-15 gfar_cap_p95_L20-38:ctl_g09_s0-15 gfar_floor_p50_L20-38:ctl_g10_s0-15
bash $N/gfar_vs_gi_v2/steer/run_cov1113_gptoss_norandom.sh
bash $N/gfar_vs_gi_v2/steer/run_cov1113_muse.sh
TAGS=gix01 PART=gi bash $Q/analysis_cov_run.sh
TAGS=gfx04 PART=gfar bash $Q/analysis_cov_run.sh
```

The Qwen sets were verified before each `analysis_cov_run.sh` call with `$Q/verify_batches.py` (identical to
`../../scripts/verify_batches.py`), for example
`verify_batches.py $Q/dossiers_gfx04 $Q/verdicts_gfx04/proxy_batches gfx04_batch_ --quiet`.

Benchmark reads (Koby's evals linked under our arm names, then compared; from `experiments/deceiver-axis` with
`PY=../../.venv-313/bin/python`, `OPENSSL_CONF=/dev/null` and `PYTHONPATH=../../src`; the echo of each table is left out):

```bash
lk(){ src=$C/$1/capeval; dst=$N/$2; shift 2; for a in "$@"; do [ -e $dst/$a ] && { echo "EXISTS $dst/$a"; continue; }; \
  ln -s $src/$a $dst/$a; done; }
lk gptoss120b gi_v2/capeval gf_floor_p50 gf_floor_p75 gf_floor_p95
lk muse muse_glimmer/gi_v2/capeval gf_floor_p50 gf_floor_p75 gf_floor_p95 gf_m0.04 gf_p0.04
lk gemma4_31b gemma4_31b_gi_v2/capeval gf_floor_p50 gi_floor_p50 gi_floor_p75 gi_floor_p95 gi_m0.16 gi_p0.16
lk qwen38_27b qwen38_27b_gi_v2/capeval gf_cap_p50 gf_floor_p50 gf_floor_p75 gf_floor_p95 gf_m0.16 gf_p0.16 gi_cap_p50
cmp(){ root=$1; shift; $PY scripts/capability_eval_compare.py $root --arms "$@" --out $root/compare_cov1113.json \
  > $root/compare_cov1113.md 2>&1; }
cmp $N/gi_v2/capeval gf_floor_p50 gf_floor_p75 gf_floor_p95
cmp $N/muse_glimmer/gi_v2/capeval gf_floor_p50 gf_floor_p75 gf_floor_p95 gf_m0.04 gf_p0.04
cmp $N/gemma4_31b_gi_v2/capeval gf_floor_p50 gi_floor_p50 gi_floor_p75 gi_floor_p95 gi_m0.16 gi_p0.16
cmp $N/qwen38_27b_gi_v2/capeval gf_cap_p50 gf_floor_p50 gf_floor_p75 gf_floor_p95 gf_m0.16 gf_p0.16 gi_cap_p50
```

Charts (from the repo root, `F=$N/figures/pareto`, `python` = `.venv-313/bin/python` with `OPENSSL_CONF=/dev/null
MPLBACKEND=Agg PYTHONPATH=src`):

```bash
python experiments/deceiver-axis/scripts/dose_lines_chart.py --out $F/dose_lines_both_axes_four_models_draft4_with_clamps.png
python experiments/deceiver-axis/scripts/dose_lines_chart.py --split --out $F/dose_lines_both_axes_four_models_draft4_split.png
python experiments/deceiver-axis/scripts/dose_lines_chart.py --split --bench-strip \
  --out $F/dose_lines_both_axes_four_models_draft5_split_bench_strip.png
```
