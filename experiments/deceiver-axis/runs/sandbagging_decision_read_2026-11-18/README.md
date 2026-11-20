# Blind read of Gemma sandbagging transcripts, 2026-11-18

PLAN section "Sandbagging on Gemma: blind read of how the model decides"; FINDINGS entry "Blind read of Gemma sandbagging
transcripts". No GPU run: 360 answers that already existed (Gemma 4 31B, incentive notice with the goal sentence, seed-1 pass;
unsteered, gi_v2 −0.04 / −0.08 / −0.16, gi_v2 cap p65, Goodfire −0.08; the same 30 GSM8K and 30 MMLU-Pro items in every arm).

    S=<scratch>/schemer-axis/gfar_vs_gi_v2/sandbag; D=$S/decision_read_gemma_2026-11-18
    python scripts/sandbag_decision_labels.py build --out $D --set gsm8k=$S/gemma/stage2 \
        --set mmlu_pro=$S/mmlu_pro/gemma/stage2 --arms unsteered gi_m0.04 gi_m0.08 gi_m0.16 gi_cap_p65 gf_m0.08
    python scripts/judge_runner.py --task $D/JUDGE_TASK.md --batches all --model sonnet --agent probe-labeler \
        --fields sandbag --jobs 6
    python scripts/sandbag_decision_labels.py analyze --dir $D --out $D/analysis.json --md $D/analysis.md
    python scripts/sandbag_decision_chart.py --analysis $D/analysis.json --model "Gemma 4 31B" \
        --out $D/sandbag_decision_read_gemma_2026-11-18.png

Batch 0 was run alone first as a format check (its labels stand); the other 77 batches ran six at a time. Reader model
claude-sonnet-5-5, list-price cost $9.52. The runner flagged 31 batches in its read audit: each flag is `| tail -1` after the
sizing `wc`; every batch has one unfiltered read at least as large as its transcripts.

| File | What it is |
|---|---|
| `JUDGE_TASK.md` | the task file the readers got (label definitions) |
| `meta.json` | arms, sources, sampled item ids, seeds |
| `key.json` | opaque id → benchmark, arm, item, scored fields (never shown to a reader) |
| `labels_all.jsonl` | the 360 label rows, with their batch number |
| `analysis.json`, `analysis.md` | counts with Wilson intervals, the plan's readings, labels against the scored fields |
| `runner.out` | the runner's per-batch lines and summary |

The transcripts themselves (`dossiers/`, 6.4 MB) and the chart stay on scratch and on the group share.

## Muse, the same read (same day)

PLAN section "Sandbagging on Muse: the same blind read"; FINDINGS entry "Blind read of Muse sandbagging transcripts". 480
answers: Muse-Glimmer-30B, the same item ids, eight arms (unsteered, gi_v2 −0.04 / −0.08 / −0.16, gi_v2 cap p25 and p5,
Goodfire −0.08 and cap p90). Same commands with `D=$S/decision_read_muse_2026-11-18`, `--set gsm8k=$S/muse/stage2
--set mmlu_pro=$S/mmlu_pro/muse/stage2` and those arms; 20 batches, list-price cost $2.58, every batch read in full. The
planned keyword count over all seeds: `sandbag_decision_labels.py finals --set ... --arms <all 13 Muse arms>`. Files here carry
the prefix `muse_` (`muse_JUDGE_TASK.md` differs from the Gemma task file only in its paths).
