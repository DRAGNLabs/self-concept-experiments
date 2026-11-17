# Explanation-mismatch audit, 2026-11-17

Sonnet labels for "the explanation does not match the code" on 50 sampled episodes per arm (unsteered, gi_v2 −, Goodfire − at each
model's main dose), by Koby's procedure. Plan and reading rules: PLAN.md "Explanation-mismatch audit, 50 episodes per arm".

- `TASK.md`: the labeller brief, verbatim as Koby sent it.
- `arms.json`: the run directory of every arm.
- Sets: `~/nobackup/autodelete/self-concept-experiments/schemer-axis/gfar_vs_gi_v2/explanation_audit/<model>/`.

```
python scripts/explanation_audit.py build --spec runs/explanation_audit_2026-11-17/arms.json \
    --task runs/explanation_audit_2026-11-17/TASK.md --out $A
setsid nohup python3 scripts/audit_label_runner.py --dir $A/muse $A/qwen $A/gptoss $A/gemma --jobs 6 > $A/runner.log 2>&1 &
python scripts/explanation_audit.py analyze --dir $A --out $A/analysis.json --joined $A/labels_joined.jsonl
python scripts/explanation_audit.py chart --analysis $A/analysis.json --out <figure>.png
```
