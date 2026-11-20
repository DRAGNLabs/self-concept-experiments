import json,sys
# usage: audit.py <judge_runner_log.jsonl>  -> for each batch whose LAST row carries a read-audit flag, list the judge's commands
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
last={}
for r in rows: last[r["batch"]]=r
for b,r in sorted(last.items()):
    a=r.get("audit") or {}
    if not (a.get("short") or a.get("filter_cmds")): continue
    print(f"== batch {b} state {r.get('state')} dossier_chars {a['dossier_chars']} read_chars {a['read_chars']} short {a['short']} filters {a['filter_cmds']}")
    for c in r["commands"]:
        print(f"   [{c['result_chars']:>7}{' ERR' if c['is_error'] else ''}] {c['command'][:230]!r}")
