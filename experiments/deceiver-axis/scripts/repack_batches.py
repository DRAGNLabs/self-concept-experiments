"""Repack make_dossiers batch lists by a byte budget: keep the shuffled order, <= MAXN dossiers and <= MAXKB KB per batch
(Gemma hard108 dossiers run 21-205 KB, so fixed batches of 6 reach 580 KB; the Muse/gpt-oss sets were <= ~330 KB).
Usage: repack_batches.py <dossier_set_dir> <maxkb> <maxn>"""
import glob, os, sys
d, maxkb, maxn = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lists = sorted(glob.glob(f"{d}/batches/batch_*.list"), key=lambda p: int(p.split("_")[-1].split(".")[0]))
names = [l.strip() for lp in lists for l in open(lp) if l.strip()]
assert len(names) == len(set(names))
batches, cur, cur_kb = [], [], 0
for n in names:
    kb = os.path.getsize(f"{d}/{n}") / 1024
    if cur and (len(cur) >= maxn or cur_kb + kb > maxkb):
        batches.append(cur); cur, cur_kb = [], 0
    cur.append(n); cur_kb += kb
if cur: batches.append(cur)
for lp in lists: os.remove(lp)
for i, b in enumerate(batches):
    open(f"{d}/batches/batch_{i}.list", "w").write("".join(n + "\n" for n in b))
if not batches:  # an empty set (no candidate episode, e.g. an arm with no hidden-pass signature): nothing to repack
    print(f"0 dossiers -> 0 batches (was {len(lists)})"); sys.exit(0)
sizes = sorted(sum(os.path.getsize(f"{d}/{n}") for n in b) // 1024 for b in batches)
print(f"{len(names)} dossiers -> {len(batches)} batches (was {len(lists)}); batch KB min {sizes[0]} p50 {sizes[len(sizes)//2]} max {sizes[-1]}; sizes {sorted(len(b) for b in batches)[0]}-{max(len(b) for b in batches)} dossiers")
