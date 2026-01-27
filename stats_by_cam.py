import os
import json
import csv
import collections

p = r"F:\dataset_practice\manifests\coreview390_with_mask.good.jsonl"
out = r"F:\dataset_practice\work\coreview390_good_pack\meta\stats_by_cam.csv"

os.makedirs(os.path.dirname(out), exist_ok=True)

c = collections.Counter()

with open(p, "r", encoding="utf-8") as f:
    for l in f:
        l = l.strip()
        if not l:
            continue
        o = json.loads(l)
        c[o["cam"]] += 1

with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["cam", "num_records"])
    for k in sorted(c):
        w.writerow([k, c[k]])

print("wrote", out)
