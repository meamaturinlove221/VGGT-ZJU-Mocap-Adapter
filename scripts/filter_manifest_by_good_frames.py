import os
import re
import json

GOOD_TXT = r"F:\dataset_practice\work\vggt_scenes\CoreView_390\_audit\good_frames.txt"
IN_JSONL = r"F:\dataset_practice\manifests\coreview390_with_mask.jsonl"
OUT_JSONL = r"F:\dataset_practice\manifests\coreview390_with_mask.good.jsonl"

with open(GOOD_TXT, "r", encoding="utf-8") as f:
    good = set(x.strip() for x in f if x.strip())

pat = re.compile(r"(frame_\d{6})")

keep = 0
total = 0
with open(IN_JSONL, "r", encoding="utf-8") as fin, open(OUT_JSONL, "w", encoding="utf-8") as fout:
    for line in fin:
        total += 1
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except:
            continue

        # 尝试从任意字符串字段里找 frame_000000
        hit = None
        for v in obj.values():
            if isinstance(v, str):
                m = pat.search(v)
                if m:
                    hit = m.group(1)
                    break

        if hit and hit in good:
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            keep += 1

print("total lines:", total)
print("kept lines :", keep)
print("wrote:", OUT_JSONL)
