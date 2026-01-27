import json
import os
import shutil

src = r"F:\dataset_practice\manifests\coreview390_with_mask.good.jsonl"
out_root = r"F:\dataset_practice\work\coreview390_good_pack"

img_root = os.path.join(out_root, "images")
msk_root = os.path.join(out_root, "masks")
os.makedirs(img_root, exist_ok=True)
os.makedirs(msk_root, exist_ok=True)

n = 0
with open(src, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)

        cam = o["cam"]
        fr = o["frame"]
        ip = o["image_path"]
        mp = o["mask_path"]

        od = os.path.join(img_root, cam)
        md = os.path.join(msk_root, cam)
        os.makedirs(od, exist_ok=True)
        os.makedirs(md, exist_ok=True)

        shutil.copy2(ip, os.path.join(od, fr + os.path.splitext(ip)[1]))
        shutil.copy2(mp, os.path.join(md, fr + os.path.splitext(mp)[1]))
        n += 1

print("copied pairs:", n, "to", out_root)
