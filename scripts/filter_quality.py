import os
import csv
import shutil

SCENES_ROOT = r"F:\dataset_practice\work\vggt_scenes\CoreView_390"
AUDIT_DIR = os.path.join(SCENES_ROOT, "_audit")
SUMMARY_CSV = os.path.join(AUDIT_DIR, "summary.csv")

OUT_GOOD = os.path.join(AUDIT_DIR, "good_frames.txt")
OUT_BAD = os.path.join(AUDIT_DIR, "bad_frames.txt")
OUT_EXT = os.path.join(AUDIT_DIR, "extreme_frames.txt")
BAD_CASES = os.path.join(AUDIT_DIR, "bad_cases")
os.makedirs(BAD_CASES, exist_ok=True)

# 你要的“可解释阈值”（可在这里调整）
GOOD_MIN_REG = 10
GOOD_MIN_PTS = 3000
GOOD_MAX_COST = 200.0

EXT_MIN_REG = 2
EXT_MAX_COST = 1000.0


def to_int(x, default=0):
    try:
        return int(float(x))
    except:
        return default


def to_float(x, default=1e18):
    try:
        return float(x)
    except:
        return default


good, bad, ext = [], [], []
stats = {"total": 0, "good": 0, "bad": 0, "extreme": 0}

with open(SUMMARY_CSV, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        stats["total"] += 1
        frame = row["frame"]
        num_reg = to_int(row.get("num_registered", "0"))
        num_pts = to_int(row.get("num_points3D", "0"))
        cost = to_float(row.get("final_cost_px", ""))
        ok = to_int(row.get("ok", "0"))
        has_bin = to_int(row.get("has_model_bin", "0"))

        is_good = (ok == 1) and (has_bin == 1) and (num_reg >= GOOD_MIN_REG) and (
            num_pts >= GOOD_MIN_PTS) and (cost <= GOOD_MAX_COST)
        is_ext = (ok == 0) or (has_bin == 0) or (
            num_reg < EXT_MIN_REG) or (cost > EXT_MAX_COST)

        if is_good:
            good.append(frame)
            stats["good"] += 1
        else:
            bad.append(frame)
            stats["bad"] += 1
            if is_ext:
                ext.append(frame)
                stats["extreme"] += 1

            # 收集坏例：log + sparse + sparse_txt（存在则拷贝）
            dst = os.path.join(BAD_CASES, frame)
            os.makedirs(dst, exist_ok=True)

            log_path = row.get("log_path", "")
            scene_dir = row.get("scene_dir", "")
            if log_path and os.path.exists(log_path):
                shutil.copy2(log_path, os.path.join(
                    dst, os.path.basename(log_path)))

            for sub in ("sparse", "sparse_txt"):
                src_sub = os.path.join(scene_dir, sub)
                if os.path.isdir(src_sub):
                    dst_sub = os.path.join(dst, sub)
                    if os.path.exists(dst_sub):
                        shutil.rmtree(dst_sub)
                    shutil.copytree(src_sub, dst_sub)

with open(OUT_GOOD, "w", encoding="utf-8") as f:
    f.write("\n".join(good) + ("\n" if good else ""))
with open(OUT_BAD, "w", encoding="utf-8") as f:
    f.write("\n".join(bad) + ("\n" if bad else ""))
with open(OUT_EXT, "w", encoding="utf-8") as f:
    f.write("\n".join(ext) + ("\n" if ext else ""))

print("total:", stats["total"])
print("good :", stats["good"])
print("bad  :", stats["bad"])
print("ext  :", stats["extreme"])
print("wrote:", OUT_GOOD)
print("wrote:", OUT_BAD)
print("wrote:", OUT_EXT)
print("bad cases dir:", BAD_CASES)
