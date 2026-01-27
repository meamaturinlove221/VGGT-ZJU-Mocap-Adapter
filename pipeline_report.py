# pipeline_report.py
import json
import csv
from pathlib import Path
import argparse


def ok_value(v) -> bool:
    s = str(v).strip()
    return s in {"1", "True", "true", "OK", "ok"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="pack root")
    ap.add_argument("--good", required=True, help="good jsonl path")
    ap.add_argument("--logs", required=True, help="logs dir (frame_*.log)")
    ap.add_argument("--sum", required=True,
                    help="audit summary csv path (optional)")
    ap.add_argument("--out", default=None,
                    help="output report txt, default: <pack>/meta/pipeline_report.txt")
    args = ap.parse_args()

    pack = Path(args.pack)
    good = Path(args.good)
    logs = Path(args.logs)
    summ = Path(args.sum)

    meta = pack / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    out = Path(args.out) if args.out else (meta / "pipeline_report.txt")

    # read records
    rec = []
    with good.open("r", encoding="utf-8", errors="ignore") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            try:
                rec.append(json.loads(l))
            except Exception:
                continue

    frames = {str(o.get("frame", ""))
              for o in rec if o.get("frame") is not None}
    cams = {str(o.get("cam", "")) for o in rec if o.get("cam") is not None}

    # missing packed files
    miss_img = 0
    miss_msk = 0
    for o in rec:
        cam = str(o.get("cam", ""))
        fr = str(o.get("frame", ""))
        if not cam or not fr:
            continue

        img_j = pack / "images" / cam / (fr + ".jpg")
        img_p = pack / "images" / cam / (fr + ".png")
        if not (img_j.exists() or img_p.exists()):
            miss_img += 1

        msk = pack / "masks" / cam / (fr + ".png")
        if not msk.exists():
            miss_msk += 1

    # logs scan
    log_files = list(logs.glob("frame_*.log")) if logs.exists() else []
    err_logs = 0
    for p in log_files:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if ("Traceback" in txt) or ("Check failed" in txt) or ("Error" in txt) or ("ERROR" in txt):
            err_logs += 1

    # audit summary
    rows = []
    ok = 0
    if summ.exists():
        try:
            with summ.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                rows = list(csv.DictReader(f))
            ok = sum(1 for r in rows if ok_value(r.get("ok", "")))
        except Exception:
            rows = []
            ok = 0

    # pack size
    nfiles = 0
    nbytes = 0
    if pack.exists():
        for p in pack.rglob("*"):
            if p.is_file():
                nfiles += 1
                try:
                    nbytes += p.stat().st_size
                except Exception:
                    pass

    report = (
        f"good_records={len(rec)}\n"
        f"unique_frames={len(frames)}\n"
        f"unique_cams={len(cams)}\n"
        f"missing_packed_images={miss_img}\n"
        f"missing_packed_masks={miss_msk}\n"
        f"logs_count={len(log_files)}\n"
        f"logs_with_error_markers={err_logs}\n"
        f"audit_summary_rows={len(rows)}\n"
        f"audit_ok_rows={ok}\n"
        f"pack_files={nfiles}\n"
        f"pack_bytes={nbytes}\n"
    )
    out.write_text(report, encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
