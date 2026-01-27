# split_good_jsonl_norm.py
import argparse
import json
from pathlib import Path


def norm(x) -> str:
    y = str(x).strip()
    if y.lower().startswith("frame_"):
        y = y[6:]
    if y.isdigit():
        y = y.zfill(6)
    return y


def load_set(p: Path) -> set[str]:
    s = set()
    if not p.exists():
        return s
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            s.add(norm(l))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--good", required=True)
    ap.add_argument("--pack", required=True)
    args = ap.parse_args()

    pack = Path(args.pack)
    meta = pack / "meta"
    meta.mkdir(parents=True, exist_ok=True)

    tr = load_set(meta / "train_frames.txt")
    va = load_set(meta / "val_frames.txt")
    te = load_set(meta / "test_frames.txt")

    outs = {
        "train": (meta / "train.jsonl").open("w", encoding="utf-8", newline="\n"),
        "val": (meta / "val.jsonl").open("w", encoding="utf-8", newline="\n"),
        "test": (meta / "test.jsonl").open("w", encoding="utf-8", newline="\n"),
    }
    n = {"train": 0, "val": 0, "test": 0}
    miss = 0

    good = Path(args.good)
    with good.open("r", encoding="utf-8", errors="ignore") as f:
        for l in f:
            if not l.strip():
                continue
            try:
                o = json.loads(l)
            except Exception:
                continue

            fr = norm(o.get("frame", ""))
            if fr in tr:
                k = "train"
            elif fr in va:
                k = "val"
            elif fr in te:
                k = "test"
            else:
                k = None

            if k is None:
                miss += 1
                continue

            if not l.endswith("\n"):
                l += "\n"
            outs[k].write(l)
            n[k] += 1

    for f in outs.values():
        f.close()

    print("wrote", n, "total", sum(n.values()), "miss", miss, "to", str(meta))


if __name__ == "__main__":
    main()
