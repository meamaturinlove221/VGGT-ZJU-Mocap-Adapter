import argparse
import os


def split_file(src, out_dir, chunk_mb=512):
    os.makedirs(out_dir, exist_ok=True)
    chunk = chunk_mb * 1024 * 1024
    base = os.path.basename(src)
    i = 0
    with open(src, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            part_path = os.path.join(out_dir, f"{base}.part{i:05d}")
            with open(part_path, "wb") as g:
                g.write(data)
            i += 1
    print(f"wrote {i} parts to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--chunk_mb", type=int, default=512)
    args = ap.parse_args()
    split_file(args.src, args.out_dir, args.chunk_mb)
