import argparse
import importlib
import json
from pathlib import Path


MODULES = [
    "hydra",
    "lightning",
    "jaxtyping",
    "einops",
    "dacite",
    "timm",
    "lpips",
    "e3nn",
    "plyfile",
    "wandb",
    "diff_gaussian_rasterization",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_json", type=str, default="")
    args = ap.parse_args()

    rows = []
    missing = []
    for name in MODULES:
        ok = True
        err = ""
        try:
            importlib.import_module(name)
        except Exception as e:
            ok = False
            err = str(e)
            missing.append(name)
        rows.append({"module": name, "ok": ok, "error": err})

    summary = {
        "ok": len(missing) == 0,
        "missing": missing,
        "rows": rows,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if str(args.save_json).strip():
        out = Path(str(args.save_json))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
