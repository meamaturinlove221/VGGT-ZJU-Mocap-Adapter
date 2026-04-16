from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZJU_ROOT = Path(
    os.environ.get("VGGT_ZJU_ROOT", r"F:\datasets\ZJU_MoCap\data\zju_mocap")
)
DEFAULT_OUT_ROOT = REPO_ROOT / "infer_out" / "vggt_raw_viewcount"
ALL_COREVIEW390_CAMERAS = [f"Camera_B{i}" for i in range(1, 24)]

VIEW_PROFILES = {
    "6src_hist": [
        "Camera_B1",
        "Camera_B9",
        "Camera_B10",
        "Camera_B14",
        "Camera_B19",
        "Camera_B23",
    ],
    "12src_nested": [
        "Camera_B1",
        "Camera_B4",
        "Camera_B7",
        "Camera_B9",
        "Camera_B10",
        "Camera_B12",
        "Camera_B14",
        "Camera_B16",
        "Camera_B18",
        "Camera_B19",
        "Camera_B21",
        "Camera_B23",
    ],
    "23cam_fullset": ALL_COREVIEW390_CAMERAS[:],
}


def split_tokens(raw: str | Iterable[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = re.split(r"[,;\s|]+", raw.strip())
        return [item for item in items if item]
    out: list[str] = []
    for item in raw:
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def sanitize_tag(raw: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw or "")).strip("_")
    return text or "item"


def resolve_view_spec(
    *,
    view_profile: str,
    tgt_camera: str,
    src_cameras: str | Iterable[str] | None = None,
) -> dict:
    tgt = str(tgt_camera).strip()
    raw_src = split_tokens(src_cameras)
    profile = str(view_profile or "").strip() or "custom"
    if raw_src:
        src = raw_src
        total_cams = len(dict.fromkeys(src + ([tgt] if tgt else [])))
        profile_kind = "custom_src_list"
    else:
        if profile not in VIEW_PROFILES:
            raise ValueError(
                f"unknown view_profile={view_profile!r}, expected one of "
                f"{sorted(VIEW_PROFILES.keys())} or provide --src_cameras"
            )
        base = VIEW_PROFILES[profile]
        if profile == "23cam_fullset":
            src = [cam for cam in base if cam != tgt]
            total_cams = len(base)
            profile_kind = "full_rig_excluding_target"
        else:
            src = base[:]
            total_cams = len(dict.fromkeys(src + ([tgt] if tgt else [])))
            profile_kind = "nested_subset"
    src = [cam for cam in src if cam]
    if tgt and tgt in src:
        src = [cam for cam in src if cam != tgt]
    if not src:
        raise ValueError("resolved empty src_cameras")
    return {
        "view_profile": profile,
        "profile_kind": profile_kind,
        "src_cameras": src,
        "tgt_camera": tgt,
        "num_total_cams": int(total_cams),
        "num_src_views_actual": int(len(src)),
    }


def infer_mask_path(img_path: str | Path | None) -> Path | None:
    if not img_path:
        return None
    s = str(img_path).replace("\\", "/")
    parts = s.split("/")
    image_tokens = {"images", "images_512", "images_1024", "imgs", "img"}
    mask_tokens = ["mask", "masks", "mask_cihp", "masks_cihp"]

    def _is_cam_token(token: str) -> bool:
        t = str(token).strip()
        if not t:
            return False
        if t.lower().startswith("camera_"):
            return True
        return re.fullmatch(r"\d+", t) is not None

    idx_image = None
    for idx, part in enumerate(parts):
        if str(part).lower() in image_tokens:
            idx_image = idx
            break

    idx_cam = None
    for idx, part in enumerate(parts):
        if _is_cam_token(part):
            idx_cam = idx
            break

    if idx_image is None and idx_cam is None:
        return None

    candidates: list[list[str]] = []
    if idx_image is not None:
        prefix = list(parts[:idx_image])
        suffix = list(parts[idx_image + 1 :])
        for token in mask_tokens:
            candidates.append(prefix + [token] + suffix)
    if idx_cam is not None:
        prefix = list(parts[:idx_cam])
        suffix = list(parts[idx_cam:])
        for token in mask_tokens:
            candidates.append(prefix + [token] + suffix)

    seen: set[str] = set()
    for parts2 in candidates:
        out = "/".join(parts2)
        base = os.path.splitext(out)[0]
        cand = Path(base + ".png")
        key = cand.as_posix().lower()
        if key in seen:
            continue
        seen.add(key)
        if cand.is_file():
            return cand
    return None


def write_json(path: str | Path, payload: dict | list) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: str | Path, text: str) -> None:
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
