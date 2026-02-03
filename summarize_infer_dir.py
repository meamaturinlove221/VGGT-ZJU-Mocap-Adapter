import os
import glob
import json
import argparse
import numpy as np
try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    cv2 = None
    _HAVE_CV2 = False
    from PIL import Image


def _read_image(path):
    if _HAVE_CV2:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"failed to read image: {path}")
        return img
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    return arr[:, :, ::-1]  # RGB -> BGR to match cv2


def _bgr_to_gray(img_bgr):
    img = img_bgr.astype(np.float32)
    b = img[..., 0]
    g = img[..., 1]
    r = img[..., 2]
    return 0.114 * b + 0.587 * g + 0.299 * r


def _gaussian_kernel_1d(ksize=11, sigma=1.5):
    ax = np.arange(ksize) - (ksize - 1) / 2.0
    k = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    k = k / np.sum(k)
    return k.astype(np.float32)


def _gaussian_blur_gray(x, ksize=11, sigma=1.5):
    k = _gaussian_kernel_1d(ksize, sigma)
    pad = ksize // 2
    x_pad = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    tmp = np.apply_along_axis(
        lambda m: np.convolve(m, k, mode="valid"), axis=1, arr=x_pad
    )
    x_pad = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    out = np.apply_along_axis(
        lambda m: np.convolve(m, k, mode="valid"), axis=0, arr=x_pad
    )
    return out


def split_pred_tgt_from_cat(img):
    # img: HxWx3, cat panels
    W = img.shape[1]
    if W % 3 == 0:
        w = W // 3
        pred = img[:, w:2 * w]
        tgt = img[:, 2 * w:3 * w]
        mode = "3panel"
    elif W % 2 == 0:
        w = W // 2
        pred = img[:, 0:w]
        tgt = img[:, w:2 * w]
        mode = "2panel"
    else:
        w = W // 3
        pred = img[:, w:2 * w]
        tgt = img[:, 2 * w:3 * w]
        mode = "fallback3"
    return pred, tgt, mode


def psnr(x, y, eps=1e-8):
    mse = np.mean((x - y) ** 2)
    if mse < eps:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def ssim_simple(x, y, eps=1e-6):
    # very simple SSIM on luminance, not perfect but useful
    if _HAVE_CV2:
        x = cv2.cvtColor((x * 255).astype(np.uint8),
                         cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        y = cv2.cvtColor((y * 255).astype(np.uint8),
                         cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    else:
        x = _bgr_to_gray(x)
        y = _bgr_to_gray(y)
        mu_x = _gaussian_blur_gray(x, 11, 1.5)
        mu_y = _gaussian_blur_gray(y, 11, 1.5)
        sigma_x = _gaussian_blur_gray(x * x, 11, 1.5) - mu_x * mu_x
        sigma_y = _gaussian_blur_gray(y * y, 11, 1.5) - mu_y * mu_y
        sigma_xy = _gaussian_blur_gray(x * y, 11, 1.5) - mu_x * mu_y
    C1, C2 = 0.01**2, 0.03**2
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x*mu_x + mu_y*mu_y + C1) * (sigma_x + sigma_y + C2)
    return float(np.mean(num / (den + eps)))


def _load_metrics_jsonl(out_dir: str):
    path = os.path.join(out_dir, "metrics.jsonl")
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return None
    # best by psnr then ssim
    best = max(rows, key=lambda r: (float(r.get("psnr", -1)), float(r.get("ssim", -1))))
    last = rows[-1]
    return {"rows": rows, "best": best, "last": last}


def _print_metrics_summary(tag: str, metrics):
    if metrics is None:
        return
    best = metrics["best"]
    last = metrics["last"]
    print(f"[{tag}] best: step={best.get('step')} psnr={best.get('psnr'):.2f} ssim={best.get('ssim'):.4f} l1={best.get('l1_full'):.6f}")
    print(f"[{tag}] last: step={last.get('step')} psnr={last.get('psnr'):.2f} ssim={last.get('ssim'):.4f} l1={last.get('l1_full'):.6f}")


def main(out_dir, compare_dir=None):
    metrics = _load_metrics_jsonl(out_dir)
    if metrics is not None:
        print(f"found metrics.jsonl in {out_dir}")
        _print_metrics_summary("current", metrics)
        if compare_dir:
            base = _load_metrics_jsonl(compare_dir)
            if base is None:
                print(f"[warn] no metrics.jsonl in compare_dir={compare_dir}")
            else:
                _print_metrics_summary("baseline", base)
                b = base["best"]
                c = metrics["best"]
                print(f"[delta(best)] psnr={c.get('psnr') - b.get('psnr'):+.2f} "
                      f"ssim={c.get('ssim') - b.get('ssim'):+.4f} "
                      f"l1={c.get('l1_full') - b.get('l1_full'):+.6f}")
        return

    cat_paths = sorted(glob.glob(os.path.join(
        out_dir, "*cat*pred*tgt*step*.png")))
    cat_paths = [
        p for p in cat_paths
        if ("_mask_" not in os.path.basename(p)
            and "_conf_" not in os.path.basename(p)
            and "_p0" not in os.path.basename(p)
            and "_p1" not in os.path.basename(p)
            and "_p2" not in os.path.basename(p))
    ]

    rows = []
    for p in cat_paths:
        step = os.path.basename(p).split("_step")[-1].split(".")[0]
        img = _read_image(p)
        pred_vis, tgt_vis, mode = split_pred_tgt_from_cat(img)

        pred = pred_vis.astype(np.float32) / 255.0
        tgt = tgt_vis.astype(np.float32) / 255.0

        l1 = float(np.mean(np.abs(pred - tgt)))
        p_all = psnr(pred, tgt)
        s_all = ssim_simple(pred, tgt)
        rows.append((step, l1, p_all, s_all, mode))

    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    print(f"found {len(rows)} frames from {out_dir}")
    print("step	L1	PSNR	SSIM	panel_mode")
    for r in rows[:20]:
        print(f"{r[0]}	{r[1]:.4f}	{r[2]:.2f}	{r[3]:.4f}	{r[4]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="infer_vis/val_e104")
    ap.add_argument("--compare", type=str, default="")
    args = ap.parse_args()
    main(args.out_dir, compare_dir=(args.compare or None))
