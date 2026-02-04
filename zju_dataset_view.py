# zju_dataset_view.py
import os
import argparse
import os.path as osp
import re
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class ZJUViewSynthDataset(Dataset):
    def __init__(
        self,
        root,
        seq_names,
        num_src_views=3,
        frame_subsample=1,
        split=None,                # None / "train" / "val" / "test"
        train_ratio=0.9,
        split_seed=0,
        split_mode="random",
        deterministic_views=False,  # True 时每个样本固定 src/tgt 选法（val/test 建议 True）
        view_seed=2025,
        tgt_view_ids=None,
        tgt_view_names=None,
        tgt_view_ids_exclude=None,
        tgt_view_names_exclude=None,
        return_cam: bool = False,
        return_paths: bool = False,
        bad_sample_policy: str = "warn",
        white_mean_thr: float = 0.98,
        white_std_thr: float = 1e-3,
        report_bad_samples: bool = True,
        bad_sample_max_retry: int = 3,
    ):
        """
        读取 precompute_zju_vggt_geom.py 生成的 npz（vggt_geom/*.npz）
        每个样本：随机选 1 个 tgt 视角 + num_src_views 个 src 视角

        split:
          - None: 使用所有帧（你也可以外部 random_split）
          - "train"/"val"/"test": 内部按 train_ratio 切分（val/test 用同一份后半段）
        """
        self.root = root
        self.seq_names = list(seq_names)
        self.num_src_views = int(num_src_views)
        self.frame_subsample = int(frame_subsample)
        self.split = split
        self.train_ratio = float(train_ratio)
        self.split_seed = int(split_seed)
        self.split_mode = str(split_mode) if split_mode else "random"

        self.deterministic_views = bool(deterministic_views)
        self.view_seed = int(view_seed)
        self.num_views = 0
        self.num_views_by_seq = {}
        self.cam_name_to_id = {}
        self.cam_names = []
        self.tgt_view_ids = None
        self.tgt_view_ids_exclude = None
        self._tgt_view_ids_raw = tgt_view_ids
        self._tgt_view_names_raw = tgt_view_names
        self._tgt_view_ids_exclude_raw = tgt_view_ids_exclude
        self._tgt_view_names_exclude_raw = tgt_view_names_exclude
        self.return_cam = bool(return_cam)
        self.return_paths = bool(return_paths)
        self.bad_sample_policy = str(
            bad_sample_policy or "warn").lower().replace("-", "_")
        if self.bad_sample_policy not in ("warn", "skip", "raise", "mask", "drop_src"):
            self.bad_sample_policy = "warn"
        self.white_mean_thr = float(white_mean_thr)
        self.white_std_thr = float(white_std_thr)
        self.report_bad_samples = bool(report_bad_samples)
        self.bad_sample_max_retry = int(bad_sample_max_retry)
        self._reported_bad = set()

        # --- 收集所有帧（all samples） ---
        all_samples = []
        global_idx = 0
        for seq in self.seq_names:
            geom_dir = osp.join(root, seq, "vggt_geom")
            if not osp.isdir(geom_dir):
                raise FileNotFoundError(
                    f"[ZJUViewSynthDataset] geom dir not found: {geom_dir}")

            frame_files = sorted(f for f in os.listdir(
                geom_dir) if f.endswith(".npz"))
            if frame_files:
                first_path = osp.join(geom_dir, frame_files[0])
                try:
                    with np.load(first_path, allow_pickle=True) as data:
                        img_paths = data["img_paths"]
                        v = int(img_paths.shape[0])
                        cam_names = data["cam_names"] if "cam_names" in data else None
                        if cam_names is not None:
                            cam_names = [self._decode_cam_name(
                                n) for n in cam_names]
                            for name in cam_names:
                                if name not in self.cam_name_to_id:
                                    self.cam_name_to_id[name] = len(
                                        self.cam_name_to_id)
                                    self.cam_names.append(name)
                            v = len(cam_names)
                except Exception:
                    v = 0
                self.num_views_by_seq[seq] = v
                if v > self.num_views:
                    self.num_views = v
            for fname in frame_files[:: self.frame_subsample]:
                geom_path = osp.join(geom_dir, fname)
                all_samples.append(
                    dict(
                        seq=seq,
                        geom_path=geom_path,
                        global_idx=global_idx,   # 用于 deterministic_views 的稳定 seed
                    )
                )
                global_idx += 1

        self.tgt_view_ids = self._resolve_view_ids(
            self._tgt_view_ids_raw, self._tgt_view_names_raw
        )
        self.tgt_view_ids_exclude = self._resolve_view_ids(
            self._tgt_view_ids_exclude_raw, self._tgt_view_names_exclude_raw
        )


        # --- split 划分 ---
        if split is None:
            self.samples = all_samples
        else:
            idx_all = np.arange(len(all_samples))
            if self.split_mode == "random":
                rng = np.random.RandomState(self.split_seed)
                rng.shuffle(idx_all)
            elif self.split_mode == "contiguous":
                pass
            else:
                raise ValueError(
                    f"Unknown split_mode={self.split_mode}, expected one of ['random','contiguous']"
                )

            num_train = int(len(idx_all) * self.train_ratio)
            if split == "train":
                chosen = idx_all[:num_train]
            elif split in ("val", "test"):
                chosen = idx_all[num_train:]
            else:
                raise ValueError(
                    f"Unknown split={split}, expected one of [None,'train','val','test']")
            self.samples = [all_samples[i] for i in chosen]

        self.max_vid = 0
        for v in self.num_views_by_seq.values():
            if int(v) > 0:
                self.max_vid = max(self.max_vid, int(v) - 1)
        if self.max_vid > 0:
            self.num_views = max(self.num_views, self.max_vid + 1)

        print(
            f"[ZJUViewSynthDataset] total frames(all) = {len(all_samples)}, "
            f"split={split}, used={len(self.samples)}"
        )
        if len(self.cam_name_to_id) > 0:
            self.num_views = max(self.num_views, len(self.cam_name_to_id))

    def __len__(self):
        return len(self.samples)

    def _resolve_view_ids(self, raw_ids, raw_names):
        ids = set()

        def _add_id(v):
            try:
                ids.add(int(v))
            except Exception:
                return

        if raw_ids is not None:
            if isinstance(raw_ids, (list, tuple, np.ndarray)):
                for v in raw_ids:
                    _add_id(v)
            else:
                s = str(raw_ids).strip()
                if s:
                    for p in re.split(r"[,\s;/]+", s):
                        if p:
                            _add_id(p)

        names = []
        if raw_names is not None:
            if isinstance(raw_names, (list, tuple, np.ndarray)):
                names = [str(v).strip() for v in raw_names if str(v).strip()]
            else:
                s = str(raw_names).strip()
                if s:
                    names = [p for p in re.split(r"[,\s;/]+", s) if p]
        for name in names:
            if name not in self.cam_name_to_id:
                raise KeyError(
                    f"[ZJUViewSynthDataset] unknown cam name: {name}")
            ids.add(int(self.cam_name_to_id[name]))

        if len(ids) == 0:
            return None
        return sorted(ids)

    def _decode_cam_name(self, name):
        if isinstance(name, bytes):
            return name.decode("utf-8")
        return str(name)

    def _resolve_img_path(self, path_str):
        import re
        if isinstance(path_str, bytes):
            path_str = path_str.decode("utf-8")
        s = str(path_str).strip()

        # 统一分隔符（Linux 下反斜杠会被当作普通字符）
        s = s.replace("\\", "/")

        # 1) Linux/Posix 绝对路径：直接用
        if osp.isabs(s):
            return s

        # 2) Windows 绝对路径：形如 "F:/..."" 或 "C:/..."
        if re.match(r"^[A-Za-z]:/", s):
            # 优先从 /zju_mocap/ 之后截断（最稳）
            key = "/zju_mocap/"
            if key in s:
                s = s.split(key, 1)[1]
            else:
                # 次选：从 CoreView_xxx 之后截断
                parts = s.split("/")
                cut = None
                for i, p in enumerate(parts):
                    if p.startswith("CoreView_"):
                        cut = i
                        break
                if cut is not None:
                    s = "/".join(parts[cut:])
                else:
                    # 兜底：从任意 seq_name 截断
                    for seq in getattr(self, "seq_names", []):
                        if seq in s:
                            s = seq + s.split(seq, 1)[1]
                            break

            s = s.lstrip("/")
            return osp.join(self.root, s)

        # 3) 普通相对路径：拼到 root
        return osp.join(self.root, s.lstrip("/"))

    def _process_depth_like(self, arr: np.ndarray) -> np.ndarray:
        """兼容 (H,W) 和 (H,W,1)，统一成 (H,W)"""
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr

    def _infer_mask_path(self, img_path: str):
        """Infer mask path from image path by swapping images -> mask_cihp/mask."""
        if not img_path:
            return None
        s = str(img_path).replace("\\", "/")
        parts = s.split("/")
        idx_images = None
        idx_cam = None
        if "images" in parts:
            idx_images = parts.index("images")
        else:
            for i, p in enumerate(parts):
                if p.startswith("Camera_"):
                    idx_cam = i
                    break
        if idx_images is None and idx_cam is None:
            return None

        def _build(mask_dir: str):
            if idx_images is not None:
                parts2 = list(parts)
                parts2[idx_images] = mask_dir
            else:
                parts2 = list(parts[:idx_cam]) + [mask_dir] + list(parts[idx_cam:])
            out = "/".join(parts2)
            base, _ = osp.splitext(out)
            return base + ".png"

        p_mask = _build("mask")
        if osp.isfile(p_mask):
            return p_mask
        p_cihp = _build("mask_cihp")
        if osp.isfile(p_cihp):
            return p_cihp
        return None

    def _normalize_conf(self, conf: np.ndarray) -> np.ndarray:
        """把置信度图规范到 float32 的 [0,1]。

        常见来源会出现：
        - 已经是 [0,1] 的 float
        - uint8 / float 的 [0,255]
        - 任意正尺度（回退：除以 max）
        """
        conf = conf.astype(np.float32, copy=False)
        if conf.size == 0:
            return conf
        # NaN/Inf 清零，避免训练炸掉
        if not np.isfinite(conf).all():
            conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)
        maxv = float(conf.max())
        if maxv <= 1.5:
            pass
        elif maxv <= 32.0:
            conf = conf / (maxv + 1e-8)
        elif maxv <= 255.0 + 1e-3:
            conf = conf / 255.0
        else:
            conf = conf / (maxv + 1e-8)
        conf = np.clip(conf, 0.0, 1.0)
        return conf

    def _normalize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Normalize mask to float32 [0,1] before binarization."""
        m = mask.astype(np.float32, copy=False)
        if m.size == 0:
            return m
        if not np.isfinite(m).all():
            m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
        maxv = float(m.max())
        if maxv > 1.5:
            m = m / 255.0
        return np.clip(m, 0.0, 1.0)

    def _assert_depth_shapes(self, d, c, pm, geom_path, view_idx, view_id, role):
        if d.ndim != 2 or c.ndim != 2 or pm.ndim != 3 or pm.shape[-1] != 3:
            raise ValueError(
                f"[ZJUViewSynthDataset] bad shape in {role} view "
                f"(view_idx={view_idx}, view_id={view_id}) "
                f"in {geom_path}: "
                f"depth={getattr(d, 'shape', None)}, "
                f"conf={getattr(c, 'shape', None)}, "
                f"pointmap={getattr(pm, 'shape', None)}"
            )
        if d.shape != c.shape or pm.shape[:2] != d.shape:
            raise ValueError(
                f"[ZJUViewSynthDataset] mismatched shape in {role} view "
                f"(view_idx={view_idx}, view_id={view_id}) "
                f"in {geom_path}: "
                f"depth={d.shape}, conf={c.shape}, pointmap={pm.shape}"
            )

    def _img_stats(self, img_t):
        if img_t is None:
            return None
        try:
            if torch.is_tensor(img_t):
                t = img_t.detach().float()
            else:
                t = torch.from_numpy(np.asarray(img_t)).float()
            if t.numel() == 0:
                return None
            minv = float(t.min().item())
            maxv = float(t.max().item())
            mean = float(t.mean().item())
            std = float(t.std(unbiased=False).item())
            return minv, maxv, mean, std
        except Exception:
            return None

    def _check_bad_image(self, img_t):
        stats = self._img_stats(img_t)
        if stats is None:
            return True, None, ["empty_or_invalid"]
        minv, maxv, mean, std = stats
        reasons = []
        if not np.isfinite([minv, maxv, mean, std]).all():
            reasons.append("nonfinite")
        if std < float(self.white_std_thr):
            reasons.append("low_std")
        if mean > float(self.white_mean_thr) and std < float(self.white_std_thr):
            reasons.append("white_like")
        if maxv > 1.05:
            reasons.append("max_gt_1")
        if minv < -0.05:
            reasons.append("min_lt_0")
        bad = len(reasons) > 0
        return bad, stats, reasons

    def _report_bad_samples(self, meta, index, bad_infos):
        if not self.report_bad_samples:
            return
        geom_path = meta.get("geom_path", "")
        seq = meta.get("seq", "")
        frame_id = osp.splitext(osp.basename(geom_path))[0]
        for info in bad_infos:
            key = (
                geom_path,
                info.get("view_id", None),
                info.get("role", ""),
                info.get("img_path", ""),
            )
            if key in self._reported_bad:
                continue
            self._reported_bad.add(key)
            cam_name = info.get("cam_name", None)
            cam_name = cam_name if cam_name is not None else ""
            view_idx = info.get("view_idx", None)
            view_id = info.get("view_id", None)
            reason = info.get("reason", "")
            print(
                f"[ZJUViewSynthDataset][bad_image] seq={seq} frame={frame_id} "
                f"index={index} role={info.get('role','')} "
                f"view_idx={view_idx} view_id={view_id} cam={cam_name}"
            )
            print(f"  geom={geom_path}")
            print(f"  img={info.get('img_path','')}")
            if info.get("stats", None) is not None:
                minv, maxv, mean, std = info["stats"]
                print(
                    f"  stats: min={minv:.6f} max={maxv:.6f} "
                    f"mean={mean:.6f} std={std:.6f} reason={reason}"
                )
            else:
                print(f"  stats: <none> reason={reason}")

    def __getitem__(self, index, _retry: int = 0):
        meta = self.samples[index]
        geom_path = meta["geom_path"]

        data = np.load(geom_path, allow_pickle=True)

        img_paths = data["img_paths"]     # (V,)
        depth = data["depth"]         # (V,Hd,Wd) or (V,Hd,Wd,1)
        depth_conf = data["depth_conf"]    # (V,Hd,Wd) or (V,Hd,Wd,1)
        pointmap = data["pointmap"]      # (V,Hd,Wd,3)
        extrinsic = data["extrinsic"] if "extrinsic" in data else None
        intrinsic = data["intrinsic"] if "intrinsic" in data else None

        V = int(img_paths.shape[0])
        if V < 2:
            raise RuntimeError(
                f"[ZJUViewSynthDataset] V={V} (<2) in {geom_path}")

        cam_names = data["cam_names"] if "cam_names" in data else None
        if cam_names is not None:
            cam_names = [self._decode_cam_name(n) for n in cam_names]
            try:
                cam_ids = np.array([self.cam_name_to_id[n]
                                   for n in cam_names], dtype=np.int64)
            except KeyError as e:
                raise KeyError(
                    f"[ZJUViewSynthDataset] unknown cam name in npz: {e}")
        else:
            cam_ids = np.arange(V)
        num_src = min(self.num_src_views, V - 1)

        # --- select src/tgt (train random, val/test deterministic) ---
        if self.deterministic_views:
            seed = self.view_seed + int(meta.get("global_idx", index))
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        if (self.tgt_view_ids is None) and (self.tgt_view_ids_exclude is None):
            perm = rng.permutation(cam_ids)
            src_idxs = perm[:num_src]
            tgt_idx = perm[num_src]
        else:
            eligible = []
            for i in range(V):
                vid = int(cam_ids[i])
                if self.tgt_view_ids is not None and vid not in self.tgt_view_ids:
                    continue
                if self.tgt_view_ids_exclude is not None and vid in self.tgt_view_ids_exclude:
                    continue
                eligible.append(i)
            if not eligible:
                raise RuntimeError(
                    f"[ZJUViewSynthDataset] no eligible tgt views after holdout filter in {geom_path}"
                )
            if len(eligible) == 1:
                tgt_idx = eligible[0]
            else:
                tgt_idx = int(rng.choice(eligible))
            remaining = [i for i in range(V) if i != tgt_idx]
            if len(remaining) < num_src:
                num_src = len(remaining)
            perm = rng.permutation(remaining)
            src_idxs = perm[:num_src]

        tgt_vid = cam_ids[tgt_idx]

        policy = self.bad_sample_policy
        bad_infos = []
        bad_tgt_masked = False
        tgt_is_bad = False

        def _pick_next_index():
            n = len(self.samples)
            if n <= 1:
                return index
            j = int(np.random.randint(0, n - 1))
            if j >= index:
                j += 1
            return j

        def _handle_bad_samples(force_retry: bool = False):
            if not bad_infos:
                return None
            self._report_bad_samples(meta, index, bad_infos)
            if policy == "raise":
                reason = bad_infos[0].get("reason", "bad_sample")
                raise RuntimeError(
                    f"[ZJUViewSynthDataset] bad sample at index={index} geom={geom_path} reason={reason}"
                )
            do_retry = (policy == "skip") or force_retry
            if do_retry and _retry < self.bad_sample_max_retry:
                next_index = _pick_next_index()
                return self.__getitem__(next_index, _retry=_retry + 1)
            return None

        # --- 读取 src ---
        src_imgs = []
        src_depths = []
        src_confs = []
        src_pointmaps = []
        src_indices_used = []
        src_vids_list = []
        need_src_pad = 0

        src_pool = list(src_idxs)
        if policy == "drop_src":
            extra_idxs = [
                i for i in range(V) if i not in src_idxs and i != tgt_idx]
            if extra_idxs:
                try:
                    extra_idxs = list(rng.permutation(extra_idxs))
                except Exception:
                    extra_idxs = list(extra_idxs)
            src_pool = list(src_idxs) + extra_idxs

        for idx in src_pool:
            if len(src_imgs) >= num_src:
                break
            img_path = self._resolve_img_path(img_paths[idx])
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                cam_name = cam_names[idx] if cam_names is not None else None
                bad_infos.append({
                    "role": "src",
                    "view_idx": int(idx),
                    "view_id": int(cam_ids[idx]),
                    "cam_name": cam_name,
                    "img_path": img_path,
                    "stats": None,
                    "reason": f"open_failed:{e}",
                })
                if policy == "drop_src":
                    continue
                maybe = _handle_bad_samples(force_retry=True)
                if maybe is not None:
                    return maybe
                _handle_bad_samples(force_retry=True)
                raise
            img_t = TF.to_tensor(img)  # (3,H,W) [0,1]
            bad, stats, reasons = self._check_bad_image(img_t)
            if bad:
                cam_name = cam_names[idx] if cam_names is not None else None
                bad_infos.append({
                    "role": "src",
                    "view_idx": int(idx),
                    "view_id": int(cam_ids[idx]),
                    "cam_name": cam_name,
                    "img_path": img_path,
                    "stats": stats,
                    "reason": ",".join(reasons),
                })
                if policy == "drop_src":
                    continue

            d = self._process_depth_like(depth[idx])       # (Hd,Wd)
            c = self._normalize_conf(
                self._process_depth_like(depth_conf[idx]))  # (Hd,Wd)
            pm = pointmap[idx]                              # (Hd,Wd,3)
            self._assert_depth_shapes(
                d, c, pm, geom_path,
                int(idx), int(cam_ids[idx]), "src"
            )

            d_t = torch.from_numpy(d).float().unsqueeze(0)         # (1,Hd,Wd)
            c_t = torch.from_numpy(c).float().unsqueeze(0)         # (1,Hd,Wd)
            pm_t = torch.from_numpy(pm).permute(2, 0, 1).float()    # (3,Hd,Wd)

            src_imgs.append(img_t)
            src_depths.append(d_t)
            src_confs.append(c_t)
            src_pointmaps.append(pm_t)
            src_indices_used.append(int(idx))
            src_vids_list.append(int(cam_ids[idx]))

        if len(src_imgs) < num_src:
            bad_infos.append({
                "role": "src",
                "view_idx": None,
                "view_id": None,
                "cam_name": None,
                "img_path": "",
                "stats": None,
                "reason": f"insufficient_src:{len(src_imgs)}/{num_src}",
            })
            maybe = _handle_bad_samples(force_retry=True)
            if maybe is not None:
                return maybe
            if policy == "drop_src":
                need_src_pad = num_src - len(src_imgs)

        # --- 读取 tgt ---
        tgt_img_path = self._resolve_img_path(img_paths[tgt_idx])
        try:
            tgt_img_pil = Image.open(tgt_img_path).convert("RGB")
        except Exception as e:
            cam_name = cam_names[tgt_idx] if cam_names is not None else None
            bad_infos.append({
                "role": "tgt",
                "view_idx": int(tgt_idx),
                "view_id": int(tgt_vid),
                "cam_name": cam_name,
                "img_path": tgt_img_path,
                "stats": None,
                "reason": f"open_failed:{e}",
            })
            maybe = _handle_bad_samples(force_retry=True)
            if maybe is not None:
                return maybe
            _handle_bad_samples(force_retry=True)
            raise
        tgt_img = TF.to_tensor(tgt_img_pil)  # (3,H,W)
        bad, stats, reasons = self._check_bad_image(tgt_img)
        if bad:
            tgt_is_bad = True
            cam_name = cam_names[tgt_idx] if cam_names is not None else None
            bad_infos.append({
                "role": "tgt",
                "view_idx": int(tgt_idx),
                "view_id": int(tgt_vid),
                "cam_name": cam_name,
                "img_path": tgt_img_path,
                "stats": stats,
                "reason": ",".join(reasons),
            })
            if policy == "mask":
                bad_tgt_masked = True
        mask_path = self._infer_mask_path(tgt_img_path)
        if mask_path is None:
            raise FileNotFoundError(
                f"[ZJUViewSynthDataset] mask not found for: {tgt_img_path}"
            )
        tgt_mask_pil = Image.open(mask_path).convert("L")
        if tgt_mask_pil.size != tgt_img_pil.size:
            tgt_mask_pil = tgt_mask_pil.resize(
                tgt_img_pil.size, resample=Image.NEAREST)
        tgt_mask_np = np.array(tgt_mask_pil)
        tgt_mask_np = self._normalize_mask(tgt_mask_np)
        tgt_mask = torch.from_numpy(tgt_mask_np).float()
        # ensure binary {0,1} foreground mask
        tgt_fg = (tgt_mask > 0.5).float()
        if bad_tgt_masked:
            tgt_fg = torch.zeros_like(tgt_fg)

        d = self._process_depth_like(depth[tgt_idx])
        c = self._normalize_conf(self._process_depth_like(depth_conf[tgt_idx]))
        pm = pointmap[tgt_idx]
        self._assert_depth_shapes(
            d, c, pm, geom_path,
            int(tgt_idx), int(tgt_vid), "tgt"
        )

        tgt_depth = torch.from_numpy(d).float().unsqueeze(0)      # (1,Hd,Wd)
        tgt_conf = torch.from_numpy(c).float().unsqueeze(0)      # (1,Hd,Wd)
        tgt_pointmap = torch.from_numpy(pm).permute(
            2, 0, 1).float()  # (3,Hd,Wd)

        if need_src_pad > 0:
            pad_vid = src_vids_list[0] if src_vids_list else int(tgt_vid)
            pad_idx = src_indices_used[0] if src_indices_used else int(tgt_idx)
            dummy_img = torch.zeros_like(tgt_img)
            dummy_depth = torch.zeros_like(tgt_depth)
            dummy_conf = torch.zeros_like(tgt_conf)
            dummy_pm = torch.zeros_like(tgt_pointmap)
            for _ in range(int(need_src_pad)):
                src_imgs.append(dummy_img.clone())
                src_depths.append(dummy_depth.clone())
                src_confs.append(dummy_conf.clone())
                src_pointmaps.append(dummy_pm.clone())
                src_indices_used.append(pad_idx)
                src_vids_list.append(pad_vid)

        # --- 堆叠 ---
        src_imgs = torch.stack(src_imgs, dim=0)        # (S,3,H,W)
        src_depths = torch.stack(src_depths, dim=0)      # (S,1,Hd,Wd)
        src_confs = torch.stack(src_confs, dim=0)       # (S,1,Hd,Wd)
        src_pointmaps = torch.stack(src_pointmaps, dim=0)   # (S,3,Hd,Wd)
        src_vids = np.array(src_vids_list, dtype=np.int64)

        sample = {
            "src_imgs": src_imgs,
            "src_depth": src_depths,
            "src_depth_conf": src_confs,
            "src_pointmap": src_pointmaps,
            "tgt_img": tgt_img,
            "tgt_depth": tgt_depth,
            "tgt_depth_conf": tgt_conf,
            "tgt_conf": tgt_conf,  # 兼容旧 key
            "tgt_fg": tgt_fg,
            "tgt_pointmap": tgt_pointmap,
            "tgt_vid": torch.tensor(tgt_vid, dtype=torch.long),
            "src_vids": torch.tensor(src_vids, dtype=torch.long),
            "bad_tgt_masked": torch.tensor(
                1 if bad_tgt_masked else 0, dtype=torch.uint8),
        }
        if self.return_cam and (extrinsic is not None) and (intrinsic is not None):
            try:
                src_T = torch.from_numpy(extrinsic[src_indices_used]).float()
                src_K = torch.from_numpy(intrinsic[src_indices_used]).float()
                tgt_T = torch.from_numpy(extrinsic[tgt_idx]).float()
                tgt_K = torch.from_numpy(intrinsic[tgt_idx]).float()
                sample.update({
                    "src_T": src_T,
                    "src_K": src_K,
                    "tgt_T": tgt_T,
                    "tgt_K": tgt_K,
                })
            except Exception:
                pass
        if self.return_paths:
            try:
                sample.update({
                    "geom_path": geom_path,
                    "tgt_img_path": tgt_img_path,
                    "src_img_paths": [self._resolve_img_path(img_paths[i]) for i in src_indices_used],
                    "cam_names": cam_names if cam_names is not None else None,
                })
            except Exception:
                pass
        if bad_infos:
            force_retry = (policy == "drop_src" and tgt_is_bad)
            maybe = _handle_bad_samples(force_retry=force_retry)
            if maybe is not None:
                return maybe
        return sample



def _dump_one_batch_from_args(args):
    seq_names = args.seq_names
    if isinstance(seq_names, str):
        seq_names = [s for s in re.split(r"[,\s]+", seq_names) if s]
    ds = ZJUViewSynthDataset(
        root=args.zju_root,
        seq_names=seq_names,
        num_src_views=int(args.num_src_views),
        frame_subsample=int(args.frame_subsample),
        split=None if args.split == "None" else args.split,
        train_ratio=float(args.train_ratio),
        split_seed=int(args.split_seed),
        split_mode=str(args.split_mode),
        deterministic_views=bool(args.deterministic_views),
        view_seed=int(args.view_seed),
        tgt_view_ids=args.tgt_view_ids,
        tgt_view_names=args.tgt_view_names,
        tgt_view_ids_exclude=args.tgt_view_ids_exclude,
        tgt_view_names_exclude=args.tgt_view_names_exclude,
        return_cam=True,
        return_paths=True,
    )
    if len(ds) == 0:
        raise SystemExit("dataset empty")
    idx = int(args.index) % len(ds)
    sample = ds[idx]

    def _to_np(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x

    out = {}
    keep_keys = [
        "src_imgs", "src_depth", "src_depth_conf", "src_pointmap",
        "tgt_img", "tgt_depth", "tgt_depth_conf", "tgt_pointmap",
        "src_K", "src_T", "tgt_K", "tgt_T",
        "src_vids", "tgt_vid",
        "geom_path", "tgt_img_path", "src_img_paths", "cam_names",
    ]
    for k in keep_keys:
        if k in sample:
            out[k] = _to_np(sample[k])
    out["index"] = np.array([idx], dtype=np.int64)
    os.makedirs(os.path.dirname(args.dump_one_batch) or ".", exist_ok=True)
    np.savez_compressed(args.dump_one_batch, **out)
    print(f"[dump_one_batch] saved: {args.dump_one_batch}")


def _build_dump_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_one_batch", type=str, default="")
    ap.add_argument("--zju_root", type=str, required=True)
    ap.add_argument("--seq_names", type=str, required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--split", type=str, default="train",
                    choices=["train", "val", "test", "None"])
    ap.add_argument("--num_src_views", type=int, default=3)
    ap.add_argument("--frame_subsample", type=int, default=1)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--split_mode", type=str, default="random")
    ap.add_argument("--deterministic_views", action="store_true", default=True)
    ap.add_argument("--view_seed", type=int, default=2025)
    ap.add_argument("--tgt_view_ids", type=str, default=None)
    ap.add_argument("--tgt_view_names", type=str, default=None)
    ap.add_argument("--tgt_view_ids_exclude", type=str, default=None)
    ap.add_argument("--tgt_view_names_exclude", type=str, default=None)
    return ap


if __name__ == "__main__":
    parser = _build_dump_parser()
    args = parser.parse_args()
    if args.dump_one_batch:
        _dump_one_batch_from_args(args)
