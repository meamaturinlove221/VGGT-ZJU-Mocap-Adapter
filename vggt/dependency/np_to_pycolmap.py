# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pycolmap
from .projection import project_3D_points_np


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    q = np.empty(4, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s
    q /= (np.linalg.norm(q) + 1e-12)
    return q


def _set_pose_if_possible(img, cam_from_world):
    # New/old pycolmap variants handle cam_from_world differently.
    if cam_from_world is None:
        return
    if hasattr(img, "set_cam_from_world"):
        img.set_cam_from_world(cam_from_world)
        return
    try:
        img.cam_from_world = cam_from_world
    except Exception:
        # Some versions expose a read-only property; skip gracefully.
        pass


def _get_image_id(img):
    # Compat: id vs image_id vs frame_id.
    for key in ("id", "image_id", "frame_id"):
        if hasattr(img, key):
            return getattr(img, key)
    return None


def _assign_points2D(image, points2D_list):
    # points2D_list: python list[pycolmap.Point2D] or list-like
    # New pycolmap: use Point2DList.
    if hasattr(pycolmap, "Point2DList"):
        image.points2D = pycolmap.Point2DList(points2D_list)
        return

    # fallback: hope it accepts plain list
    image.points2D = points2D_list


def _add_frame_to_reconstruction(reconstruction, frame_id=1, rig_id=1):
    if not hasattr(pycolmap, "Frame"):
        return None

    if hasattr(reconstruction, "add_rig") and hasattr(pycolmap, "Rig"):
        rig = None
        try:
            rig = pycolmap.Rig(rig_id=rig_id)
        except Exception:
            try:
                rig = pycolmap.Rig()
                if hasattr(rig, "rig_id"):
                    rig.rig_id = rig_id
            except Exception:
                rig = None
        if rig is not None:
            try:
                reconstruction.add_rig(rig)
            except Exception:
                pass

    try:
        frame = pycolmap.Frame(frame_id=frame_id, rig_id=rig_id)
        reconstruction.add_frame(frame)
        return frame_id
    except Exception:
        try:
            frame = pycolmap.Frame(frame_id=frame_id)
            reconstruction.add_frame(frame)
            return frame_id
        except Exception:
            try:
                frame = pycolmap.Frame(frame_id=frame_id, rig_id=0)
                reconstruction.add_frame(frame)
                return frame_id
            except Exception:
                return None


def _register_frame_if_possible(reconstruction, frame_id):
    if frame_id is None:
        return
    if hasattr(reconstruction, "register_frame"):
        try:
            reconstruction.register_frame(frame_id)
        except Exception:
            pass


def make_pycolmap_image(**kwargs):
    """
    Compatible with different pycolmap APIs:
    - some versions use `id`
    - some versions use `image_id`
    - some versions require setting cam_from_world after construction
    """
    cam_from_world = kwargs.pop("cam_from_world", None)

    # Prefer image_id when available.
    if "id" in kwargs and "image_id" not in kwargs:
        kwargs["image_id"] = kwargs.pop("id")

    try:
        img = pycolmap.Image(**kwargs)
    except Exception as exc:
        attempts = []
        if "frame_id" in kwargs:
            alt = dict(kwargs)
            alt.pop("frame_id", None)
            attempts.append(alt)
        if "image_id" in kwargs and "id" not in kwargs:
            alt = dict(kwargs)
            alt["id"] = alt.pop("image_id")
            attempts.append(alt)
        if "frame_id" in kwargs and "image_id" in kwargs and "id" not in kwargs:
            alt = dict(kwargs)
            alt.pop("frame_id", None)
            alt["id"] = alt.pop("image_id")
            attempts.append(alt)

        img = None
        for attempt in attempts:
            try:
                img = pycolmap.Image(**attempt)
                break
            except Exception:
                continue
        if img is None:
            raise exc

    _set_pose_if_possible(img, cam_from_world)
    return img


def batch_np_matrix_to_pycolmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    image_size,
    masks=None,
    max_reproj_error=None,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,
    min_inlier_per_frame=64,
    points_rgb=None,
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Check https://github.com/colmap/pycolmap for more details about its format

    NOTE that colmap expects images/cameras/points3D to be 1-indexed
    so there is a +1 offset between colmap index and batch index


    NOTE: different from VGGSfM, this function:
    1. Use np instead of torch
    2. Frame index and camera id starts from 1 rather than 0 (to fit the format of PyCOLMAP)
    """
    # points3d: Px3
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # tracks: NxPx2
    # masks: NxP
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    assert image_size.shape[0] == 2

    reproj_mask = None

    if max_reproj_error is not None:
        projected_points_2d, projected_points_cam = project_3D_points_np(
            points3d, extrinsics, intrinsics)
        projected_diff = np.linalg.norm(projected_points_2d - tracks, axis=-1)
        projected_points_2d[projected_points_cam[:, -1] <= 0] = 1e6
        reproj_mask = projected_diff < max_reproj_error

    if masks is not None and reproj_mask is not None:
        masks = np.logical_and(masks, reproj_mask)
    elif masks is not None:
        masks = masks
    else:
        masks = reproj_mask

    assert masks is not None

    if masks.sum(1).min() < min_inlier_per_frame:
        print(f"Not enough inliers per frame, skip BA.")
        return None, None

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()
    frame_id = _add_frame_to_reconstruction(reconstruction, frame_id=1, rig_id=1)

    inlier_num = masks.sum(0)
    valid_mask = inlier_num >= 2  # a track is invalid if without two inliers
    valid_idx = np.nonzero(valid_mask)[0]

    # Only add 3D points that have sufficient 2D points
    for vidx in valid_idx:
        # Use RGB colors if provided, otherwise use zeros
        rgb = points_rgb[vidx] if points_rgb is not None else np.zeros(3)
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), rgb)

    num_points3D = len(valid_idx)
    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(
                fidx, intrinsics, camera_type, extra_params)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=fidx + 1
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(
                extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        image_kwargs = {
            "image_id": int(fidx + 1),
            "name": f"image_{fidx + 1}",
            "camera_id": int(camera.camera_id),
            "cam_from_world": cam_from_world,
        }
        if frame_id is not None:
            image_kwargs["frame_id"] = int(frame_id)
        image = make_pycolmap_image(**image_kwargs)

        w2c = extrinsics[fidx]
        w2c = np.asarray(w2c, dtype=np.float64)  # (4,4) or (3,4)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        q = rotmat_to_qvec(R)  # (w,x,y,z)
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(q), t)
        image.cam_from_world = cam_from_world

        points2D_list = []

        point2D_idx = 0

        # NOTE point3D_id start by 1
        for point3D_id in range(1, num_points3D + 1):
            original_track_idx = valid_idx[point3D_id - 1]

            if (reconstruction.points3D[point3D_id].xyz < max_points3D_val).all():
                if masks[fidx][original_track_idx]:
                    # It seems we don't need +0.5 for BA
                    point2D_xy = tracks[fidx][original_track_idx]
                    # Please note when adding the Point2D object
                    # It not only requires the 2D xy location, but also the id to 3D point
                    points2D_list.append(
                        pycolmap.Point2D(point2D_xy, point3D_id))

                    # add element
                    track = reconstruction.points3D[point3D_id].track
                    track.add_element(fidx + 1, point2D_idx)
                    point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            _assign_points2D(image, points2D_list)
        except Exception:
            image.points2D = points2D_list

        # add image
        reconstruction.add_image(image)
        if hasattr(reconstruction, "register_image"):
            reconstruction.register_image(int(image.image_id))
        elif hasattr(reconstruction, "RegisterImage"):
            reconstruction.RegisterImage(int(image.image_id))

    _register_frame_if_possible(reconstruction, frame_id)
    return reconstruction, valid_mask


def pycolmap_to_batch_np_matrix(reconstruction, device="cpu", camera_type="SIMPLE_PINHOLE"):
    """
    Convert a PyCOLMAP Reconstruction Object to batched NumPy arrays.

    Args:
        reconstruction (pycolmap.Reconstruction): The reconstruction object from PyCOLMAP.
        device (str): Ignored in NumPy version (kept for API compatibility).
        camera_type (str): The type of camera model used (default: "SIMPLE_PINHOLE").

    Returns:
        tuple: A tuple containing points3D, extrinsics, intrinsics, and optionally extra_params.
    """

    num_images = len(reconstruction.images)
    max_points3D_id = max(reconstruction.point3D_ids())
    points3D = np.zeros((max_points3D_id, 3))

    for point3D_id in reconstruction.points3D:
        points3D[point3D_id - 1] = reconstruction.points3D[point3D_id].xyz

    extrinsics = []
    intrinsics = []

    extra_params = [] if camera_type == "SIMPLE_RADIAL" else None

    for i in range(num_images):
        # Extract and append extrinsics
        pyimg = reconstruction.images[i + 1]
        pycam = reconstruction.cameras[pyimg.camera_id]
        matrix = pyimg.cam_from_world.matrix()
        extrinsics.append(matrix)

        # Extract and append intrinsics
        calibration_matrix = pycam.calibration_matrix()
        intrinsics.append(calibration_matrix)

        if camera_type == "SIMPLE_RADIAL":
            extra_params.append(pycam.params[-1])

    # Convert lists to NumPy arrays instead of torch tensors
    extrinsics = np.stack(extrinsics)
    intrinsics = np.stack(intrinsics)

    if camera_type == "SIMPLE_RADIAL":
        extra_params = np.stack(extra_params)
        extra_params = extra_params[:, None]

    return points3D, extrinsics, intrinsics, extra_params


########################################################


def batch_np_matrix_to_pycolmap_wo_track(
    points3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    image_size,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Different from batch_np_matrix_to_pycolmap, this function does not use tracks.

    It saves points3d to colmap reconstruction format only to serve as init for Gaussians or other nvs methods.

    Do NOT use this for BA.
    """
    # points3d: Px3
    # points_xyf: Px3, with x, y coordinates and frame indices
    # points_rgb: Px3, rgb colors
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N = len(extrinsics)
    P = len(points3d)

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()
    frame_id = _add_frame_to_reconstruction(reconstruction, frame_id=1, rig_id=1)

    for vidx in range(P):
        reconstruction.add_point3D(
            points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(
                fidx, intrinsics, camera_type)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=fidx + 1
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(
                extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        image_kwargs = {
            "image_id": int(fidx + 1),
            "name": f"image_{fidx + 1}",
            "camera_id": int(camera.camera_id),
            "cam_from_world": cam_from_world,
        }
        if frame_id is not None:
            image_kwargs["frame_id"] = int(frame_id)
        image = make_pycolmap_image(**image_kwargs)

        w2c = extrinsics[fidx]
        w2c = np.asarray(w2c, dtype=np.float64)  # (4,4) or (3,4)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        q = rotmat_to_qvec(R)  # (w,x,y,z)
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(q), t)
        image.cam_from_world = cam_from_world

        points2D_list = []

        point2D_idx = 0

        points_belong_to_fidx = points_xyf[:, 2].astype(np.int32) == fidx
        points_belong_to_fidx = np.nonzero(points_belong_to_fidx)[0]

        for point3D_batch_idx in points_belong_to_fidx:
            point3D_id = point3D_batch_idx + 1
            point2D_xyf = points_xyf[point3D_batch_idx]
            point2D_xy = point2D_xyf[:2]
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

            # add element
            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            _assign_points2D(image, points2D_list)
        except Exception:
            image.points2D = points2D_list

        # add image
        reconstruction.add_image(image)
        if hasattr(reconstruction, "register_image"):
            reconstruction.register_image(int(image.image_id))
        elif hasattr(reconstruction, "RegisterImage"):
            reconstruction.RegisterImage(int(image.image_id))

    _register_frame_if_possible(reconstruction, frame_id)
    return reconstruction


def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """
    Helper function to get camera parameters based on camera type.

    Args:
        fidx: Frame index
        intrinsics: Camera intrinsic parameters
        camera_type: Type of camera model
        extra_params: Additional parameters for certain camera types

    Returns:
        pycolmap_intri: NumPy array of camera parameters
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1],
                intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array(
            [focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array(
            [focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2], extra_params[fidx][0]])
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")

    return pycolmap_intri
