import os.path as osp
import pickle as pkl
import time
from typing import Any, Dict, List, Optional

import cv2
import open3d as o3d
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

# import open3d as o3d  # isort:skip
import numpy as np  # isort:skip


# from spot_wrapper.spot import Spot, SpotCamIds

# PATH_TO_LOGS = osp.join(
#     osp.dirname(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))))),
#     "data/data_logs",
# )
PATH_TO_LOG = (
    "/home/kavit/fair/rgbd_trajectories_real_world/SpotData/fremont_apt_scan/data.pkl"
)
PICKLE_PROTOCOL_VERSION = 4


def depth_to_xyz(depth: np.ndarray, camera_intrinsic: np.ndarray):
    """
    get depth from numpy using simple pinhole camera model
    """
    height, width = depth.shape
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z = depth

    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    px = camera_intrinsic[0, 2]
    py = camera_intrinsic[1, 2]

    # pixel indices start at top-left corner. for these equations, it starts at bottom-left
    x = (indices[:, :, 1] - px) * (z / fx)
    y = (indices[:, :, 0] - py) * (z / fy)

    # Should now be batch x height x width x 3, after this:
    xyz = np.stack([x, y, z], axis=-1)
    return xyz


def unproject_masked_depth_to_xyz_coordinates(
    depth: torch.Tensor,
    pose: torch.Tensor,
    inv_intrinsics: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Returns the XYZ coordinates for a batch posed RGBD image.

    Args:
        depth: The depth tensor, with shape (B, 1, H, W)
        mask: The mask tensor, with the same shape as the depth tensor,
            where True means that the point should be masked (not included)
        inv_intrinsics: The inverse intrinsics, with shape (B, 3, 3)
        pose: The poses, with shape (B, 4, 4)

    Returns:
        XYZ coordinates, with shape (N, 3) where N is the number of points in
        the depth image which are unmasked
    """

    batch_size, _, height, width = depth.shape
    if mask is None:
        mask = torch.full_like(depth, fill_value=False, dtype=torch.bool)
    flipped_mask = ~mask

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(0, width, device=depth.device),
        torch.arange(0, height, device=depth.device),
        indexing="xy",
    )
    xy = torch.stack([xs, ys], dim=-1)[None, :, :].repeat_interleave(batch_size, dim=0)
    xy = xy[flipped_mask.squeeze(1)]
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)

    # Associates poses and intrinsics with XYZ coordinates.
    inv_intrinsics = inv_intrinsics[:, None, None, :, :].expand(
        batch_size, height, width, 3, 3
    )[flipped_mask.squeeze(1)]
    pose = pose[:, None, None, :, :].expand(batch_size, height, width, 4, 4)[
        flipped_mask.squeeze(1)
    ]
    depth = depth[flipped_mask]

    # Applies intrinsics and extrinsics.
    xyz = xyz.to(inv_intrinsics).unsqueeze(1) @ inv_intrinsics.permute([0, 2, 1])
    xyz = xyz * depth[:, None, None]
    xyz = (xyz[..., None, :] * pose[..., None, :3, :3]).sum(dim=-1) + pose[
        ..., None, :3, 3
    ]
    xyz = xyz.squeeze(1)

    return xyz


def main(logfile_name: str = ""):
    log_packet_list = []  # type: List[Dict[str, Any]]
    # Read pickle file
    file_path = PATH_TO_LOG
    with open(file_path, "rb") as handle:  # should raise an error if invalid file path
        # Load pickle file
        log_packet_list = pkl.load(handle)

    xyz = []
    colors = []
    depths = []
    Rs = []
    tvecs = []
    intrinsics = []

    for i, _ in enumerate(tqdm(log_packet_list)):
        ## Spot GripperCam ##
        world_T_base = torch.from_numpy(log_packet_list[i]["vision_T_base"])
        base_T_camera = torch.from_numpy(
            log_packet_list[i]["camera_data"][0]["base_T_camera"]
        )
        cam_to_world = world_T_base @ base_T_camera
        K = log_packet_list[i]["camera_data"][0]["camera_intrinsics"]
        color = torch.from_numpy(
            log_packet_list[i]["camera_data"][0]["raw_image"][:, :, ::-1]
            / 255.0
            # log_packet_list[i]["camera_data"][0]["raw_image"]/ 255.0
        )
        depth = torch.from_numpy(
            np.asarray(
                log_packet_list[i]["camera_data"][1]["raw_image"] / 1000.0,
                dtype=np.float32,
            )
        )
        keep_mask = (0.6 < depth) & (depth < 3)

        ## IntelRS ##
        # world_T_base = torch.from_numpy(log_packet_list[i]["vision_T_base"])
        # base_T_camera = torch.from_numpy(
        #     log_packet_list[i]["camera_data"][2]["base_T_camera"]
        # )
        # cam_to_world = world_T_base @ base_T_camera
        # K = log_packet_list[i]["camera_data"][2]["camera_intrinsics"]
        # color = torch.from_numpy(
        #     log_packet_list[i]["camera_data"][2]["raw_image"][:, :, ::-1] / 255.0
        #     # log_packet_list[i]["camera_data"][2]["raw_image"] / 255.0
        # )
        # depth = torch.from_numpy(
        #     np.asarray(
        #         log_packet_list[i]["camera_data"][3]["raw_image"] / 1000.0,
        #         dtype=np.float32,
        #     )
        # )
        # keep_mask = (0.2 < depth) & (depth < 2)

        full_world_xyz = unproject_masked_depth_to_xyz_coordinates(
            depth=depth.unsqueeze(0).unsqueeze(1),
            pose=cam_to_world.unsqueeze(0),
            mask=~keep_mask.unsqueeze(0).unsqueeze(1),
            inv_intrinsics=torch.linalg.inv(torch.tensor(K[:3, :3])).unsqueeze(0),
        )
        xyz.append(full_world_xyz.view(-1, 3))
        colors.append(color[keep_mask])
        depths.append(depth)
        Rs.append(cam_to_world[:3, :3])
        tvecs.append(cam_to_world[:3, 3])
        intrinsics.append(torch.from_numpy(K[:3, :3]))

    print("Done processing all log-packets")
    print("Starting Open3D visualizer")
    stacked_xyz = np.vstack(xyz)
    stacked_colors = np.vstack(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(stacked_xyz)
    pcd.colors = o3d.utility.Vector3dVector(stacked_colors)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    # o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
