import copy
import os.path as osp
import time

import cv2
import magnum as mn
import numpy as np
import open3d as o3d
import rospy
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.utils.heuristic_nav import get_3d_point
from spot_rl.utils.plane_detection import plane_detect
from spot_wrapper.spot import Spot, image_response_to_cv2

GRIPPER_T_INTEL = osp.join(osp.dirname(osp.abspath(__file__)), "gripper_T_intel.npy")


def sample_patch_around_point(
    cx: int, cy: int, depth_raw: np.ndarray, patch_size: int = 5
) -> int:
    """
    Samples a median depth in 5x5 patch around given x, y (pixel location in depth image array) as center in raw depth image
    """
    h, w = depth_raw.shape
    x1, x2 = cx - patch_size // 2, cx + patch_size // 2
    y1, y2 = cy - patch_size // 2, cy + patch_size // 2
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    deph_patch = depth_raw[y1:y2, x1:x2]
    deph_patch = deph_patch[deph_patch > 0]
    return np.median(deph_patch)


def convert_point_in_body_to_place_waypoint(point_in_body: mn.Vector3, spot: Spot):
    """
    Converts point in body frame to spot's global frame following the spot_rl_record_waypoint script
    """
    # spotwaypoint recorder method to convert xyz point from hand to global frame
    position, rotation = spot.get_base_transform_to(
        "link_wr1"
    )  # body_T_linkwr1 center wr1 in body
    position = [point_in_body.x, point_in_body.y, point_in_body.z]
    rotation = [rotation.x, rotation.y, rotation.z, rotation.w]
    wrist_T_base = SpotBaseEnv.spot2habitat_transform(position, rotation)

    base_place_target_habitat = np.array(wrist_T_base.translation)
    base_place_target = base_place_target_habitat[[0, 2, 1]]
    x, y, yaw = spot.get_xy_yaw()
    base_T_global = mn.Matrix4.from_(
        mn.Matrix4.rotation_z(mn.Rad(yaw)).rotation(),
        mn.Vector3(mn.Vector3(x, y, 0.5)),
    )
    global_place_target = base_T_global.transform_point(base_place_target)
    global_place_target = np.array(
        [global_place_target.x, global_place_target.y, global_place_target.z]
    )
    return global_place_target


def farthest_point_sampling(points, num_samples):
    """
    Downsamples N X 3 point cloud data to num_samples X 3 where num_samples <= N, selects farthest points
    """
    assert (
        num_samples <= points.shape[0]
    ), f"Num of points {num_samples} greater than shape of point cloud {points.shape[0]}"
    farthest_pts = np.zeros((num_samples, 3))
    farthest_pts[0] = points[np.random.randint(len(points))]
    distances = np.linalg.norm(points - farthest_pts[0], axis=1)
    for i in range(1, num_samples):
        farthest_pts[i] = points[np.argmax(distances)]
        distances = np.minimum(
            distances, np.linalg.norm(points - farthest_pts[i], axis=1)
        )
    return farthest_pts


def project_3d_to_pixel_uv(points_3d, fx, fy, cx, cy):
    """
    Back projects given xyz 3d point to pixel location u,v using camera intrinsics
    """
    Z = points_3d[:, -1]
    X_Z = points_3d[:, 0] / Z
    Y_Z = points_3d[:, 1] / Z
    u = (fx * X_Z) + cx
    v = (fy * Y_Z) + cy
    return np.stack([u.flatten(), v.flatten()], axis=1).reshape(-1, 2)


def generate_point_cloud(
    image, depth, mask, fx=383.883, fy=383.883, cx=324.092, cy=238.042
):
    """
    Generate a point cloud from an RGB image, depth image, and bounding box.

    Parameters:
    - rgb_image: RGB image as a numpy array.
    - depth_image: Depth image as a numpy array.
    - bbox: Bounding box as a tuple (x_min, y_min, x_max, y_max).
    - fx, fy: Focal lengths of the camera in x and y dimensions.
    - cx, cy: Optical center of the camera in x and y dimensions.

    Returns:
    - point_cloud: Open3D point cloud object.
    """
    rows, cols = np.where(mask)

    # Get the depth values at these indices
    depth_values = depth[rows, cols]

    depth_values = depth_values.astype(np.float32)
    depth_values /= 1000.0

    # Compute the 3D coordinates
    X = (cols - cx) * depth_values / fx
    Y = (rows - cy) * depth_values / fy
    Z = depth_values

    # Combine the X, Y, Z coordinates into a single N x 3 array
    points_3D = np.vstack((X, Y, Z)).T

    # Optional: Filter out points with a depth of 0 or below a certain threshold
    valid_depth_indices = depth_values > 0  # Example threshold
    print(
        f"Total vertices: {len(points_3D)}, Corrupt vertices: {len(points_3D) - len(valid_depth_indices)}"
    )
    points_3D = points_3D[valid_depth_indices]

    print(f"3D point cloud shape {points_3D.shape}")
    colors = image[rows, cols].reshape(-1, 3) / 255.0
    colors = colors[valid_depth_indices]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3D)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud


def get_target_points_by_heuristic(np_points, x_percentile=50, x_sd_multiplier=0.1):
    """
    Heurisitic function to sample suitable pixel location on table
    """
    x_thresh = np.percentile(np_points[:, 0], x_percentile)
    x_distances = np_points[:, 0] - x_thresh
    sd = np.sqrt(np.mean((x_distances) ** 2))

    # Points in the middle of the table are kept, in 0.1 std
    np_points_mid_point_filtered = np.array(
        [
            x
            for i, x in enumerate(np_points)
            if (
                x_distances[i] > (x_thresh - (x_sd_multiplier * sd))
                and x_distances[i] < (x_thresh + (x_sd_multiplier * sd))
            )
        ]
    ).reshape(-1, 3)
    return np_points_mid_point_filtered, [
        np.median(np_points_mid_point_filtered[:, 0]),
        np.median(np_points_mid_point_filtered[:, 1]),
        np.median(np_points_mid_point_filtered[:, -1]),
    ]


def plot_intel_point_in_gripper_image(
    gripper_image_resps, gripper_T_intel: mn.Matrix4, point3d_in_intel: np.ndarray
):
    """
    For visualization to verify point in intel is corresponding to same point in gripper using our calibaration transform
    """
    gripper_intrinsics = gripper_image_resps[0].source.pinhole.intrinsics
    gripper_image: np.ndarray = [
        image_response_to_cv2(gripper_image_resp)
        for gripper_image_resp in gripper_image_resps
    ]
    gripper_image = gripper_image[0]
    point3d_in_gripper = gripper_T_intel.transform_point(mn.Vector3(point3d_in_intel))
    point3d_in_gripper = np.array(
        [point3d_in_gripper.x, point3d_in_gripper.y, point3d_in_gripper.z]
    )

    fx = gripper_intrinsics.focal_length.x
    fy = gripper_intrinsics.focal_length.y
    cx = gripper_intrinsics.principal_point.x
    cy = gripper_intrinsics.principal_point.y
    point2d_in_gripper = project_3d_to_pixel_uv(
        point3d_in_gripper.reshape(1, 3), fx, fy, cx, cy
    )[0]
    img_with_point = cv2.circle(
        gripper_image.copy(),
        (int(point2d_in_gripper[0]), int(point2d_in_gripper[1])),
        2,
        (0, 0, 255),
    )
    return img_with_point


def get_arguments(spot: Spot, gripper_T_intel: np.ndarray):
    """
    Helper function to parse intel response images from spot, intrinsics parsing, transform estimation
    """
    place_point_generation_src: int = 1
    # Garther gripper image response -> snapshot tree
    gripper_resps = spot.get_hand_image()

    intrinsics_gripper = gripper_resps[0].source.pinhole.intrinsics
    snapshot_tree = gripper_resps[0].shot.transforms_snapshot
    # Switch to intel/gripper depending on place_point_generation_src
    rospy.set_param("is_gripper_blocked", place_point_generation_src)

    # Gather image & depth from Intel
    image_resps = (
        spot.get_hand_image()
    )  # assume gripper source, if intel source use caliberation gripper_T_intel.npy to multiply with vision_T_hand
    inrinsics_intel = image_resps[0].source.pinhole.intrinsics

    hand_T_intel = gripper_T_intel if place_point_generation_src else np.identity(4)
    # hand_T_intel[:3, :3] = np.identity(3)
    hand_T_intel = mn.Matrix4(
        hand_T_intel.T.tolist()
    )  # Load hand_T_intel from caliberation
    image_resps = [image_response_to_cv2(image_resp) for image_resp in image_resps]
    body_T_hand = spot.get_magnum_Matrix4_spot_a_T_b(
        "body", "hand_color_image_sensor", snapshot_tree
    )  # load body_T_hand
    # body_T_hand = body_T_hand.__matmul__(hand_T_intel)  # body_T_intel
    return (
        image_resps[0],
        image_resps[1],
        inrinsics_intel,
        intrinsics_gripper,
        body_T_hand,
        hand_T_intel,
    )


def detect_place_point_by_pcd_method(
    spot, GAZE_ARM_JOINT_ANGLES, percentile: float = 70, visualize=True
):
    """
    Tries to estimate point on the table in front of the Spot, using PCD & plane fitting heuristic
    Overview:
        Generate PCD -> detect plane -> select a point in 3D in intel RS camera.
        Convert 3D point in Intel to 3D point in gripper
        3D point in gripper is further back projected to pixel space u,v
        Resample depth at u,v in gripper & then project to 3D
    """
    assert osp.exists(GRIPPER_T_INTEL), f"{GRIPPER_T_INTEL} not found"
    gripper_T_intel = np.load(GRIPPER_T_INTEL)
    spot.close_gripper()
    gaze_arm_angles = copy.deepcopy(GAZE_ARM_JOINT_ANGLES)
    spot.set_arm_joint_positions(np.deg2rad(gaze_arm_angles), 1)

    # Wait for a bit to stabalized the gripper
    time.sleep(1.5)

    (
        img,
        depth_raw,
        camera_intrinsics_intel,
        camera_intrinsics_gripper,
        body_T_hand,
        gripper_T_intel,
    ) = get_arguments(spot, gripper_T_intel)

    h, w = img.shape[:2]
    mask = np.ones_like(depth_raw).astype(bool)
    fx = camera_intrinsics_intel.focal_length.x
    fy = camera_intrinsics_intel.focal_length.y
    cx = camera_intrinsics_intel.principal_point.x
    cy = camera_intrinsics_intel.principal_point.y
    # u,v in pixel -> depth at u,v, intriniscs -> xyz in 3D
    pcd = generate_point_cloud(img, depth_raw, mask, fx, fy, cx, cy)
    plane_pcd = plane_detect(pcd)
    # DownSample
    plane_pcd.points = o3d.utility.Vector3dVector(
        farthest_point_sampling(np.array(plane_pcd.points), 1024)
    )
    color = np.zeros(np.array(plane_pcd.points).shape)
    color[:, 0] = 1
    color[:, 1] = 0
    color[:, 2] = 0
    plane_pcd.colors = o3d.utility.Vector3dVector(color)
    target_points, selected_point = get_target_points_by_heuristic(
        np.array(plane_pcd.points)
    )
    corners_xys = project_3d_to_pixel_uv(target_points, fx, fy, cx, cy)

    y_threshold = np.percentile(corners_xys[:, -1], percentile)

    indices_gtr_thn_y_thresh = corners_xys[:, 1] >= y_threshold
    corners_gtr_than_y_thresh = corners_xys[indices_gtr_thn_y_thresh]
    indices = np.argsort(corners_gtr_than_y_thresh[:, 1])
    sorted_corners_gtr_than_y_thresh = corners_gtr_than_y_thresh[indices]
    selected_xy = sorted_corners_gtr_than_y_thresh[0]
    corners_xys = corners_gtr_than_y_thresh[indices]
    print(f"Selected XY {selected_xy}")
    depth_at_selected_xy = (
        sample_patch_around_point(int(selected_xy[0]), int(selected_xy[1]), depth_raw)
        / 1000.0
    )
    print(f"Depth in Intel {depth_at_selected_xy}")
    assert (
        depth_at_selected_xy != 0.0
    ), f"Non zero depth required found {depth_at_selected_xy}"
    selected_point = get_3d_point(
        camera_intrinsics_intel, selected_xy, depth_at_selected_xy
    )
    # Convert selected point in gripper 3D to 3D then 3D to 2D then 2D to 3D using depth at Gripper
    # Ideally body-T_hand*hand_T_intel*point_in_intel should work but hand_T_intel has errors plus depth in Intel at closer points is not same as gripper
    # thus we convert intel 3D point to Gripper 3D, convert 3D to 2D & then again from 2D to 3D using depth map
    selected_point_in_gripper = gripper_T_intel.transform_point(
        mn.Vector3(*selected_point)
    )
    selected_point_in_gripper = np.array(
        [
            selected_point_in_gripper.x,
            selected_point_in_gripper.y,
            selected_point_in_gripper.z,
        ]
    )
    fx = camera_intrinsics_gripper.focal_length.x
    fy = camera_intrinsics_gripper.focal_length.y
    cx = camera_intrinsics_gripper.principal_point.x
    cy = camera_intrinsics_gripper.principal_point.y
    selected_xy_in_gripper = project_3d_to_pixel_uv(
        selected_point_in_gripper.reshape(1, 3), fx, fy, cx, cy
    )[0]
    depth_at_selected_xy_in_gripper = (
        depth_at_selected_xy + 0.02
    )  # Add 2 cm in intel depth to get depth in gripper
    print(f"Depth in gripper {depth_at_selected_xy_in_gripper}")
    assert (
        depth_at_selected_xy_in_gripper != 0.0
    ), f"Expeceted gripper depth at point {int(selected_xy_in_gripper[1]), int(selected_xy_in_gripper[0])} to be non zero but found {depth_at_selected_xy_in_gripper}"
    selected_point_in_gripper = get_3d_point(
        camera_intrinsics_gripper,
        selected_xy_in_gripper,
        depth_at_selected_xy_in_gripper,
    )

    img_with_bbox = None
    if visualize:
        o3d.visualization.draw_geometries([pcd])
        img_with_bbox = img.copy()
        for xy in corners_xys:
            img_with_bbox = cv2.circle(
                img_with_bbox, (int(xy[0]), int(xy[1])), 1, (255, 0, 0)
            )
        img_with_bbox = cv2.circle(
            img_with_bbox, (int(selected_xy[0]), int(selected_xy[1])), 2, (0, 0, 255)
        )
        cv2.imshow("Table detection", img_with_bbox)
        cv2.waitKey(0)
        # o3d.visualization.draw_geometries([pcd, plane_pcd])
        cv2.imwrite("table_detection.png", img_with_bbox)
        cv2.destroyAllWindows()

    point_in_body = body_T_hand.transform_point(mn.Vector3(*selected_point_in_gripper))
    placexyz = convert_point_in_body_to_place_waypoint(point_in_body, spot)
    # Static Offset adjustment
    placexyz[0] += 0.10  # reduced from 0.20
    placexyz[2] += 0.10  # reduced from 0.15
    return placexyz, selected_point_in_gripper, img_with_bbox


# Legacy code
"""
def remove_outliers(arr: np.ndarray):
    mean = np.mean(arr[:, 0])
    sd = np.std(arr[:, 0])
    return np.array(
        [x for x in arr if (x[0] > (mean - (2 * sd)) and x[0] < (mean + (2 * sd)))]
    )

def percentage_threshold(coordinates, percentage):
    length = coordinates.max() - coordinates.min() + 1
    print(
        f"Extent of y cooridnates {length}, {coordinates.min()}, {coordinates.max()} percetage {(percentage*length)/100.}"
    )
    return (percentage * length) / 100.0

def area(x1, y1, x2, y2):
    return np.abs(y2 - y1) * np.abs(x2 - x1)


def select_the_bbox_closer_to_camera(predictions, raw_depth, camera_intrinsics):
    closer_dist = np.inf
    # upper_dist = -np.inf
    closer_i = 0
    h, w = raw_depth.shape
    for i, prediction in enumerate(predictions):
        print(i, prediction)
        (x1, y1, x2, y2), _ = prediction[0], prediction[-1]
        if x1 < x2 and y1 < y2:
            if 0 <= x1 <= w and 0 <= x2 <= w and 0 <= y1 <= h and 0 <= y2 <= h:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                depth_patch = sample_patch_around_point(cx, cy, raw_depth)
                mean_dist = np.nan_to_num(depth_patch, nan=np.inf)
                xyz = get_3d_point(camera_intrinsics, (cx, cy), mean_dist)
                # mean_dist = mean_dist + np.abs(xyz[1]) #y -
                print(f"mean dist to prediction {i} is {mean_dist}, {xyz}")
                if mean_dist <= closer_dist:
                    closer_dist = mean_dist
                    closer_i = i
    return predictions[closer_i], closer_dist


def segment_with_socket(img, bbox):
    port = 21001
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")
    print(f"Socket Connected at {port}")
    socket.send_pyobj((img, bbox))
    return socket.recv_pyobj()


def detect_with_socket(img, object_name, thresh=0.01, device="cpu"):
    port = 21001
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")
    print(f"Socket Connected at {port}")
    socket.send_pyobj((img, object_name, thresh, device))
    return socket.recv_pyobj()
def search_table(
    img,
    depth_raw,
    camera_intrinsics,
    vision_T_hand: mn.Matrix4,
    object_name: str = "table top",
    percentile=95,
    owlvitmodel=None,
    proceesor=None,
    sammodel=None,
    device="cpu",
):
    if owlvitmodel is None and proceesor is None:
        owlvitmodel, proceesor = load_model("owlvit", device)

    # if sammodel is None:
    # sammodel = load_model("sam", device)

    h, w = img.shape[:2]
    predictions = detect(img, object_name, 0.01, device, owlvitmodel, proceesor)
    # print(predictions)
    prediction, distance_to_prediction = select_the_bbox_closer_to_camera(
        predictions, depth_raw
    )
    if distance_to_prediction > 2000:
        return owlvitmodel, proceesor, sammodel, None, None
    img_with_bbox = img.copy()
    for bbox in [prediction]:
        x1, y1, x2, y2 = bbox[0]
        score = bbox[-1]
        img_with_bbox = cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (255, 0, 0), 1)
        img_with_bbox = cv2.putText(
            img_with_bbox,
            f"{score:.2f}, {area(x1, y1, x2, y2)}",
            (x1, y1),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.9,
            color=(0, 0, 255),
            thickness=2,
        )
    cv2.imshow("img_with_bboox", img_with_bbox)
    cv2.waitKey(0)
    # masks = segment(img, np.array([[x1, y1, x2, y2]]), [h, w], device, sammodel)
    # mask = masks[0, 0].cpu().numpy()
    mask = segment_with_socket(img, np.array([[x1, y1, x2, y2]]))
    bin_depth = np.where(depth_raw > 0, 1, 0).astype(np.uint8)

    table_image = img * mask[:, :, None] * bin_depth[:, :, None]
    vis_table_image = table_image.copy()
    # cv2.imshow("table", table_image)
    # cv2.waitKey(0)
    gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    corners_yxs = np.argwhere(dst > 0.01 * dst.max()).reshape(-1, 2)

    # corners_yxs = remove_outliers(corners_yxs)

    zs = depth_raw[corners_yxs[:, 0], corners_yxs[:, 1]]
    non_zero_indices = np.argwhere(zs > 0.0).flatten()
    zs = zs[non_zero_indices]
    corners_yxs = corners_yxs[non_zero_indices]

    bottom_edge_center_x = (corners_yxs[:, -1].max() + corners_yxs[:, -1].min()) // 2

    # y_threshold = np.percentile(corners_yxs[:, 0], percentile)

    y_threshold = percentage_threshold(corners_yxs[:, 0], percentile)

    indices_gtr_thn_y_thresh = corners_yxs[:, 0] >= y_threshold
    corners_gtr_than_y_thresh = corners_yxs[indices_gtr_thn_y_thresh]
    indices = np.argsort(corners_gtr_than_y_thresh[:, 0])
    sorted_corners_gtr_than_y_thresh = corners_gtr_than_y_thresh[indices]

    bottom_edge_center_y = sorted_corners_gtr_than_y_thresh[0, 0]

    bottom_edge_center = np.array(
        list(map(int, (bottom_edge_center_x, bottom_edge_center_y)))
    )
    depth_at_bottom_edge_center = (
        sample_patch_around_point(*bottom_edge_center, depth_raw) / 1000.0
    )
    place_xyz: np.ndarray = get_3d_point(
        camera_intrinsics, bottom_edge_center, depth_at_bottom_edge_center
    )
    place_xyzlocal = place_xyz.copy()
    place_xyz: mn.Vector3 = vision_T_hand.transform_point(mn.Vector3(*place_xyz))
    place_xyz: np.ndarray = np.array([place_xyz.x, place_xyz.y, place_xyz.z])
    for yx in corners_yxs:
        rgb_img_vis = cv2.circle(img, (yx[-1], yx[0]), 1, (255, 0, 0))
    rgb_img_vis = cv2.circle(
        rgb_img_vis, (bottom_edge_center[0], bottom_edge_center[1]), 4, (0, 0, 255)
    )
    cv2.imshow(
        "Table Corners", np.hstack([rgb_img_vis, vis_table_image, img_with_bbox])
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return owlvitmodel, proceesor, sammodel, place_xyz, place_xyzlocal
"""
