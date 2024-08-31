import copy
import os.path as osp
import time

import cv2
import magnum as mn
import numpy as np
import open3d as o3d
import rospy
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.utils.gripper_t_intel_path import GRIPPER_T_INTEL_PATH
from spot_rl.utils.pixel_to_3d_conversion_utils import (
    get_3d_point,
    get_best_uvz_from_detection,
    project_3d_to_pixel_uv,
)
from spot_rl.utils.plane_detection import NumpyToPCD, plane_detect
from spot_rl.utils.utils import ros_topics as rt
from spot_wrapper.spot import Spot, image_response_to_cv2
from std_msgs.msg import String

try:
    import sophuspy as sp
except Exception:
    import sophus as sp


class DetectionSubscriber:
    def __init__(self):
        self.latest_message = None
        rospy.Subscriber(rt.DETECTIONS_TOPIC, String, self.callback)

    def callback(self, data):
        self.latest_message = data.data

    def get_latest_message(self):
        return self.latest_message


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
    if num_samples <= points.shape[0]:
        return points
    farthest_pts = np.zeros((num_samples, 3))
    farthest_pts[0] = points[np.random.randint(len(points))]
    distances = np.linalg.norm(points - farthest_pts[0], axis=1)
    for i in range(1, num_samples):
        farthest_pts[i] = points[np.argmax(distances)]
        distances = np.minimum(
            distances, np.linalg.norm(points - farthest_pts[i], axis=1)
        )
    return farthest_pts


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

    point2d_in_gripper = project_3d_to_pixel_uv(
        point3d_in_gripper.reshape(1, 3), gripper_intrinsics
    )[0]
    img_with_point = cv2.circle(
        gripper_image.copy(),
        (int(point2d_in_gripper[0]), int(point2d_in_gripper[1])),
        2,
        (0, 0, 255),
    )
    return img_with_point


def plot_place_point_in_gripper_image(spot: Spot, point_in_gripper_camera: np.ndarray):
    rospy.set_param("is_gripper_blocked", 0)
    spot.open_gripper()
    time.sleep(0.5)
    gripper_resps = spot.get_hand_image()
    gripper_rgb = image_response_to_cv2(gripper_resps[0])
    intrinsics_gripper = gripper_resps[0].source.pinhole.intrinsics
    pixel = project_3d_to_pixel_uv(
        point_in_gripper_camera.reshape(1, 3), intrinsics_gripper
    )[0]
    print(f"Pixel in gripper image {pixel}")
    gripper_rgb = cv2.circle(
        gripper_rgb, (int(pixel[0]), int(pixel[1])), 2, (0, 0, 255)
    )
    try:
        visualization_img = cv2.imread("table_detection.png")
        visualization_img = np.hstack((visualization_img, gripper_rgb))
        cv2.namedWindow("Table detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Table detection", visualization_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Visualization error {e}")
        visualization_img = gripper_rgb
    cv2.imwrite("table_detection.png", visualization_img)


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
    hand_T_intel = sp.SE3(hand_T_intel[:3, :3], hand_T_intel[:3, 3])
    # Load hand_T_intel from caliberation
    image_resps = [image_response_to_cv2(image_resp) for image_resp in image_resps]
    body_T_hand: sp.SE3 = spot.get_sophus_SE3_spot_a_T_b(
        None,
        GRAV_ALIGNED_BODY_FRAME_NAME,  # "body",
        "link_wr1",
    )
    hand_T_gripper: sp.SE3 = spot.get_sophus_SE3_spot_a_T_b(
        snapshot_tree,
        "arm0.link_wr1",
        "hand_color_image_sensor",
    )
    body_T_hand = body_T_hand * hand_T_gripper
    # load body_T_hand
    # body_T_hand = body_T_hand.__matmul__(hand_T_intel)  # body_T_intel
    return (
        image_resps[0],
        image_resps[1],
        inrinsics_intel,
        intrinsics_gripper,
        body_T_hand,
        hand_T_intel,
    )


def filter_pointcloud_by_normals_in_the_given_direction(
    pcd_with_normals: o3d.geometry.PointCloud,
    direction_vector: np.ndarray,
    cosine_thresh: float = 0.25,
    body_T_hand: sp.SE3 = sp.SE3(np.eye(4)),
    gripper_T_intel: sp.SE3 = sp.SE3(np.eye(4)),
    visualize: bool = False,
):
    """Filter point clouds based on the normal"""
    direction_vector = direction_vector.reshape(3)
    normals = np.asarray(pcd_with_normals.normals).reshape(-1, 3)
    body_T_intel = sp.SO3((body_T_hand * gripper_T_intel).rotationMatrix())
    normals_in_body = body_T_intel * normals
    # Compute the dot product to get the cosines
    cosines = (normals_in_body @ direction_vector).reshape(-1)
    # Filter out the point clouds
    pcd_dir_filtered = pcd_with_normals.select_by_index(
        np.where(cosines > cosine_thresh)[0]
    )
    # if visualize:
    # o3d.visualization.draw_geometries([pcd_dir_filtered])
    # if visualize:
    # o3d.visualization.draw_geometries([pcd_dir_filtered])
    return pcd_dir_filtered


def rank_planes(
    planes: np.ndarray,
    all_plane_normals: np.ndarray,
    body_T_hand: sp.SE3 = sp.SE3(np.eye(4)),
    gripper_T_intel: sp.SE3 = sp.SE3(np.eye(4)),
    max_number_of_points: float = 1e8,
    percentile_thresh=30,
):
    """Filter point clouds based on the normal"""
    # breakpoint()
    body_T_intel: sp.SE3 = body_T_hand * gripper_T_intel

    # Compute the dot product to get the cosines
    # Calculate the angle using the dot product formula
    euclidean_dist, number_of_pts = [], []
    # indices_of_min_dist = []
    all_distances = []
    planes_points = [np.array(plane.points).reshape(-1, 3) for plane in planes]

    for plane_points in planes_points:
        plane_points_in_body = body_T_intel * plane_points
        norms_of_plane_points = np.linalg.norm(plane_points_in_body, axis=1)
        argmin_dist = int(np.argmin(norms_of_plane_points))
        all_distances.append(np.linalg.norm(plane_points_in_body[:, :3], axis=1))
        # indices_of_min_dist.append(argmin_dist)
        euclidean_dist.append(norms_of_plane_points[argmin_dist] / 0.5)
        number_of_pts.append(plane_points.shape[0] / max_number_of_points)

    # euclidean_dist =  np.array(euclidean_dist) / 0.5

    cost = -0.8 * np.array(euclidean_dist) + 0.2 * np.array(
        number_of_pts
    )  # + 0.9*cosines
    # cost = cosines + 0.3*height_of_plane
    # Filter out the point clouds
    # breakpoint()
    argmax = int(np.argmax(cost))
    distances_of_points_for_selected_plane = all_distances[argmax]
    sorted_distances_of_points = np.sort(distances_of_points_for_selected_plane.copy())
    selected_distance_based_on_percentile = np.percentile(
        sorted_distances_of_points, percentile_thresh
    )
    index_in_og_plane = int(
        np.argmin(
            np.abs(
                distances_of_points_for_selected_plane
                - selected_distance_based_on_percentile
            )
        )
    )
    return (
        argmax,
        planes_points[argmax][
            index_in_og_plane
        ],  # if not visualize else planes_points[argmax][indices_of_max_dist[argmax]],
        sp.SO3(body_T_intel.rotationMatrix()) * all_plane_normals,
    )


def compute_plane_normal_by_average(pcd):
    # pcd = NumpyToPCD(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    normals = np.array(pcd.normals)
    return normals.mean(axis=0), pcd.get_center().reshape(-1)


def compute_plane_normal(pcd):
    """
    Compute the normal of a plane given a point cloud on the plane.

    Parameters:
    points (numpy.ndarray): An array of shape (m, 3) representing points on the plane.

    Returns:
    numpy.ndarray: The normal vector of the plane.
    """
    points = np.array(pcd.points)
    # Step 1: Center the points (subtract the mean)
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Step 2: Compute the covariance matrix
    cov_matrix = np.cov(centered_points, rowvar=False)

    # Step 3: Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # The normal vector is the eigenvector corresponding to the smallest eigenvalue
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]

    return -1 * normal_vector, centroid


def visualize_all_planes(
    planes,
    image_rgb_orig,
    camera_intrinsics,
    all_planes_normals=None,
    normal_in_body=None,
):
    # project all 3D plane points back to image_rgb
    # fx, fy, cx, cy = camera_intrinsics
    np.random.seed(42)  # Optional: set a seed for reproducibility
    image_rgb = image_rgb_orig.copy()
    for p_i, plane in enumerate(planes):
        if all_planes_normals is not None:
            plane_normal = all_planes_normals[p_i]
            plane_origin = plane.get_center().reshape(-1)  # np.mean(plane, axis=0)
        else:
            plane_normal, plane_origin = compute_plane_normal_by_average(plane)
        plane_extend_origin = plane_origin + 0.1 * plane_normal
        plane_normals_2d = project_3d_to_pixel_uv(
            np.vstack((plane_origin, plane_extend_origin)), camera_intrinsics
        )
        plane_normals_2d = plane_normals_2d.astype(int).tolist()
        image_rgb = cv2.arrowedLine(
            image_rgb,
            plane_normals_2d[0],
            plane_normals_2d[1],
            (0, 0, 255),
            2,
            tipLength=0.2,
        )
        print(f"Plane id {p_i} plane normal {plane_normal}")
        plane_pts_2d = project_3d_to_pixel_uv(np.array(plane.points), camera_intrinsics)
        color = tuple(np.random.randint(0, 256, size=3).astype(int).tolist())
        for point in plane_pts_2d:
            x, y = list(map(int, point))
            image_rgb = cv2.circle(image_rgb, (x, y), 1, color, 1)
    if len(planes) == 1:
        cv2.putText(
            image_rgb,
            f"N:{np.round(normal_in_body, 2)}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            1,
        )

    cv2.namedWindow("plane detections", cv2.WINDOW_NORMAL)
    cv2.imshow("plane detections", image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_place_point_by_pcd_method(
    spot,
    GAZE_ARM_JOINT_ANGLES,
    percentile: float = 30,
    visualize=True,
    height_adjustment_offset: float = 0.10,
):
    """
    Tries to estimate point on the table in front of the Spot, using PCD & plane fitting heuristic
    Overview:
        Generate PCD -> detect plane -> select a point in 3D in intel RS camera.
        Convert 3D point in Intel to 3D point in gripper
        3D point in gripper is further back projected to pixel space u,v
        Resample depth at u,v in gripper & then project to 3D
    """
    assert osp.exists(GRIPPER_T_INTEL_PATH), f"{GRIPPER_T_INTEL_PATH} not found"
    gripper_T_intel = np.load(GRIPPER_T_INTEL_PATH)
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
    # pcd = pcd.uniform_down_sample(every_k_points=2)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    # 0.25 - 0 < angle < 75
    pcd = filter_pointcloud_by_normals_in_the_given_direction(
        pcd,
        np.array([0.0, 0.0, 1.0]),
        float(np.cos(np.deg2rad(45))),
        body_T_hand,
        gripper_T_intel,
        visualize=visualize,
    )

    max_number_of_points = len(pcd.points)
    # Down-sample by using voxel
    # pcd = pcd.voxel_down_sample(voxel_size=0.01)
    # print(f"After Downsampling {np.array(pcd.points).shape}")

    all_planes = plane_detect(pcd, visualize=visualize)  # returns list of open3d pcd

    all_plane_normals = np.array(
        [compute_plane_normal(plane)[0] for plane in all_planes], dtype=np.float32
    )

    plane_index, selected_point, normals_in_body = rank_planes(
        all_planes,
        all_plane_normals,
        body_T_hand,
        gripper_T_intel,
        max_number_of_points,
        percentile,
    )

    plane_pcd = all_planes[plane_index]

    if visualize:
        # for plane_index in range(len(all_planes)):
        # visualize_all_planes(all_planes, img, camera_intrinsics_intel, all_plane_normals)
        visualize_all_planes(
            [all_planes[plane_index]],
            img,
            camera_intrinsics_intel,
            [all_plane_normals[plane_index]],
            normals_in_body[plane_index],
        )

    corners_xys = project_3d_to_pixel_uv(
        np.array(plane_pcd.points), camera_intrinsics_intel
    )
    selected_xy = project_3d_to_pixel_uv(
        selected_point.reshape(1, 3), camera_intrinsics_intel
    )[0]

    selected_point_in_gripper = np.array(gripper_T_intel * selected_point)
    print(f"Intel point {selected_point}, Gripper Point {selected_point_in_gripper}")

    img_with_bbox = None

    img_with_bbox = img.copy()
    for xy in corners_xys:
        img_with_bbox = cv2.circle(
            img_with_bbox, (int(xy[0]), int(xy[1])), 1, (255, 0, 0)
        )
    img_with_bbox = cv2.circle(
        img_with_bbox, (int(selected_xy[0]), int(selected_xy[1])), 2, (0, 0, 255)
    )
    # if visualize:
    #     # For debug
    #     cv2.namedWindow("table_detection", cv2.WINDOW_NORMAL)
    #     cv2.imshow("table_detection", img_with_bbox)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    cv2.imwrite("table_detection.png", img_with_bbox)

    point_in_body = body_T_hand * selected_point_in_gripper
    placexyz = np.array(point_in_body)
    # This is useful if we want the place target to be in global frame:
    # convert_point_in_body_to_place_waypoint(point_in_body, spot)
    # Static Offset adjustment
    placexyz[0] += 0.10
    placexyz[2] += height_adjustment_offset
    return placexyz, selected_point_in_gripper, img_with_bbox


def detect_with_rospy_subscriber(object_name, image_scale=0.7):
    """Fetch the detection result"""
    # We use rospy approach reac the detection string from topic
    rospy.set_param("object_target", object_name)
    subscriber = DetectionSubscriber()
    fetch_time_threshold = 1.0
    begin_time = time.time()
    while (time.time() - begin_time) < fetch_time_threshold:
        latest_message = subscriber.get_latest_message()
        try:
            bbox_str = latest_message.split(",")[-4:]
            break
        except Exception:
            pass

    prediction = [int(float(num) / image_scale) for num in bbox_str]
    return prediction


def contrained_place_point_estimation(
    object_target: str,
    proposition: str,
    spot: Spot,
    GAZE_ARM_JOINT_ANGLES: list,
    percentile: float = 70,
    visualize=True,
    height_adjustment_offset: float = 0.10,
    image_scale: float = 0.7,
):
    # detect object target
    assert osp.exists(GRIPPER_T_INTEL_PATH), f"{GRIPPER_T_INTEL_PATH} not found"
    gripper_T_intel = np.load(GRIPPER_T_INTEL_PATH)
    spot.close_gripper()
    gaze_arm_angles = copy.deepcopy(GAZE_ARM_JOINT_ANGLES)
    spot.set_arm_joint_positions(np.deg2rad(gaze_arm_angles), 1)

    (
        img,
        depth_raw,
        camera_intrinsics_intel,
        camera_intrinsics_gripper,
        body_T_hand,
        gripper_T_intel,
    ) = get_arguments(spot, gripper_T_intel)
    time.sleep(1.5)
    print(f"Lookign for {object_target}")

    x1, y1, x2, y2 = detect_with_rospy_subscriber(object_target, image_scale)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    z = get_best_uvz_from_detection(depth_raw, [x1, y1, x2, y2])[-1]
    centee3d = get_3d_point(camera_intrinsics_intel, [cx, cy], z)

    mask = np.ones_like(depth_raw).astype(bool)

    fx = camera_intrinsics_intel.focal_length.x
    fy = camera_intrinsics_intel.focal_length.y
    cx = camera_intrinsics_intel.principal_point.x
    cy = camera_intrinsics_intel.principal_point.y
    # u,v in pixel -> depth at u,v, intriniscs -> xyz in 3D
    pcd = generate_point_cloud(img, depth_raw, mask, fx, fy, cx, cy)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    points = np.array(pcd.points)

    # First element is in x direction
    if proposition == "left":
        pcd_filtered = pcd.select_by_index(
            np.where(points[:, 0] - centee3d[0] < 0.1)[0]
        )
    elif proposition == "right":
        pcd_filtered = pcd.select_by_index(
            np.where(points[:, 0] - centee3d[0] > 0.1)[0]
        )
    elif proposition == "next-to":
        pcd_filtered = pcd.select_by_index(
            np.where(abs(points[:, 0] - centee3d[0]) > 0.1)[0]
        )

    # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    pcd = pcd_filtered
    if visualize:
        o3d.visualization.draw_geometries([pcd_filtered], point_show_normal=True)
    pcd = filter_pointcloud_by_normals_in_the_given_direction(
        pcd,
        np.array([0.0, -1.0, 0.0]),
        0.5,
        body_T_hand,
        gripper_T_intel,
        visualize=visualize,
    )

    plane_pcd = plane_detect(pcd)

    # Down-sample
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
    corners_xys = project_3d_to_pixel_uv(target_points, camera_intrinsics_intel)

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
        selected_point_in_gripper.reshape(1, 3), camera_intrinsics_gripper
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
        img_with_bbox = img.copy()
        for xy in corners_xys:
            img_with_bbox = cv2.circle(
                img_with_bbox, (int(xy[0]), int(xy[1])), 1, (255, 0, 0)
            )
        img_with_bbox = cv2.circle(
            img_with_bbox, (int(selected_xy[0]), int(selected_xy[1])), 2, (0, 0, 255)
        )
        cv2.imwrite("table_detection.png", img_with_bbox)

    point_in_body = body_T_hand.transform_point(mn.Vector3(*selected_point_in_gripper))
    placexyz = np.array(point_in_body)
    # This is useful if we want the place target to be in global frame:
    # convert_point_in_body_to_place_waypoint(point_in_body, spot)
    # Static Offset adjustment
    placexyz[0] += 0.10
    placexyz[2] += height_adjustment_offset
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
