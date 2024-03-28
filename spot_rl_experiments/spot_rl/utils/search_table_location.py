import os
import time
from copy import deepcopy
from glob import glob
from typing import Any

import cv2
import magnum as mn
import numpy as np
import open3d as o3d
import zmq
from spot_rl.utils.heuristic_nav import (
    ImageSearch,
    get_3d_point,
    get_arguments_for_image_search,
    pull_back_point_along_theta_by_offset,
)
from spot_rl.utils.plane_detection import plane_detect
from spot_rl.utils.pose_correction import (
    detect,
    generate_point_cloud,
    load_model,
    segment,
)
from spot_wrapper.spot import Spot, image_response_to_cv2


def heurisitic_object_search(
    x: float,
    y: float,
    theta: float,
    object_target: str,
    image_search: ImageSearch = None,
    save_cone_search_images: bool = True,
    pull_back: bool = True,
    skillmanager=None,
    angle_start=-90,
    angle_end=110,
    angle_interval=20,
):
    """
    Args:
            x (float): x coordinate of the nav target (in meters) specified in the world frame
            y (float): y coordinate of the nav target (in meters) specified in the world frame
            theta (float): yaw for the nav target (in radians) specified in the world frame
            object_target: str object to search
            image_search : spot_rl.utils.heuristic_nav.ImageSearch, Optional, default=None, ImageSearch (object detector wrapper), if none creates a new one for you uses OwlVit
            save_cone_search_images: bool, optional, default= True, saves image with detections in each search cone
            pull_back : bool, optional, default=True, pulls back x,y along theta direction
            skill_manager: skill_manager object to perform low level skills
        Returns:
            bool: True if navigation was successful, False otherwise, if True you are good to fire .pick metho

    """
    if save_cone_search_images:
        previously_saved_images = glob("imagesearch*.png")
        for f in previously_saved_images:
            os.remove(f)

    if image_search is None:
        image_search = ImageSearch(
            corner_static_offset=0.5,
            use_yolov8=False,
            visualize=save_cone_search_images,
            version=2,
        )

    (x, y) = (
        pull_back_point_along_theta_by_offset(x, y, theta, 0.2) if pull_back else (x, y)
    )
    # print(f"Nav targets adjusted on the theta direction ray {x, y, np.degrees(theta)}")
    # skillmanager.nav(x, y, theta)
    # skillmanager.nav_controller.nav_env.enable_nav_by_hand()

    spot: Spot = skillmanager.spot
    spot.open_gripper()
    gaze_arm_angles = deepcopy(skillmanager.pick_config.GAZE_ARM_JOINT_ANGLES)
    spot.set_arm_joint_positions(np.deg2rad(gaze_arm_angles), 1)
    time.sleep(1.5)
    found, (x, y, theta), point3d_in_vision, visulize_img = image_search.search(
        object_target, *get_arguments_for_image_search(spot)
    )
    rate = angle_interval  # control time taken to rotate the arm, higher the rotation higher is the time
    if not found:
        # start semi circle search
        semicircle_range = np.arange(angle_start, angle_end, angle_interval)
        for i_a, angle in enumerate(semicircle_range):
            print(f"Searching in {angle} cone")
            angle_time = int(np.abs(gaze_arm_angles[0] - angle) / rate)
            gaze_arm_angles[0] = angle
            spot.set_arm_joint_positions(np.deg2rad(gaze_arm_angles), angle_time)
            time.sleep(1.5)
            (
                found,
                (x, y, theta),
                point3d_in_vision,
                visulize_img,
            ) = image_search.search(  # type : ignore
                object_target, *get_arguments_for_image_search(spot)
            )
            if save_cone_search_images:
                cv2.imwrite(f"imagesearch_{angle}.png", visulize_img)
            if found:
                print(f"In Cone Search object found at {(x,y,theta)}")
                break

    else:
        if save_cone_search_images:
            cv2.imwrite("imagesearch_looking_forward.png", visulize_img)
    angle_time = int(
        np.abs(gaze_arm_angles[0] - skillmanager.pick_config.GAZE_ARM_JOINT_ANGLES[0])
        / rate
    )
    spot.set_arm_joint_positions(
        np.deg2rad(skillmanager.pick_config.GAZE_ARM_JOINT_ANGLES),
        angle_time,
    )
    if found:
        print(f"Nav goal after cone search {x, y, np.degrees(theta)}")
        point_in_home = np.array([x, y, point3d_in_vision[-1]])
        # backup_steps = skillmanager.nav_controller.nav_env.max_episode_steps
        # skillmanager.nav_controller.nav_env.max_episode_steps = 50
        # skillmanager.nav(x, y, theta)
        # skillmanager.nav_controller.nav_env.max_episode_steps = backup_steps
    # skillmanager.nav_controller.nav_env.disable_nav_by_hand()
    return found, point_in_home, (x, y, theta)


def area(x1, y1, x2, y2):
    return np.abs(y2 - y1) * np.abs(x2 - x1)


def select_the_bbox_closer_to_camera(predictions, raw_depth, camera_intrinsics):
    closer_dist = np.inf
    upper_dist = -np.inf
    closer_i = 0
    h, w = raw_depth.shape
    for i, prediction in enumerate(predictions):
        print(i, prediction)
        (x1, y1, x2, y2), score = prediction[0], prediction[-1]
        if x1 < x2 and y1 < y2:
            if 0 <= x1 <=  w and 0 <= x2 <=  w and 0 <= y1 <=  h and 0 <= y2 <=  h:
                cx, cy = (x1 + x2) //2, (y1 + y2)//2
                depth_patch = sample_patch_around_point(cx, cy, raw_depth)
                mean_dist = np.nan_to_num(depth_patch, nan=np.inf)
                xyz = get_3d_point(camera_intrinsics, (cx, cy), mean_dist)
                #mean_dist = mean_dist + np.abs(xyz[1]) #y -
                print(f"mean dist to prediction {i} is {mean_dist}, {xyz}")
                if mean_dist <= closer_dist :
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


def sample_patch_around_point(
    cx: int, cy: int, depth_raw: np.ndarray, patch_size: int = 5
) -> int:
    h, w = depth_raw.shape
    x1, x2 = cx - patch_size // 2, cx + patch_size // 2
    y1, y2 = cy - patch_size // 2, cy + patch_size // 2
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    deph_patch = depth_raw[y1:y2, x1:x2]
    deph_patch = deph_patch[deph_patch > 0]
    return np.median(deph_patch)


def percentage_threshold(coordinates, percentage):
    length = coordinates.max() - coordinates.min() + 1
    print(
        f"Extent of y cooridnates {length}, {coordinates.min()}, {coordinates.max()} percetage {(percentage*length)/100.}"
    )
    return (percentage * length) / 100.0


def remove_outliers(arr: np.ndarray):
    mean = np.mean(arr[:, 0])
    sd = np.std(arr[:, 0])
    return np.array(
        [x for x in arr if (x[0] > (mean - (2 * sd)) and x[0] < (mean + (2 * sd)))]
    )


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
    # breakpoint()
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


def farthest_point_sampling(points, num_samples):
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
    Z = points_3d[:, -1]  # n
    X_Z = points_3d[:, 0] / Z  # n/n
    Y_Z = points_3d[:, 1] / Z
    u = (fx * X_Z) + cx
    v = (fy * Y_Z) + cy
    return np.stack([u.flatten(), v.flatten()], axis=1).reshape(-1, 2)


def get_target_points_by_heuristic(
    np_points,
    x_percentile=50,
    x_sd_multiplier=0.1,
    z_percentage=20,
    z_sd_multiplier=0.2,
):

    x_thresh = np.percentile(np_points[:, 0], x_percentile)
    x_distances = np_points[:, 0] - x_thresh
    sd = np.sqrt(np.mean((x_distances) ** 2))

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

    """
    z_thresh = percentage_threshold(np_points_mid_point_filtered[:, -1], z_percentage)
    z_distances = np_points_mid_point_filtered[:, -1] - z_thresh
    mean = z_thresh  # np.mean(z_distances)
    sd = np.sqrt(np.mean((np_points_mid_point_filtered[:, -1] - z_thresh) ** 2))
    thresh = z_sd_multiplier
    np_points_mid_point_filtered_z_filtered = np.array(
        [
            x
            for i, x in enumerate(np_points_mid_point_filtered)
            if (
                z_distances[i] >= (mean - (thresh * sd))
                and z_distances[i] < (mean + (thresh * sd))
            )
        ]
    ).reshape(-1, 3)
    assert len(np_points_mid_point_filtered_z_filtered) > 0, f"No points left after z filering {len(np_points_mid_point_filtered_z_filtered)}"
    """
    np_points_mid_point_filtered_z_filtered = np_points_mid_point_filtered
    return np_points_mid_point_filtered_z_filtered, [
        np.median(np_points_mid_point_filtered_z_filtered[:, 0]),
        np.median(np_points_mid_point_filtered_z_filtered[:, 1]),
        np.median(np_points_mid_point_filtered_z_filtered[:, -1]),
    ]


def plot_intel_point_in_gripper_image(
    gripper_image_resps, gripper_T_intel:mn.Matrix4, point3d_in_intel: np.ndarray
):
    gripper_intrinsics = gripper_image_resps[0].source.pinhole.intrinsics
    gripper_image: np.ndarray = [
        image_response_to_cv2(gripper_image_resp)
        for gripper_image_resp in gripper_image_resps
    ]
    gripper_image, gripper_depth = gripper_image[0], gripper_image[1]
    # breakpoint()
    point3d_in_gripper: mn.Matrix3 = gripper_T_intel.transform_point(mn.Vector3(point3d_in_intel)) 
    point3d_in_gripper:np.ndarray = np.array([point3d_in_gripper.x, point3d_in_gripper.y, point3d_in_gripper.z])
    
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
    #print(f"Point in gripper {point3d_in_gripper}, depth in gripper {gripper_depth[int(point2d_in_gripper[1]), int(point2d_in_gripper[0])]/1000.}")
    return img_with_point


def detect_place_point_by_pcd_method(
    img,
    depth_raw,
    gripper_depth,
    camera_intrinsics_intel,
    camera_intrinsics_gripper,
    body_T_hand: mn.Matrix4,
    gripper_T_intel:mn.Matrix4,
    object_name: str = "table top",
    percentile=95,
    owlvitmodel=None,
    proceesor=None,
    sammodel=None,
    device="cpu",
):
    # if owlvitmodel is None and proceesor is None:
    #     owlvitmodel, proceesor = load_model("owlvit", device)

    h, w = img.shape[:2]
    predictions = detect_with_socket(img, object_name, 0.01, device)
    # predictions = detect(img, object_name, 0.01, device, owlvitmodel, proceesor)
    prediction, distance_to_prediction = select_the_bbox_closer_to_camera(
        predictions, depth_raw, camera_intrinsics_intel
    )
    if distance_to_prediction > 2000:
        return owlvitmodel, proceesor, sammodel, None, None
    img_with_bbox = img.copy()
    for bbox in predictions:
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
    #cv2.imshow("bbox", img_with_bbox)
    #cv2.waitKey(0)
    mask = np.ones_like(depth_raw).astype(bool)
    mask[y1:y2, x1:x2] = True
    fx = camera_intrinsics_intel.focal_length.x
    fy = camera_intrinsics_intel.focal_length.y
    cx = camera_intrinsics_intel.principal_point.x
    cy = camera_intrinsics_intel.principal_point.y
    #u,v in pixel -> depth at u,v, intriniscs -> xyz in 3D
    pcd = generate_point_cloud(img, depth_raw, mask, prediction[0], fx, fy, cx, cy)
    # o3d.io.write_point_cloud("original_pcd.ply", pcd)
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
    #print(corners_xys)
    y_threshold = np.percentile(corners_xys[:, -1], 70)

    # breakpoint()
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
    #depth_at_selected_xy = depth_raw[int(selected_xy[-1]), int(selected_xy[0])] / 1000.0
    assert (
        depth_at_selected_xy != 0.0
    ), f"Non zero depth required found {depth_at_selected_xy}"
    selected_point = get_3d_point(camera_intrinsics_intel, selected_xy, depth_at_selected_xy)
    #print(f"Point in Intel {selected_point}")
    
    #Convert selected point in gripper 3D to 3D then 3D to 2D then 2D to 3D using depth at Gripper
    #Ideally body-T_hand*hand_T_intel*point_in_intel should work but hand_T_intel has errors plus depth in Intel at closer points is not same as gripper 
    #thus we convert intel 3D point to Gripper 3D, convert 3D to 2D & then again from 2D to 3D using depth map
    selected_point_in_gripper = gripper_T_intel.transform_point(mn.Vector3(*selected_point))
    selected_point_in_gripper = np.array([selected_point_in_gripper.x, selected_point_in_gripper.y, selected_point_in_gripper.z])
    fx = camera_intrinsics_gripper.focal_length.x
    fy = camera_intrinsics_gripper.focal_length.y
    cx = camera_intrinsics_gripper.principal_point.x
    cy = camera_intrinsics_gripper.principal_point.y
    selected_xy_in_gripper = project_3d_to_pixel_uv(selected_point_in_gripper.reshape(1, 3), fx, fy, cx, cy)[0]
    depth_at_selected_xy_in_gripper = depth_at_selected_xy + 0.02 #Add 2 cm in intel depth to get depth in gripper
    print(f"Depth in gripper {depth_at_selected_xy_in_gripper}")
    assert depth_at_selected_xy_in_gripper != 0., f"Expeceted gripper depth at point {int(selected_xy_in_gripper[1]), int(selected_xy_in_gripper[0])} to be non zero but found {depth_at_selected_xy_in_gripper}"
    selected_point_in_gripper = get_3d_point(camera_intrinsics_gripper, selected_xy_in_gripper, depth_at_selected_xy_in_gripper)
    
    for xy in corners_xys:
        img_with_bbox = cv2.circle(
            img_with_bbox, (int(xy[0]), int(xy[1])), 1, (255, 0, 0)
        )
    img_with_bbox = cv2.circle(
        img_with_bbox, (int(selected_xy[0]), int(selected_xy[1])), 2, (0, 0, 255)
    )
    breakpoint()
    cv2.imshow("Table detection", img_with_bbox)
    cv2.waitKey(0)
    #o3d.visualization.draw_geometries([pcd, plane_pcd])
    cv2.imwrite("table_detection.png", img_with_bbox)
    cv2.destroyAllWindows()
    # o3d.io.write_point_cloud("plane_pcd.ply", plane_pcd)
    point_in_body = np.array(body_T_hand.transform_point(mn.Vector3(*selected_point_in_gripper)))
    #point_in_vision_with_direct_transform = body_T_hand.transform_point(gripper_T_intel.transform_point(mn.Vector3(selected_point)))
    #print(f"Point in body with direct transform {point_in_vision_with_direct_transform}, without direct transform {point_in_vision}")
    return owlvitmodel, proceesor, None, point_in_body, selected_point_in_gripper, img_with_bbox
#detected object intel RS 2D- > 3D -3D gripper- > 2D in gripper 
#3D intel - 3D gripper problem in z coordinate
#Intel point -> spot 
#3D point in Intel - 3D point gripper -> 2D point in gripper + Depth at Intel with offset -> 3D point in gripper