import math
import os

import cv2
import numpy as np
import open3d as o3d
import rospy
import torch
import zmq
from scipy.spatial.transform import Rotation
from segment_anything import SamPredictor, build_sam
from transformers import Owlv2ForObjectDetection, Owlv2Processor

socket = None
model, processor = None, None
sam = None
device = "cpu"


def load_model(model_name="owlvit", device="cpu"):
    global model
    global processor
    global sam
    if model_name == "owlvit":
        print("Loading OwlVit2")
        model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        ).to(device)
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        # run segment anything (SAM)
    if model_name == "sam":
        print("Loading SAM")
        sam = SamPredictor(
            build_sam(
                checkpoint="/home/tusharsangam/Desktop/spot-sim2real/spot_rl_experiments/weights/sam_vit_h_4b8939.pth"
            ).to(device)
        )


load_model("owlvit", device)
load_model("sam", device)


def detect(img, text_queries, score_threshold, device):
    global model
    global processor

    if model is None or processor is None:
        load_model("owlvit", device)

    text_queries = text_queries
    text_queries = text_queries.split(",")
    size = max(img.shape[:2])
    target_sizes = torch.Tensor([[size, size]])
    device = model.device
    inputs = processor(text=text_queries, images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    outputs.logits = outputs.logits.cpu()
    outputs.pred_boxes = outputs.pred_boxes.cpu()
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes
    )
    boxes, scores, labels = (
        results[0]["boxes"],
        results[0]["scores"],
        results[0]["labels"],
    )

    result_labels = []
    for box, score, label in zip(boxes, scores, labels):
        box = [int(i) for i in box.tolist()]
        if score < score_threshold:
            continue
        result_labels.append((box, text_queries[label.item()], score))
    result_labels.sort(key=lambda x: x[-1], reverse=True)
    return result_labels


def segment(image, boxes, size, device):

    global sam
    if sam is None:
        load_model("sam", device)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam.set_image(image)

    # H, W = size[1], size[0]

    for i in range(boxes.shape[0]):
        boxes[i] = torch.Tensor(boxes[i])

    boxes = torch.tensor(boxes, device=sam.device)

    transformed_boxes = sam.transform.apply_boxes_torch(boxes, image.shape[:2])

    masks, _, _ = sam.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks


def generate_point_cloud(
    image, depth, mask, bbox, fx=383.883, fy=383.883, cx=324.092, cy=238.042
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
    # depth_image /= depth_image.max()
    # x_min, y_min, x_max, y_max = bbox
    # fx, fy, cx, cy = # provide your values for fx, fy, cx, cy

    # Get the indices where mask is True

    x1, y1, x2, y2 = bbox
    rows, cols = np.where(mask)

    # depth[y1:y2, x1:x2] = inpaint_depth_map(depth[y1:y2, x1:x2])

    # Get the depth values at these indices
    depth_values = depth[rows, cols]

    depth_values = depth_values.astype(np.float32)
    depth_values /= 1000.0

    # assert np.any(depth_values == 0), "depth must not be zero"
    # depth_values[...] = np.mean(depth_values)
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
    Rotmat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(np.pi), -np.sin(np.pi)],
            [0.0, np.sin(np.pi), np.cos(np.pi)],
        ]
    )
    point_cloud.rotate(Rotmat, center=point_cloud.get_center())
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    point_cloud = point_cloud.select_by_index(ind)
    # cl, imd = point_cloud.remove_radius_outlier(nb_points=20, radius=0.5)
    # point_cloud = point_cloud.select_by_index(ind)
    # point_cloud = rotate_point_cloud(point_cloud, 'z', -180)
    return point_cloud


def connect_socket(port):
    global socket
    if socket is None:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect(f"tcp://localhost:{port}")
        print(f"Socket Connected at {port}")
    else:
        print("Connected socket found")
    return socket


def correct_pose(input_point_cloud, save_path, socket_conneted_to_server):
    assert socket_conneted_to_server is not None, "socket is not connected to server"
    if isinstance(input_point_cloud, str):
        pcd = np.array(o3d.io.read_point_cloud(input_point_cloud).points)
    else:
        pcd = input_point_cloud

    socket_conneted_to_server.send_pyobj(pcd)
    response = socket_conneted_to_server.recv_pyobj()
    R, pcd_aligned = response["R"], response["pcd_aligned"]

    if save_path is not None:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd_aligned)
        print(f"Aligned Point cloud saved to {save_path}")
        o3d.io.write_point_cloud(save_path, point_cloud)
    return R, pcd_aligned


def pose_correction_pipeline(
    rgb_image, depth_raw, bbox, object_name, cam_intrinsics, port=8000, device="cpu"
):
    # load_models(device)

    fx = cam_intrinsics.focal_length.x
    fy = cam_intrinsics.focal_length.y
    cx = cam_intrinsics.principal_point.x
    cy = cam_intrinsics.principal_point.y
    intrinsics = [fx, fy, cx, cy]

    if tuple(rgb_image.shape) != (480, 640, 3):
        rgb_image = cv2.resize(rgb_image, (640, 480))
        if depth_raw is not None:
            depth_raw = cv2.resize(depth_raw, (640, 480))
    h, w, _ = rgb_image.shape

    if bbox is None:
        label = detect(rgb_image, object_name, 0.01, device)
        print("prediction label", label)
        bbox = label[0][0]  # Example bounding box
    x1, y1, x2, y2 = bbox
    masks = segment(rgb_image, np.array([[x1, y1, x2, y2]]), [h, w], device)
    mask = masks[0, 0].cpu().numpy()

    depth_image = (depth_raw / depth_raw.max()) * 255.0
    h, w = depth_image.shape
    depth_image = np.asarray(
        np.dstack((depth_image, depth_image, depth_image)), dtype=np.uint8
    )
    masked_img = rgb_image * mask[:, :, None]
    masked_img = masked_img * depth_image
    cv2.imwrite("masked_image.png", masked_img)

    pc = generate_point_cloud(rgb_image, depth_raw, mask, bbox, *intrinsics)

    o3d.io.write_point_cloud("pc_before_sending.ply", pc)
    pc = np.array(pc.points)
    socket = connect_socket(port)
    save_path = "pc_after_correction.ply"
    R, aligned_pcd = correct_pose(pc, save_path, socket)
    print(R)

    euler_angles = R.tolist()

    # permute euler angles
    euler_zxy = [euler_angles[-1], euler_angles[0], euler_angles[1]]

    for i, angle in enumerate(euler_zxy):
        sign = angle / np.abs(angle)
        angle = np.abs(angle)
        if angle < 50:
            angle = 0
        elif angle < 90:
            angle = 90
        elif angle < 150:
            angle = 90
        euler_zxy[i] = sign * angle
    euler_zxy[-1] = 0.0
    euler_zxy[1] = 0.0
    # R = Rotation.from_euler("xyz", euler_zxy, True).as_quat().reshape(-1)
    rospy.set_param("pose_correction_success", True)
    rospy.set_param(
        "pose_correction",
        [float(euler_zxy[0]), float(euler_zxy[1]), float(euler_zxy[-1])],
    )


if __name__ == "__main__":
    port = 8000
    from spot_wrapper.spot import Spot, SpotCamIds, image_response_to_cv2

    intrinsics = [
        383.2665100097656,
        383.2665100097656,
        324.305419921875,
        236.64828491210938,
    ]

    """psuedo plys"""
    root_folder = "data"
    object_name = "bottle"
    variations = ["horizontal_1", "horizontal_2", "horizontal_3", "horizontal_4"]
    src = "intel"

    if src == "intel":
        intrinsics = [
            383.2665100097656,
            383.2665100097656,
            324.305419921875,
            236.64828491210938,
        ]
    else:
        intrinsics = [552.0291012161067, 552.0291012161067, 320.0, 240.0]
    """
    for variation in variations:
        rgb_file_name = os.path.join(root_folder, f"{object_name}_{variation}_{src}_rgb.png")
        assert os.path.exists(rgb_file_name)
        depth_file_name = os.path.join(root_folder, f"{object_name}_{variation}_{src}_depth.png")
        assert os.path.exists(depth_file_name), f"{depth_file_name}"
        rgb_image = cv2.imread(rgb_file_name)
        depth_raw = cv2.imread(depth_file_name, -1)
        R = pose_correction_pipeline(rgb_image, depth_raw, object_name, intrinsics, port)
    """
    spot = Spot("test")
    while True:
        image_responses = spot.get_hand_image_old(
            img_src=[SpotCamIds.INTEL_REALSENSE_COLOR, SpotCamIds.INTEL_REALSENSE_DEPTH]
        )
        camera_intrinsics = image_responses[0].source.pinhole.intrinsics
        image_responses = [
            image_response_to_cv2(image_response) for image_response in image_responses
        ]
        label = detect(image_responses[0], "bottle", 0.1, "cpu")
        bbox = label[0][0]
        x1, y1, x2, y2 = bbox
        img_with_label = cv2.rectangle(
            image_responses[0], (x1, y1), (x2, y2), (255, 0, 0)
        )
        cv2.imshow("with det", img_with_label)
        cv2.waitKey(1)
    """In real setup OwlVit2 & SAM & generate a pointcloud using camera intrinsics"""
