try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp

from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_wrapper.spot import Spot, SpotCamIds
from spot_wrapper.spot_qr_detector import SpotQRDetector

# Waypoints on left of the dock to observe the QR code clearly
SPOT_DOCK_OBSERVER_WAYPOINT_LEFT = [
    0.3840912007597151,
    -0.5816728569741766,
    149.58030524832756,
]
# Waypoints on right of the dock to observe the QR code clearly
SPOT_DOCK_OBSERVER_WAYPOINT_RIGHT = [
    0.5419185005639034,
    0.5319243891865243,
    -154.1506754378722,
]

# Waypoints in front of the dock to observe the QR code clearly
SPOT_DOCK_OBSERVER_WAYPOINT_FRONT = [
    1.5,
    0.0,
    180.0,
]


def extract_rotation_translation(matrices):
    rotations = []
    translations = []
    for matrix in matrices:
        rotations.append(matrix[:3, :3])
        translations.append(matrix[:3, 3])
    return np.array(rotations), np.array(translations)


def mean_rotation(rotations):
    quats = []
    for rot in rotations:
        quat = R.from_matrix(rot).as_quat()

        if quat[3] > 0:
            quat = -1.0 * quat
        quats.append(quat)

    quats = np.array(quats)

    # Compute the mean quaternion
    mean_quat = np.mean(quats, axis=0)

    # Normalize the mean quaternion
    mean_quat /= np.linalg.norm(mean_quat)

    # Convert mean quaternion back to rotation matrix
    mean_rot = R.from_quat(mean_quat).as_matrix()
    return mean_rot


def mean_translation(translations):
    return np.mean(translations, axis=0)


def rotation_std_dev(rotations, mean_rotation):
    mean_rot_object = R.from_matrix(mean_rotation)
    diffs = []
    for rot in rotations:
        rot_object = R.from_matrix(rot)
        diff = rot_object.inv() * mean_rot_object
        diffs.append(diff.magnitude())
    return np.std(diffs)


def translation_std_dev(translations):
    return np.std(translations, axis=0)


def analyze_transformations(matrices):
    rotations, translations = extract_rotation_translation(matrices)

    mean_rot = mean_rotation(rotations)
    mean_trans = mean_translation(translations)

    rot_std = rotation_std_dev(rotations, mean_rot)
    trans_std = translation_std_dev(translations)

    # Combine mean rotation and mean translation into a single matrix
    best_world_T_marker = np.eye(4)
    best_world_T_marker[:3, :3] = mean_rot
    best_world_T_marker[:3, 3] = mean_trans

    return best_world_T_marker, mean_rot, mean_trans, rot_std, trans_std


# def main():
def main(spot: Spot, verbose: bool = True, use_policies: bool = True):
    # Get Spot Skill Manager
    # @FIXME: This is initializing a ros-node silently (at core of inheritence in
    # SpotRobotSubscriberMixin). Make it explicit
    skill_manager = SpotSkillManager(
        spot=spot,
        verbose=verbose,
        use_policies=use_policies,
    )

    waypoint_list = [
        SPOT_DOCK_OBSERVER_WAYPOINT_LEFT,
        SPOT_DOCK_OBSERVER_WAYPOINT_RIGHT,
        SPOT_DOCK_OBSERVER_WAYPOINT_FRONT,
    ]

    cam_id = SpotCamIds.HAND_COLOR
    spot_qr = SpotQRDetector(spot=spot, cam_ids=[cam_id])

    world_T_marker_list = []  # type: List[sp.SE3]

    for waypoint in waypoint_list:
        # Navigate Spot to SPOT_DOCK_OBSERVER_WAYPOINT_LEFT so that it can see the marker
        status, msg = skill_manager.nav(
            waypoint[0],
            waypoint[1],
            np.deg2rad(waypoint[2]),
        )
        if not status:
            print(
                f"Failed to navigate to spot dock observer waypoint. Error: {msg}. Exiting..."
            )
            return

        # Sit Spot down
        skill_manager.sit()

        # Detect the marker and get the average pose of marker w.r.t spotWorld frame
        world_T_marker_list.append(
            spot_qr.get_avg_spotWorld_T_marker(cam_id=cam_id).matrix()
        )

    (
        best_world_T_marker,
        mean_rot,
        mean_trans,
        rot_std,
        trans_std,
    ) = analyze_transformations(world_T_marker_list)
    np.save("spotWorld_T_marker.npy", best_world_T_marker)
    print("Please move this file to spot-sim2real/bd_spot_wrapper/data")


if __name__ == "__main__":
    # spot = Spot("GenerateSpotWorldToMarker")
    # with spot.get_lease(hijack=True):
    #     main(spot)
    x = np.load("spotWorld_T_marker.npy")
    breakpoint()
