import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import click
import cv2
import numpy as np
import rospy
import sophus as sp
from aria_data_utils.detector_wrappers.april_tag_detector import AprilTagDetectorWrapper
from aria_data_utils.detector_wrappers.object_detector import ObjectDetectorWrapper
from aria_data_utils.image_utils import decorate_img_with_text
from aria_data_utils.perception.april_tag_pose_estimator import AprilTagPoseEstimator
from bosdyn.client.frame_helpers import get_a_tform_b
from fairotag.scene import Scene
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from matplotlib import pyplot as plt
from projectaria_tools.core import calibration, data_provider, mps
from scipy.spatial.transform import Rotation as R
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.models.owlvit import OwlVit
from spot_wrapper.spot import (
    Spot,
    SpotCamIds,
    generate_PoseStamped_a_T_b_from_spSE3,
    generate_spSE3_a_T_b_from_PoseStamped,
    generate_spSE3_a_T_b_from_TransformStamped,
    generate_TransformStamped_a_T_b_from_spSE3,
    image_response_to_cv2,
    xyt2sophus,
)
from std_msgs.msg import Bool
from tf2_ros import (
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    StaticTransformBroadcaster,
)
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))
FILTER_DIST = 2.4  # in meters (distance for valid detection)

############## Simple Helper Methods to keep code clean ##############


######################################################################


class SpotQRDetector:
    def __init__(self, spot: Spot):
        self.spot = spot
        print("...Spot initialized...")

    def _to_camera_metadata_dict(self, camera_intrinsics):
        """Converts a camera intrinsics proto to a 3x3 matrix as np.array"""
        intrinsics = {
            "fx": camera_intrinsics.focal_length.x,
            "fy": camera_intrinsics.focal_length.x,
            "ppx": camera_intrinsics.principal_point.x,
            "ppy": camera_intrinsics.principal_point.y,
            "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
        return intrinsics

    def _get_body_T_handcam(self, frame_tree_snapshot_hand):
        raise RuntimeError("This method is not used anymore. Please use get_spot_a_T_b")

    def get_spot_a_T_b(self, a: str, b: str) -> sp.SE3:
        frame_tree_snapshot = (
            self.spot.get_robot_state().kinematic_state.transforms_snapshot
        )
        se3_pose = get_a_tform_b(frame_tree_snapshot, a, b)
        pos = se3_pose.get_translation()
        quat = se3_pose.rotation.normalize()
        return sp.SE3(quat.to_matrix(), pos)

    def _get_body_T_headcam(self, frame_tree_snapshot_head):
        raise RuntimeError("This method is not used anymore. Please use get_spot_a_T_b")

    def _get_spotWorld_T_handcam(
        self, frame_tree_snapshot_hand, spot_frame: str = "vision"
    ):
        if spot_frame != "vision" and spot_frame != "odom":
            raise ValueError("spot_frame should be either vision or odom")
        spot_world_frame = spot_frame

        hand_bd_wrist_T_handcam_dict = (
            frame_tree_snapshot_hand.child_to_parent_edge_map.get(
                "hand_color_image_sensor"
            ).parent_tform_child
        )
        hand_mn_wrist_T_handcam = self.spot.convert_transformation_from_BD_to_magnum(
            hand_bd_wrist_T_handcam_dict
        )

        hand_bd_body_T_wrist_dict = (
            frame_tree_snapshot_hand.child_to_parent_edge_map.get(
                "arm0.link_wr1"
            ).parent_tform_child
        )
        hand_mn_body_T_wrist = self.spot.convert_transformation_from_BD_to_magnum(
            hand_bd_body_T_wrist_dict
        )

        hand_bd_body_T_spotWorld_dict = (
            frame_tree_snapshot_hand.child_to_parent_edge_map.get(
                spot_world_frame
            ).parent_tform_child
        )
        hand_mn_body_T_spotWorld = self.spot.convert_transformation_from_BD_to_magnum(
            hand_bd_body_T_spotWorld_dict
        )
        hand_mn_spotWorld_T_body = hand_mn_body_T_spotWorld.inverted()

        hand_mn_body_T_handcam = hand_mn_body_T_wrist @ hand_mn_wrist_T_handcam
        hand_mn_spotWorld_T_handcam = hand_mn_spotWorld_T_body @ hand_mn_body_T_handcam

        return hand_mn_spotWorld_T_handcam

    def _get_spotWorld_T_headcam(
        self, frame_tree_snapshot_head, use_vision_as_world: bool = True
    ):
        raise RuntimeError("This method is not used anymore. Please use get_spot_a_T_b")

    def get_avg_spotWorld_T_marker_HAND(
        self,
        spot_frame: str = "vision",
        data_size_for_avg: int = 10,
        filter_dist: float = 2.2,
    ):
        """
        Returns the average transformation of spot world frame to marker frame

        We get a camera_T_marker for each frame in which marker is detected.
        Depending on the frame rate of image capture, multiple frames may have captured the marker.
        Averaging all transforms would be best way to compensate for any noise that may exist in any frame's detections
        camera_T_marker is used to compute spot_T_marker[i] and thus spotWorld_T_marker[i].
        Then we average all spotWorld_T-marker to find average marker pose wrt spotWorld.
        """
        if spot_frame != "vision" and spot_frame != "odom":
            raise ValueError("base_frame should be either vision or odom")
        cv2.namedWindow("hand_image", cv2.WINDOW_AUTOSIZE)
        spot_world_frame = spot_frame

        # Get Hand camera intrinsics
        hand_cam_intrinsics = self.spot.get_camera_intrinsics(SpotCamIds.HAND_COLOR)
        hand_cam_intrinsics = self._to_camera_metadata_dict(hand_cam_intrinsics)
        hand_cam_pose_estimator = AprilTagPoseEstimator(hand_cam_intrinsics)

        # Register marker ids
        marker_ids_list = [DOCK_ID]
        # marker_ids_list = [i for i in range(521, 550)]
        hand_cam_pose_estimator.register_marker_ids(marker_ids_list)

        marker_position_from_dock_list = []  # type: List[Any]
        marker_quaternion_form_dock_list = []  # type: List[Any]

        marker_position_from_robot_list = []  # type: List[Any]
        marker_quaternion_form_robot_list = []  # type: List[Any]

        while len(marker_position_from_dock_list) < data_size_for_avg:
            print(f"Iterating - {len(marker_position_from_dock_list)}")
            is_marker_detected_from_hand_cam = False
            img_response_hand = self.spot.get_hand_image()
            img_hand = image_response_to_cv2(img_response_hand)

            (
                img_rend_hand,
                hand_mn_handcam_T_marker,
            ) = hand_cam_pose_estimator.detect_markers_and_estimate_pose(
                img_hand, should_render=True
            )

            if hand_mn_handcam_T_marker is not None:
                print("Trackedddd")
                is_marker_detected_from_hand_cam = True

            # Spot - spotWorld_T_handcam computation
            frame_tree_snapshot_hand = img_response_hand.shot.transforms_snapshot
            hand_mn_body_T_handcam = self._get_body_T_handcam(frame_tree_snapshot_hand)
            hand_mn_spotWorld_T_handcam = self._get_spotWorld_T_handcam(
                frame_tree_snapshot_hand, spot_frame=spot_frame
            )

            if is_marker_detected_from_hand_cam:
                hand_mn_spotWorld_T_marker = (
                    hand_mn_spotWorld_T_handcam @ hand_mn_handcam_T_marker
                )

                hand_mn_body_T_marker = (
                    hand_mn_body_T_handcam @ hand_mn_handcam_T_marker
                )

                img_rend_hand = decorate_img_with_text(
                    img=img_rend_hand,
                    frame_name=spot_world_frame,
                    position=hand_mn_spotWorld_T_marker.translation,
                )

                dist = hand_mn_handcam_T_marker.translation.length()

                print(
                    f"Dist = {dist}, Recordings - {len(marker_position_from_dock_list)}"
                )
                if dist < filter_dist:
                    marker_position_from_dock_list.append(
                        np.array(hand_mn_spotWorld_T_marker.translation)
                    )
                    marker_quaternion_form_dock_list.append(
                        R.from_matrix(hand_mn_spotWorld_T_marker.rotation()).as_quat()
                    )
                    marker_position_from_robot_list.append(
                        np.array(hand_mn_body_T_marker.translation)
                    )
                    marker_quaternion_form_robot_list.append(
                        R.from_matrix(hand_mn_body_T_marker.rotation()).as_quat()
                    )

            cv2.imshow("hand_image", img_rend_hand)
            cv2.waitKey(1)

        marker_position_from_dock_np = np.array(marker_position_from_dock_list)
        avg_marker_position_from_dock = np.mean(marker_position_from_dock_np, axis=0)

        marker_quaternion_from_dock_np = np.array(marker_quaternion_form_dock_list)
        avg_marker_quaternion_from_dock = np.mean(
            marker_quaternion_from_dock_np, axis=0
        )

        marker_position_from_robot_np = np.array(marker_position_from_robot_list)
        avg_marker_position_from_robot = np.mean(marker_position_from_robot_np, axis=0)

        marker_quaternion_from_robot_np = np.array(marker_quaternion_form_robot_list)
        avg_marker_quaternion_from_robot = np.mean(
            marker_quaternion_from_robot_np, axis=0
        )

        avg_spotWorld_T_marker = sp.SE3(
            R.from_quat(avg_marker_quaternion_from_dock).as_matrix(),
            avg_marker_position_from_dock,
        )
        avg_spot_T_marker = sp.SE3(
            R.from_quat(avg_marker_quaternion_from_robot).as_matrix(),
            avg_marker_position_from_robot,
        )

        return avg_spotWorld_T_marker, avg_spot_T_marker

    def get_avg_spotWorld_T_marker_HEAD(
        self,
        use_vision_as_world: bool = True,
        data_size_for_avg: int = 10,
        filter_dist: float = FILTER_DIST,
    ):
        pass

    def get_avg_spotWorld_T_marker(
        self,
        camera: str = "hand",
        use_vision_as_world: bool = True,
        data_size_for_avg: int = 10,
        filter_dist: float = FILTER_DIST,
    ):
        pass


class SpotAriaP1Node:
    """
    TODO: Add docstring
    """

    def __init__(self, spot: Spot, verbose=False):
        # Get Spot Skill Manager
        self.skill_manager = SpotSkillManager(
            should_stand_on_start=False
        )  # @FIXME: NEED THIS HERE AS IT INITIALIZES A Ros-node which we need here after

        spot_qr = SpotQRDetector(spot=spot)
        (
            avg_spotWorld_T_marker,
            avg_spot_T_marker,
        ) = spot_qr.get_avg_spotWorld_T_marker_HAND()

        # Instantiate static transform broadcaster for publishing marker w.r.t spotWorld transforms
        self.static_tf_broadcaster = StaticTransformBroadcaster()

        # Instantiate publisher for publishing goto pose for pick and place
        self._goto_pose_pub_for_pick = rospy.Publisher(
            "/goto_pose_for_pick", PoseStamped, queue_size=10
        )
        self._goto_pose_pub_for_place = rospy.Publisher(
            "/goto_pose_for_place", PoseStamped, queue_size=10
        )

        # Publish marker w.r.t spotWorld transforms for 5 seconds so it can be seen in rviz
        start_time = rospy.Time.now()
        while rospy.Time.now() - start_time < rospy.Duration(2.0):
            self.static_tf_broadcaster.sendTransform(
                generate_TransformStamped_a_T_b_from_spSE3(
                    a_Tform_b=avg_spotWorld_T_marker,
                    parent_frame="spotWorld",
                    child_frame="marker",
                )
            )
            rospy.sleep(0.5)

        # Initialize static transform subscriber for listening to ariaWorld w.r.t marker transform
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer)

        # WHAT TO PASS AS fixed_frame arg????
        while not rospy.is_shutdown() and not self._tf_buffer.can_transform(
            target_frame="spotWorld", source_frame="ariaWorld", time=rospy.Time(0)
        ):
            rospy.logwarn_throttle(
                5.0, "Waiting for transform from ariaWorld to spotWorld"
            )  # I have no idea what this does
            rospy.sleep(0.5)
        try:
            transform_stamped_spotWorld_T_ariaWorld = self._tf_buffer.lookup_transform(
                target_frame="spotWorld", source_frame="ariaWorld", time=rospy.Time(0)
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            raise RuntimeError("Unable to lookup transform from ariaWorld to spotWorld")
        self.spotWorld_T_ariaWorld = generate_spSE3_a_T_b_from_TransformStamped(
            transform_stamped_spotWorld_T_ariaWorld
        )

        # Initialize poses as None
        self.spotWorld_T_aria = None
        self.spotWorld_T_aria_pose_of_interest = None
        self._act_flag = False

        # Initialize subscribers for listening to aria w.r.t ariaWorld transform and aria pose of interest w.r.t ariaWorld transform
        self.ariaWorld_T_aria_subscriber = rospy.Subscriber(
            "/aria_current_pose", PoseStamped, self.ariaWorld_T_aria_callback
        )
        self.ariaWorld_T_aria_pose_of_interest_subscriber = rospy.Subscriber(
            "/aria_pose_of_interest",
            PoseStamped,
            self.ariaWorld_T_aria_pose_of_interest_callback,
        )
        self._demo_trigger_sub = rospy.Subscriber(
            "/demo_trigger", Bool, self.demo_trigger_callback
        )

        # Run in a loop until both poses are received
        while not rospy.is_shutdown() and (
            self.spotWorld_T_aria is None
            or self.spotWorld_T_aria_pose_of_interest is None
        ):
            if self.spotWorld_T_aria is None:
                rospy.logwarn_throttle(
                    5.0, "Waiting for transform from aria to ariaWorld"
                )
            if self.spotWorld_T_aria_pose_of_interest is None:
                rospy.logwarn_throttle(
                    5.0, "Waiting for transform from aria pose of interest to ariaWorld"
                )
            rospy.sleep(1.0)

        while not rospy.is_shutdown() and not self._act_flag:
            rospy.logwarn_throttle(5.0, "Waiting for demo trigger")
            rospy.sleep(1.0)

        while not rospy.is_shutdown():
            if (
                self.spotWorld_T_aria is not None
                and self.spotWorld_T_aria_pose_of_interest is not None
                and self._act_flag
            ):
                print("Got both poses Will Start Nav-Pick-Nav-Place")

                # Nav to Pose of interest (Aria's cpf frame when it saw the object)
                nav_loc_for_pick = self.get_nav_pose_to_object(
                    self.spotWorld_T_aria_pose_of_interest
                )
                rospy.loginfo(f"Nav loc for pick: {nav_loc_for_pick}")

                # Pick the object
                pick_object = "bottle"

                # Nav to Pose of interest (Aria wearer's last location)
                nav_loc_for_place = self.get_nav_pose_to_wearer(self.spotWorld_T_aria)
                rospy.loginfo(f"Nav loc for place: {nav_loc_for_place}")
                # Place the object
                place_loc = self.get_place_pose_to_wearer(self.spotWorld_T_aria)

                # Run Nav-Pick-Nav-Place
                self.run_nav_pick_nav_place(
                    nav_loc_for_pick, pick_object, nav_loc_for_place, place_loc
                )

                input("Press Enter to continue...")

            else:
                rospy.WARN(
                    f"Waiting for {self.spotWorld_T_aria} and {self.spotWorld_T_aria_pose_of_interest}"
                )

    def demo_trigger_callback(self, msg):
        """
        Save act_flag as True when demo is triggered
        """
        self._act_flag = True

    def ariaWorld_T_aria_callback(self, msg):
        """
        TODO: Add docstring
        """
        rospy.logdebug("Received message on ariaWorld_T_aria topic")
        ariaWorld_T_aria = generate_spSE3_a_T_b_from_PoseStamped(msg)
        self.spotWorld_T_aria = self.spotWorld_T_ariaWorld * ariaWorld_T_aria
        rospy.logdebug(f"Got {self.spotWorld_T_aria}")

    def ariaWorld_T_aria_pose_of_interest_callback(self, msg):
        """
        TODO: Add docstring
        """
        rospy.logdebug("Received message on ariaWorld_T_aria_pose_of_interest topic")
        ariaWorld_T_aria_pose_of_interest = generate_spSE3_a_T_b_from_PoseStamped(msg)
        self.spotWorld_T_aria_pose_of_interest = (
            self.spotWorld_T_ariaWorld * ariaWorld_T_aria_pose_of_interest
        )
        rospy.logdebug(f"Got {self.spotWorld_T_aria_pose_of_interest}")

    def get_nav_pose_to_object(self, aria_pose: sp.SE3) -> Tuple[float, float, float]:
        """
        TODO: Add docstring
        aria_pose is in ARIA CPF frame
        """
        # Position is obtained from the translation component of the pose
        position = aria_pose.translation()

        # Find the angle made by CPF's z axis with spotWorld's x axis
        # as robot should orient to the CPF's z axis. First 3 elements of
        # column 3 from spotWorld_T_cpf represents cpf's z axis in spotWorld frame
        cpf_z_axis_in_spotWorld = aria_pose.matrix()[:3, 2]

        # Project cpf's z axis onto xy plane of spotWorld frame by ignoring z component
        xy_plane_projection_array = np.array([1.0, 1.0, 0.0])
        projected_cpf_z_axis_in_spotWorld_xy = np.multiply(
            cpf_z_axis_in_spotWorld, xy_plane_projection_array
        )
        orientation = float(
            np.arctan2(
                projected_cpf_z_axis_in_spotWorld_xy[1],
                projected_cpf_z_axis_in_spotWorld_xy[0],
            )
        )  # tan^-1(y/x)

        # Publish pick pose in spotWorld frame for 2 seconds so it can be seen in rviz
        start_time = rospy.Time.now()
        while rospy.Time.now() - start_time < rospy.Duration(2.0):
            self._goto_pose_pub_for_pick.publish(
                generate_PoseStamped_a_T_b_from_spSE3(
                    a_Tform_b=xyt2sophus(
                        np.array([position[0], position[1], orientation])
                    ),
                    parent_frame="spotWorld",
                )
            )
            rospy.sleep(0.1)

        return (position[0], position[1], orientation)

    def get_nav_pose_to_wearer(
        self, aria_pose: sp.SE3, shift_offset: float = 0.6
    ) -> Tuple[float, float, float]:
        """
        Simple logic to select a point 1 m in front of the wearer, rotated towards them
        aria_pose is in ARIA CPF frame
        """
        # get position and rotation as x, y, theta
        position = aria_pose.translation()
        # Find the angle made by CPF's z axis with spotWorld's x axis
        # as robot should orient to the CPF's z axis. First 3 elements of
        # column 3 from spotWorld_T_cpf represents cpf's z axis in spotWorld frame
        cpf_z_axis_in_spotWorld = aria_pose.matrix()[:3, 2]

        # Project cpf's z axis onto xy plane of spotWorld frame by ignoring z component
        xy_plane_projection_array = np.array([1.0, 1.0, 0.0])
        projected_cpf_z_axis_in_spotWorld_xy = np.multiply(
            cpf_z_axis_in_spotWorld, xy_plane_projection_array
        )
        rotation = float(
            np.arctan2(
                projected_cpf_z_axis_in_spotWorld_xy[1],
                projected_cpf_z_axis_in_spotWorld_xy[0],
            )
        )  # tan^-1(y/x)
        x, y, theta = position[0], position[1], rotation

        # push fwd this point along theta
        offset = shift_offset
        x += offset * np.cos(theta)
        y += offset * np.sin(theta)
        # rotate theta by pi
        theta += np.pi

        # Publish place pose in spotWorld frame for 2 seconds so it can be seen in rviz
        start_time = rospy.Time.now()
        while rospy.Time.now() - start_time < rospy.Duration(2.0):
            self._goto_pose_pub_for_place.publish(
                generate_PoseStamped_a_T_b_from_spSE3(
                    a_Tform_b=xyt2sophus(np.array([x, y, theta])),
                    parent_frame="spotWorld",
                )
            )
            rospy.sleep(0.1)

        return x, y, theta

    def get_place_pose_to_wearer(self, aria_pose: sp.SE3) -> Tuple[float, float, float]:
        """
        TODO: Add docstring
        aria_pose is in ARIA CPF frame
        """
        # TODO: Device this logic
        # return (2.079572916030884, 2.289151906967163, 0.7599455118179321)
        return (
            0.8,
            0.0,
            0.8,
        )  # This is constant place target in local frame (TODO: Please test)

    def run_nav_pick_nav_place(
        self,
        nav_for_pick: Tuple[float, float, float],
        pick_object: str,
        nav_for_place: Tuple[float, float, float],
        place_loc: Tuple[float, float, float],
    ):
        """ """
        self.skill_manager.nav(nav_for_pick[0], nav_for_pick[1], nav_for_pick[2])
        self.skill_manager.pick(pick_object)
        self.skill_manager.nav(nav_for_place[0], nav_for_place[1], nav_for_place[2])
        self.skill_manager.place(
            place_loc[0], place_loc[1], place_loc[2], is_local=True
        )


@click.command()
@click.option("--data-path", help="Path to the data directory", type=str)
@click.option("--vrs-name", help="Name of the vrs file", type=str)
@click.option("--dry-run", type=bool, default=False)
@click.option("--verbose", type=bool, default=True)
@click.option("--use-spot/--no-spot", default=True)
@click.option("--object-names", type=str, multiple=True, default=["smartphone"])
@click.option("--qr/--no-qr", default=True)
def main(
    data_path: str,
    vrs_name: str,
    dry_run: bool,
    verbose: bool,
    use_spot: bool,
    object_names: List[str],
    qr: bool,
):
    # rospy.init_node('SpotAriaP1Node', anonymous=False) # Issue is SpotSkillManager already initializes a node
    spot = Spot("SomeRandomNode")
    _ = SpotAriaP1Node(spot=spot, verbose=verbose)


if __name__ == "__main__":
    main()
