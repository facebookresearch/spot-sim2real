# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import subprocess
import time
from collections import Counter
from typing import Any, Dict, Tuple

import numpy as np
import rospy
import sophus as sp
from geometry_msgs.msg import PoseStamped
from hydra import compose, initialize
from perception_and_utils.utils.conversions import (
    ros_PoseStamped_to_sophus_SE3,
    ros_TransformStamped_to_sophus_SE3,
    sophus_SE3_to_ros_PoseStamped,
    sophus_SE3_to_ros_TransformStamped,
    xyt_to_sophus_SE3,
)
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.llm.src.rearrange_llm import RearrangeEasyChain
from spot_rl.models.sentence_similarity import SentenceSimilarity
from spot_rl.utils.utils import construct_config, get_default_parser, get_waypoint_yaml
from spot_rl.utils.utils import ros_frames as rf
from spot_rl.utils.whisper_translator import WhisperTranslator
from spot_wrapper.spot import Spot, SpotCamIds
from spot_wrapper.spot_qr_detector import SpotQRDetector
from tf2_ros import (
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    StaticTransformBroadcaster,
)
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 549))

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


def get_nav_xyt_to_wearer(
    pose: sp.SE3, shift_offset: float = 0.6
) -> Tuple[float, float, float]:
    """
    Converts Sophus SE3 aria current pose to Tuple of x, y, theta, then flips it by 180 degrees and shifts it by a given offset.
    At the end, the final x, y, theta is such that the robot will face the wearer.

    Args:
        aria_pose (sp.SE3): Sophus SE3 object for aria's current pose (as cpf frame) w.r.t spotWorld frame

    Returns:
        Tuple[float, float, float]: Tuple of x,y,theta as floats representing nav target for robot
    """
    # ARROW STARTS FROM ORIGIN

    # get position and rotation as x, y, theta
    position = pose.translation()
    # Find the angle made by CPF's z axis with spotWorld's x axis
    # as robot should orient to the CPF's z axis. First 3 elements of
    # column 3 from spotWorld_T_cpf represents cpf's z axis in spotWorld frame
    cpf_z_axis_in_spotWorld = pose.matrix()[:3, 2]
    x_component_of_z_axis = cpf_z_axis_in_spotWorld[0]
    y_component_of_z_axis = cpf_z_axis_in_spotWorld[1]
    rotation = float(
        np.arctan2(
            y_component_of_z_axis,
            x_component_of_z_axis,
        )
    )  # tan^-1(y/x)
    x, y, theta = position[0], position[1], rotation

    # push fwd this point along theta
    x += shift_offset * np.cos(theta)
    y += shift_offset * np.sin(theta)
    # rotate theta by pi
    theta += np.pi

    return (x, y, theta)


def get_place_xyz_to_wearer(pose: sp.SE3) -> Tuple[float, float, float]:
    """
    Returns the xyz coordinates of the place pose in spot's local frame. The x,y,z coordinates
    are present in the local frame of the robot. Make sure to pass the "is_local" flag as true to skill_manager.place()

    Args:
        aria_pose (sp.SE3): Sophus SE3 object for aria's current pose (as cpf frame) w.r.t spotWorld frame

    Returns:
        Tuple[float, float, float]: Tuple of x,y,z as floats representing place pose in spot's local frame
    """
    return (
        0.7,
        0.0,
        0.2,
    )


def get_handoff_to_human_pose(tf_buffer: Buffer, source: str, target: str) -> sp.SE3:
    while not rospy.is_shutdown() and not tf_buffer.can_transform(
        target_frame=target, source_frame=source, time=rospy.Time()
    ):
        rospy.logwarn_throttle(5.0, f"Waiting for transform from {source} to {target}")
        rospy.sleep(0.5)
    try:
        transform_stamped_spotWorld_T_camera = tf_buffer.lookup_transform(
            target_frame=target,
            source_frame=source,
            time=rospy.Time(0),
        )
    except (LookupException, ConnectivityException, ExtrapolationException):
        raise RuntimeError(f"Unable to lookup transform from {source} to {target}")
    target_T_source = ros_TransformStamped_to_sophus_SE3(
        ros_trf_stamped=transform_stamped_spotWorld_T_camera
    )
    return target_T_source


def main(spot, config):

    spot = SpotSkillManager(use_mobile_pick=True)

    vp = SPOT_DOCK_OBSERVER_WAYPOINT_RIGHT
    status, msg = spot.nav(
        vp[0],
        vp[1],
        np.deg2rad(vp[2]),
    )
    if not status:
        rospy.logerr(
            f"Failed to navigate to spot dock observer waypoint. Error: {msg}. Exiting..."
        )
        return
    # Sit Spot down
    spot.sit()

    cam_id = SpotCamIds.HAND_COLOR
    spot_qr = SpotQRDetector(spot=spot.spot, cam_ids=[cam_id])
    avg_spotWorld_T_marker = spot_qr.get_avg_spotWorld_T_marker(cam_id=cam_id)

    # nav_PoseStamped_pub_for_place = rospy.Publisher(
    #     "/nav_pose_for_place_viz", PoseStamped, queue_size=10
    # )
    # Publish marker w.r.t spotWorld transforms for 5 seconds so it can be seen in rviz
    # Instantiate static transform broadcaster for publishing marker w.r.t spotWorld transforms
    static_tf_broadcaster = StaticTransformBroadcaster()
    tf_buffer = Buffer()
    _ = TransformListener(tf_buffer)
    start_time = rospy.Time.now()
    while rospy.Time.now() - start_time < rospy.Duration(2.0):
        static_tf_broadcaster.sendTransform(
            sophus_SE3_to_ros_TransformStamped(
                sp_se3=avg_spotWorld_T_marker,
                parent_frame=rf.SPOT_WORLD,
                child_frame=rf.MARKER,
            )
        )
        rospy.sleep(0.5)

    # Reset the viz params
    rospy.set_param("/viz_pick", "None")
    rospy.set_param("/viz_object", "None")
    rospy.set_param("/viz_place", "None")

    # Check if robot should return to base
    return_to_base = config.RETURN_TO_BASE

    # Get the waypoints from waypoints.yaml
    waypoints_yaml_dict = get_waypoint_yaml()

    audio_to_text = WhisperTranslator()
    sentence_similarity = SentenceSimilarity()
    with initialize(config_path="../llm/src/conf"):
        llm_config = compose(config_name="config")
    llm = RearrangeEasyChain(llm_config)

    print(
        "I am ready to take instructions!\n Sample Instructions : take the rubik cube from the dining table to the hamper"
    )
    print("-" * 100)
    pre_in_dock = False
    while True:
        audio_transcription_success = False
        while not audio_transcription_success:
            # try:
            input("Are you ready?")
            audio_to_text.record()
            instruction = audio_to_text.translate()
            print("Transcribed instructions : ", instruction)

            # Use LLM to convert user input to an instructions set
            # Eg: nav_1, pick, nav_2 = 'bowl_counter', "container", 'coffee_counter'
            t1 = time.time()
            nav_1, pick, nav_2, _ = llm.parse_instructions(instruction)
            print("PARSED", nav_1, pick, nav_2, f"Time {time.time()- t1} secs")

            # Find closest nav_targets to the ones robot knows locations of
            nav_1 = sentence_similarity.get_most_similar_in_list(
                nav_1, list(waypoints_yaml_dict["nav_targets"].keys())
            )
            nav_2 = sentence_similarity.get_most_similar_in_list(
                nav_2, list(waypoints_yaml_dict["nav_targets"].keys())
            )
            print("MOST SIMILAR: ", nav_1, pick, nav_2)
            audio_transcription_success = True
            # except Exception as e:
            #     print(f"Exception encountered in Speech to text : {e} \n\n Retrying...")
        # Used for Owlvit
        rospy.set_param("object_target", pick)

        # Used for Visualizations
        rospy.set_param("viz_pick", nav_1)
        rospy.set_param("viz_object", pick)
        rospy.set_param("viz_place", nav_2)

        if pre_in_dock:
            spot.spot.power_robot()

        input("Start LSC-HITL?")
        # Do LSC by calling skills
        slow_down = 0.5  # slow down time to increase the skill execution stability
        spot.nav(nav_1)
        time.sleep(slow_down)
        spot.pick(pick)
        time.sleep(slow_down)
        if nav_2 == "human":
            human_pose = get_handoff_to_human_pose(
                tf_buffer, source="camera", target="spotWorld"
            )
            handoff_xyt = get_nav_xyt_to_wearer(human_pose)
            spot.nav(handoff_xyt[0], handoff_xyt[1], handoff_xyt[2])
            time.sleep(slow_down)

            handoff_xyz = get_place_xyz_to_wearer(human_pose)
            spot.place(handoff_xyz[0], handoff_xyz[1], handoff_xyz[2], is_local=True)
            # spot.place( is_local=True, visualization=True)
            time.sleep(slow_down)
        else:
            spot.nav(nav_2)
            time.sleep(slow_down)
            spot.place(nav_2)
            time.sleep(slow_down)
        if return_to_base:
            spot.dock()
            pre_in_dock = True


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("-m", "--use-mixer", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = construct_config(opts=args.opts)

    spot = Spot("LSC_HITLNode")
    main(None, config)
