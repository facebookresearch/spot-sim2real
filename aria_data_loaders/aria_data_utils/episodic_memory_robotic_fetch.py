# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import rospy

try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp
from geometry_msgs.msg import PoseStamped
from perception_and_utils.utils.conversions import (
    ros_PoseStamped_to_sophus_SE3,
    ros_TransformStamped_to_sophus_SE3,
    sophus_SE3_to_ros_PoseStamped,
    sophus_SE3_to_ros_TransformStamped,
    xyt_to_sophus_SE3,
)
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.utils import ros_frames as rf
from spot_wrapper.spot import Spot, SpotCamIds
from spot_wrapper.spot_qr_detector import SpotQRDetector
from std_msgs.msg import Bool
from tf2_ros import (
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    StaticTransformBroadcaster,
)
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# For information on useful reference frame, please check https://github.com/facebookresearch/spot-sim2real?tab=readme-ov-file#eyeglasses-run-spot-aria-project-code

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


class EpisodicMemoryRoboticFetch:
    """
    This class handles the core logic of the Episodic Memory Robotic Fetch
    It begines with Spot looking at the marker and detecting it which results
    in a static tf of marker w.r.t spotWorld frame being published.

    Then it waits for Aria to detect marker and publish ariaWorld w.r.t marker
    as a static tf. This is done by AriaLiveReader node. This step ensures both
    Spot & Aria worlds are now mutually transformable which is necessary for the
    rest of the logic.

    It constantly listens to Aria's current pose w.r.t ariaWorld frame.
    It also waits for Aria to detect the object and publish that pose of interest
    (i.e. aria's cpf frame when it saw the object) w.r.t ariaWorld frame.

    Once both the poses are received, it waits for the spot_fetch trigger to be True
    which will be set by the user via `rostopic pub` on cli.

    Once the trigger is True, it will run the following steps:
        1. Nav to Pose of interest (Aria's cpf frame when it saw the object)
        2. Pick the object (object is hardcoded to be bottle)
        3. Nav to Pose of interest (Aria wearer's last location before triggering spot_fetch)
        4. Place the object (object is hardcoded to be bottle)

    Args:
        spot: Spot object
        verbose: bool indicating whether to print debug logs

    Warning: Do not create objects of this class directly. It is a standalone node
    """

    def __init__(self, spot: Spot, verbose: bool = True, use_policies: bool = True):
        # Get Spot Skill Manager
        # @FIXME: This is initializing a ros-node silently (at core of inheritence in
        # SpotRobotSubscriberMixin). Make it explicit
        self.skill_manager = SpotSkillManager(
            spot=spot,
            verbose=verbose,
            use_policies=use_policies,
        )

        # Navigate Spot to SPOT_DOCK_OBSERVER_WAYPOINT_LEFT so that it can see the marker
        status, msg = self.skill_manager.nav(
            SPOT_DOCK_OBSERVER_WAYPOINT_LEFT[0],
            SPOT_DOCK_OBSERVER_WAYPOINT_LEFT[1],
            np.deg2rad(SPOT_DOCK_OBSERVER_WAYPOINT_LEFT[2]),
        )
        if not status:
            rospy.logerr(
                f"Failed to navigate to spot dock observer waypoint. Error: {msg}. Exiting..."
            )
            return

        # Sit Spot down
        self.skill_manager.sit()

        # Instantiate static transform broadcaster for publishing marker w.r.t spotWorld transforms
        self.static_tf_broadcaster = StaticTransformBroadcaster()

        # Detect the marker and get the average pose of marker w.r.t spotWorld frame
        cam_id = SpotCamIds.HAND_COLOR
        spot_qr = SpotQRDetector(spot=spot, cam_ids=[cam_id])
        avg_spotWorld_T_marker = spot_qr.get_avg_spotWorld_T_marker(cam_id=cam_id)

        # Publish marker w.r.t spotWorld transforms for 5 seconds so it can be seen in rviz
        start_time = rospy.Time.now()
        while rospy.Time.now() - start_time < rospy.Duration(2.0):
            self.static_tf_broadcaster.sendTransform(
                sophus_SE3_to_ros_TransformStamped(
                    sp_se3=avg_spotWorld_T_marker,
                    parent_frame=rf.SPOT_WORLD,
                    child_frame=rf.MARKER,
                )
            )
            rospy.sleep(0.5)

        # Instantiate publisher for publishing go-to pose for pick and place
        self._nav_PoseStamped_pub_for_pick = rospy.Publisher(
            "/nav_pose_for_pick_viz", PoseStamped, queue_size=10
        )
        self._nav_PoseStamped_pub_for_place = rospy.Publisher(
            "/nav_pose_for_place_viz", PoseStamped, queue_size=10
        )
        self.nav_xyt_for_pick = None
        self.nav_xyt_for_handoff = None

        # Initialize static transform subscriber for listening to ariaWorld w.r.t marker transform
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer)

        # Wait for transform from ariaWorld to spotWorld to be available
        while not rospy.is_shutdown() and not self._tf_buffer.can_transform(
            target_frame=rf.SPOT_WORLD, source_frame=rf.ARIA_WORLD, time=rospy.Time(0)
        ):
            rospy.logwarn_throttle(
                5.0, "Waiting for transform from ariaWorld to spotWorld"
            )
            rospy.sleep(0.5)
        try:
            transform_stamped_spotWorld_T_ariaWorld = self._tf_buffer.lookup_transform(
                target_frame=rf.SPOT_WORLD,
                source_frame=rf.ARIA_WORLD,
                time=rospy.Time(0),
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            raise RuntimeError("Unable to lookup transform from ariaWorld to spotWorld")
        self.spotWorld_T_ariaWorld = ros_TransformStamped_to_sophus_SE3(
            ros_trf_stamped=transform_stamped_spotWorld_T_ariaWorld
        )

        # Initialize poses as None
        self.spotWorld_T_aria = None
        self.spotWorld_T_aria_pose_of_interest = None
        self._spot_fetch_flag = False

        # Initialize subscribers for listening to aria w.r.t ariaWorld transform and
        # aria pose of interest w.r.t ariaWorld transform
        self.ariaWorld_T_aria_subscriber = rospy.Subscriber(
            "/aria_current_pose", PoseStamped, self.aria_live_pose_callback
        )
        self.ariaWorld_T_aria_pose_of_interest_subscriber = rospy.Subscriber(
            "/aria_pose_of_interest",
            PoseStamped,
            self.aria_pose_of_interest_callback,
        )
        self._spot_fetch_trigger_sub = rospy.Subscriber(
            "/spot_fetch_trigger", Bool, self.spot_fetch_trigger_callback
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

        while not rospy.is_shutdown() and not self._spot_fetch_flag:
            rospy.logwarn_throttle(
                5.0, "Waiting for demo trigger on /spot_fetch_trigger topic"
            )
            rospy.sleep(1.0)

        # Set run status to true in the beginning
        run_status = True

        while not rospy.is_shutdown() and run_status:
            if (
                self.spotWorld_T_aria is not None
                and self.spotWorld_T_aria_pose_of_interest is not None
                and self._spot_fetch_flag
            ):
                print("Got both poses Will Start Nav-Pick-Nav-Place")

                # Nav to x,y,theta of interest (Aria's cpf frame when it saw the object)
                nav_xyt_for_pick = self.nav_xyt_for_pick
                rospy.loginfo(f"Nav 2D location for pick: {nav_xyt_for_pick}")

                # Pick the object
                pick_object = "bottle"
                rospy.loginfo(f"Pick Object - {pick_object}")

                # Nav to x,y,theta of interest (Aria wearer's last location)
                nav_xyt_for_handoff = self.nav_xyt_for_handoff
                rospy.loginfo(f"Nav 2D location for place: {nav_xyt_for_handoff}")

                # Place the object
                place_xyz_for_handoff = self.get_place_xyz_to_wearer(
                    self.spotWorld_T_aria
                )
                rospy.loginfo(f"Place 3D location for handoff: {place_xyz_for_handoff}")

                # Run Nav-Pick-Nav-Place
                run_status = self.run_nav_pick_nav_place(
                    nav_xyt_for_pick,
                    pick_object,
                    nav_xyt_for_handoff,
                    place_xyz_for_handoff,
                )

                input("Press Enter to continue...")

            else:
                rospy.WARN(
                    f"Waiting for {self.spotWorld_T_aria} and {self.spotWorld_T_aria_pose_of_interest}"
                )

    def spot_fetch_trigger_callback(self, msg):
        """
        Save _spot_fetch_flag as True when spot fetch is to be triggered

        Args:
            msg (Bool): Message received on /spot_fetch_trigger topic

        Updates:
            self._spot_fetch_flag (bool): True when spot fetch is to be triggered
        """
        self._spot_fetch_flag = True

    def aria_live_pose_callback(self, msg):
        """
        ROS subscriber callback to receive Aria's current pose (as cpf frame) w.r.t ariaWorld frame

        Args:
            msg (PoseStamped): Message received on /aria_current_pose topic

        Updates:
            self.spotWorld_T_aria (sp.SE3): Aria's current pose (as cpf frame) w.r.t spotWorld frame

        Publishes:
            nav_PoseStamped_for_handoff (PoseStamped): Pose of wearer's last location in spotWorld frame before triggering spot_fetch
        """
        rospy.logdebug("Received message on ariaWorld_T_aria topic")
        ariaWorld_T_aria = ros_PoseStamped_to_sophus_SE3(ros_pse_stamped=msg)
        self.spotWorld_T_aria = self.spotWorld_T_ariaWorld * ariaWorld_T_aria
        rospy.logdebug(f"Got {self.spotWorld_T_aria}")

        if not self._spot_fetch_flag:
            self.nav_xyt_for_handoff = self.get_nav_xyt_to_wearer(self.spotWorld_T_aria)

            # Publish place pose in spotWorld frame so it can be seen in rviz until spot_fetch is triggered
            self._nav_PoseStamped_pub_for_place.publish(
                sophus_SE3_to_ros_PoseStamped(
                    sp_se3=xyt_to_sophus_SE3(
                        np.asarray(self.nav_xyt_for_handoff, dtype=np.float32)
                    ),
                    parent_frame=rf.SPOT_WORLD,
                )
            )

    def aria_pose_of_interest_callback(self, msg):
        """
        ROS subscriber callback to receive Aria's pose pose of interest (as cpf frame) w.r.t ariaWorld frame where it last saw the object of interest

        Args:
            msg (PoseStamped): Message received on /aria_pose_of_interest topic

        Updates:
            self.spotWorld_T_aria_pose_of_interest (sp.SE3): Aria's pose of interest (as cpf frame) w.r.t spotWorld frame where it last saw the object of interest

        Publishes:
            nav_PoseStamped_for_pcik (PoseStamped): Pose of wearer's location in spotWorld frame where it last saw the object of interest before triggering spot_fetch
        """
        rospy.logdebug("Received message on ariaWorld_T_aria_pose_of_interest topic")
        ariaWorld_T_aria_pose_of_interest = ros_PoseStamped_to_sophus_SE3(
            ros_pse_stamped=msg
        )
        self.spotWorld_T_aria_pose_of_interest = (
            self.spotWorld_T_ariaWorld * ariaWorld_T_aria_pose_of_interest
        )
        rospy.logdebug(f"Got {self.spotWorld_T_aria_pose_of_interest}")

        if not self._spot_fetch_flag:
            self.nav_xyt_for_pick = self.get_nav_xyt_to_object(
                self.spotWorld_T_aria_pose_of_interest
            )

            # Publish pick pose in spotWorld frame so it can be seen in rviz until spot_fetch is triggered
            self._nav_PoseStamped_pub_for_pick.publish(
                sophus_SE3_to_ros_PoseStamped(
                    sp_se3=xyt_to_sophus_SE3(
                        np.asarray(self.nav_xyt_for_pick, dtype=np.float32)
                    ),
                    parent_frame=rf.SPOT_WORLD,
                )
            )

    def get_nav_xyt_to_object(self, aria_pose: sp.SE3) -> Tuple[float, float, float]:
        """
        Converts Sophus SE3 aria pose of interest to Tuple of x, y, theta where robot can navigate for pick.

        Args:
            aria_pose (sp.SE3): Sophus SE3 object for aria's pose (as cpf frame) where it saw the object w.r.t spotWorld

        Returns:
            Tuple[float, float, float]: Tuple of x,y,theta as floats representing nav target for robot
        """
        # Position is obtained from the translation component of the pose
        position = aria_pose.translation()

        # Find the angle made by CPF's z axis with spotWorld's x axis
        # as robot should orient to the CPF's z axis. First 3 elements of
        # column 3 from spotWorld_T_cpf represents cpf's z axis in spotWorld frame
        cpf_z_axis_in_spotWorld = aria_pose.matrix()[:3, 2]
        x_component_of_z_axis = cpf_z_axis_in_spotWorld[0]
        y_component_of_z_axis = cpf_z_axis_in_spotWorld[1]
        orientation = float(
            np.arctan2(
                y_component_of_z_axis,
                x_component_of_z_axis,
            )
        )  # tan^-1(y/x)

        return (position[0], position[1], orientation)

    def get_nav_xyt_to_wearer(
        self, aria_pose: sp.SE3, shift_offset: float = 0.6
    ) -> Tuple[float, float, float]:
        """
        Converts Sophus SE3 aria current pose to Tuple of x, y, theta, then flips it by 180 degrees and shifts it by a given offset.
        At the end, the final x, y, theta is such that the robot will face the wearer.

        Args:
            aria_pose (sp.SE3): Sophus SE3 object for aria's current pose (as cpf frame) w.r.t spotWorld frame

        Returns:
            Tuple[float, float, float]: Tuple of x,y,theta as floats representing nav target for robot
        """
        # get position and rotation as x, y, theta
        position = aria_pose.translation()
        # Find the angle made by CPF's z axis with spotWorld's x axis
        # as robot should orient to the CPF's z axis. First 3 elements of
        # column 3 from spotWorld_T_cpf represents cpf's z axis in spotWorld frame
        cpf_z_axis_in_spotWorld = aria_pose.matrix()[:3, 2]
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
        x += shift_offset * x_component_of_z_axis
        y += shift_offset * y_component_of_z_axis
        # rotate theta by pi
        theta += np.pi

        return (x, y, theta)

    def get_place_xyz_to_wearer(self, aria_pose: sp.SE3) -> Tuple[float, float, float]:
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

    def run_nav_pick_nav_place(
        self,
        nav_xyt_for_pick: Tuple[float, float, float],
        pick_object: str,
        nav_xyt_for_place: Tuple[float, float, float],
        place_xyz_for_handoff: Tuple[float, float, float],
        is_place_local: bool = True,
    ) -> bool:
        """
        Runs spot-fetch task as a sequence of navigation, pick, navigation, place tasks.

        Args:
            nav_xyt_for_pick (Tuple[float, float, float]): Nav target for robot to pick up the object of interest
            pick_object (str): Name of the object of interest to pick up
            nav_xyt_for_place (Tuple[float, float, float]): Nav target for robot to place the object of interest
            place_xyz_for_handoff (Tuple[float, float, float]): Place target for robot to place the object of interest
            is_place_local (bool, optional): Whether the place pose is in spot's local frame or not. Defaults to True.

        Returns:
            None
        """
        status, msg = self.skill_manager.nav(
            nav_xyt_for_pick[0], nav_xyt_for_pick[1], nav_xyt_for_pick[2]
        )
        if not status:
            rospy.logerr(f"Failed to navigate to pick pose. Error: {msg}. Exiting...")
            return False

        status, msg = self.skill_manager.pick(pick_object)
        if not status:
            rospy.logerr(f"Failed to pick up object. Error: {msg}. Exiting...")
            return False

        status, msg = self.skill_manager.nav(
            nav_xyt_for_place[0], nav_xyt_for_place[1], nav_xyt_for_place[2]
        )
        if not status:
            rospy.logerr(f"Failed to navigate to place pose. Error: {msg}. Exiting...")
            return False

        status, msg = self.skill_manager.place(
            place_xyz_for_handoff[0],
            place_xyz_for_handoff[1],
            place_xyz_for_handoff[2],
            is_local=is_place_local,
        )
        if not status:
            rospy.logerr(f"Failed to place object. Error: {msg}. Exiting...")
            return False

        rospy.loginfo("Successfully completed spot-fetch task!")
        return True


@click.command()
@click.option("--verbose", type=bool, default=True)
@click.option("--use-policies", type=bool, default=False)
def main(verbose: bool, use_policies: bool):
    spot = Spot("EpisodicMemoryRoboticFetchNode")

    with spot.get_lease(hijack=True) as lease:
        if lease is None:
            raise RuntimeError("Could not get lease")
        else:
            rospy.loginfo("Acquired lease")

        _ = EpisodicMemoryRoboticFetch(
            spot=spot, verbose=verbose, use_policies=use_policies
        )


if __name__ == "__main__":
    main()
