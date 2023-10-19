# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import numpy as np
from spot_rl.envs.gaze_env import GazeController, construct_config_for_gaze
from spot_rl.envs.nav_env import WaypointController, construct_config_for_nav
from spot_rl.envs.place_env import PlaceController, construct_config_for_place
from spot_rl.utils.geometry_utils import (
    is_pose_within_bounds,
    is_position_within_bounds,
)
from spot_rl.utils.utils import (
    get_waypoint_yaml,
    nav_target_from_waypoint,
    place_target_from_waypoint,
)
from spot_wrapper.spot import Spot


class SpotSkillManager:
    def __init__(
        self,
        nav_config=None,
        pick_config=None,
        place_config=None,
    ):
        # Create the spot object, init lease, and construct configs
        self.__init_spot(nav_config, pick_config, place_config)

        # Initiate the controllers for nav, gaze, and place
        self.__initiate_controllers()

        # Power on the robot
        self.spot.power_robot()

        # Create a local waypoint dictionary
        self.waypoints_yaml_dict = get_waypoint_yaml()

    def __del__(self):
        pass
        # if self.shutdownndock_on_delete:
        #     self.spot.shutdown(should_dock=True)

    def __init_spot(self, nav_config=None, pick_config=None, place_config=None):
        """
        Initialize the Spot object, acquire lease, and construct configs
        """

        # Create Spot object
        self.spot = Spot("RealSeqEnv")

        # Acquire spot's lease
        self.lease = self.spot.get_lease(hijack=True)
        if not self.lease:
            print("Failed to get lease for Spot. Exiting!")
            exit(1)

        # Construct configs for nav, gaze, and place
        self.nav_config = (
            construct_config_for_nav() if nav_config is None else nav_config
        )
        self.pick_config = (
            construct_config_for_gaze() if pick_config is None else pick_config
        )
        self.place_config = (
            construct_config_for_place() if place_config is None else place_config
        )

    def __initiate_controllers(self):
        """
        Initiate the controllers for nav, gaze, and place
        """

        self.nav_controller = WaypointController(
            config=self.nav_config,
            spot=self.spot,
            should_record_trajectories=True,
        )

        self.gaze_controller = GazeController(config=self.pick_config, spot=self.spot)
        self.place_controller = PlaceController(
            config=self.place_config, spot=self.spot, use_policies=False
        )

    def reset(self):
        # Reset the policies and environments via the controllers
        raise NotImplementedError

    def nav(self, nav_target: str = None) -> Tuple[bool, str]:
        """
        Perform the nav action on the specified navigation target

        Args:
            nav_target (str): Name of the nav target

        Returns:
            bool: True if navigation was successful, False otherwise
            str: Message indicating the status of the navigation
        """
        if nav_target is not None:
            try:
                nav_target_list = [
                    nav_target_from_waypoint(nav_target, self.waypoints_yaml_dict)
                ]

            except Exception:
                return (
                    False,
                    f"Failed - nav target {nav_target} not found - use the exact name",
                )
        else:
            return False, "No nav target specified, skipping nav"

        print(f"Navigating to {nav_target}")

        result = None
        try:
            result = self.nav_controller.execute(nav_target_list)
        except Exception:
            return False, "Error encountered while navigating"

        _nav_target_pose = nav_target_list[0]
        # Make the angle from rad to deg
        _nav_target_pose_deg = (
            _nav_target_pose[0],
            _nav_target_pose[1],
            np.rad2deg(_nav_target_pose[2]),
        )
        check_navigation_suc = is_pose_within_bounds(
            result[0][-1].get("pose"),
            _nav_target_pose_deg,
            self.nav_config.SUCCESS_DISTANCE,
            self.nav_config.SUCCESS_ANGLE_DIST,
        )

        if check_navigation_suc:
            return True, "Successfully reached the target pose by default"
        else:
            return False, "Navigation failed to reach the target pose"

    def pick(self, pick_target: str = None) -> Tuple[bool, str]:
        """
        Perform the pick action on the specified pick target

        Args:
            pick_target (str): Name of the pick target

        Returns:
            bool: True if pick was successful, False otherwise
            str: Message indicating the status of the pick
        """
        if pick_target is None:
            print("No pick target specified, skipping pick")
            return False, "No pick target specified, skipping pick"

        print(f"Picking {pick_target}")

        result = None
        try:
            result = self.gaze_controller.execute([pick_target])
        except Exception as e:
            return (
                False,
                f"Pick failed to pick the target object {result[0]['target_object']} due to error : {e}",
            )

        # Check for success and return appropriately
        if result[0].get("success"):
            return (
                True,
                f"Successfully picked the target object {result[0]['target_object']} in {result[0]['time_taken']} secs",
            )
        else:
            return (
                False,
                f"Pick failed to pick the target object {result[0]['target_object']}",
            )

    def place(self, place_target: str = None) -> Tuple[bool, str]:
        """
        Perform the place action on the specified place target

        Args:
            place_target (str): Name of the place target

        Returns:
            bool: True if place was successful, False otherwise
            str: Message indicating the status of the place
        """
        if place_target is not None:
            # Get the place target coordinates
            try:
                place_target_list = [
                    place_target_from_waypoint(place_target, self.waypoints_yaml_dict)
                ]
            except Exception:
                return (
                    False,
                    f"Failed - nav target {place_target} not found - use the exact name",
                )
        else:
            return False, "No place target specified, skipping place"

        print(f"Place target object at {place_target} i.e. {place_target_list}")

        result = None
        try:
            result = self.place_controller.execute(place_target_list)
        except Exception:
            return False, "Error encountered while placing"

        # Check for success and return appropriately
        if is_position_within_bounds(
            result[0].get("ee_pos"),
            result[0].get("place_target"),
            self.place_config.SUCC_XY_DIST,
            self.place_config.SUCC_Z_DIST,
            convention="spot",
        ):
            return True, "Successfully reached the target position"
        else:
            return False, "Place failed to reach the target position"

    def get_env(self):
        "Get the env for the ease of the access"
        return self.nav_controller.nav_env

    def dock(self):
        # Stow back the arm
        self.get_env().reset_arm()

        try:
            # Navigate to the dock
            self.nav("dock")

            # Dock
            self.spot.shutdown(should_dock=True)
        except Exception:
            return False, "Error encountered while docking"

        return True, "Success docking"


if __name__ == "__main__":
    spotskillmanager = SpotSkillManager()

    # Nav-Pick-Nav-Place sequence 1
    spotskillmanager.nav("test_square_vertex1")
    spotskillmanager.pick("penguin_plush")
    spotskillmanager.nav("test_place_left")
    spotskillmanager.place("test_place_front")

    # # Nav-Pick-Nav-Place sequence 2
    spotskillmanager.nav("test_square_vertex3")
    spotskillmanager.pick("caterpillar_plush")
    spotskillmanager.nav("test_place_right")
    spotskillmanager.place("test_place_right")

    # Navigate to dock and shutdown
    spotskillmanager.dock()
