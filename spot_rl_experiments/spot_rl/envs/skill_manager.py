# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple

import numpy as np
from multimethod import multimethod
from spot_rl.envs.gaze_env import GazeController, construct_config_for_gaze
from spot_rl.envs.nav_env import WaypointController, construct_config_for_nav
from spot_rl.envs.place_env import PlaceController, construct_config_for_place
from spot_rl.utils.geometry_utils import (
    is_pose_within_bounds,
    is_position_within_bounds,
)
from spot_rl.utils.utils import (
    conditional_print,
    get_waypoint_yaml,
    nav_target_from_waypoint,
    place_target_from_waypoint,
)
from spot_wrapper.spot import Spot


class SpotSkillManager:
    """
    Interface class to invoke skills for Spot.
    Exposes skills like nav, pick, place, and dock as functions

    Args:
        None

    How to use:
        1. Create the SpotSkillManager object
        2. Call the skill functions as needed (nav, pick, place, dock)

    Examples:
        # Create the spot skill manager object (this will create spot object, init lease, and construct configs and power on the robot)
        spotskillmanager = SpotSkillManager()

        # Skill - Navigation
        # To navigate to a waypoint (str)
        status, msg = spotskillmanager.nav("test_square_vertex1")

        # To navigate to a waypoint (x, y, theta)
        status, msg = spotskillmanager.nav(x, y, theta)

        # Skill - Pick
        # To pick an object
        status, msg = spotskillmanager.pick("ball_plush")

        # Skill - Place
        # To place an object at a waypoint (str)
        status, msg = spotskillmanager.place("test_place_front")

        # To place an object at a location (x, y, z)
        status, msg = spotskillmanager.place(x, y, z)

        # To dock
        status, msg = spotskillmanager.dock()

        # This can can be used for multiple skills in a sequence like nav-pick-nav-place
        # Nav-Pick-Nav-Place sequence 1
        spotskillmanager.nav("test_square_vertex1")
        spotskillmanager.pick("ball_plush")
        spotskillmanager.nav("test_place_front")
        spotskillmanager.place("test_place_front")
    """

    def __init__(self, use_mobile_pick=False):
        # Process the meta parameters
        self._use_mobile_pick = use_mobile_pick

        # Create the spot object, init lease, and construct configs
        self.__init_spot()

        # Initiate the controllers for nav, gaze, and place
        self.__initiate_controllers()

        # Power on the robot
        self.spot.power_robot()

        # Create a local waypoint dictionary
        self.waypoints_yaml_dict = get_waypoint_yaml()

    def __del__(self):
        pass

    def __init_spot(self):
        """
        Initialize the Spot object, acquire lease, and construct configs
        """
        # Create Spot object
        self.spot = Spot("RealSeqEnv")

        # Acquire spot's lease
        self.lease = self.spot.get_lease(hijack=True)
        if not self.lease:
            conditional_print(
                message="Failed to get lease for Spot. Exiting!", verbose=self.verbose
            )
            exit(1)

        # Construct configs for nav, gaze, and place
        self.nav_config = construct_config_for_nav()
        self.pick_config = construct_config_for_gaze(max_episode_steps=350)
        self.place_config = construct_config_for_place()

        # Set the verbose flag (from any of the configs)
        self.verbose = self.nav_config.VERBOSE

    def __initiate_controllers(self):
        """
        Initiate the controllers for nav, gaze, and place
        """

        self.nav_controller = WaypointController(
            config=self.nav_config,
            spot=self.spot,
            should_record_trajectories=True,
        )
        self.gaze_controller = GazeController(
            config=self.pick_config,
            spot=self.spot,
            use_mobile_pick=self._use_mobile_pick,
        )
        self.place_controller = PlaceController(
            config=self.place_config, spot=self.spot, use_policies=False
        )

    def reset(self):
        # Reset the policies and environments via the controllers
        raise NotImplementedError

    @multimethod  # type: ignore
    def nav(self, nav_target: str = None) -> Tuple[bool, str]:  # type: ignore
        """
        Perform the nav action on the navigation target specified as a known string

        Args:
            nav_target (str): Name of the nav target (as stored in waypoints.yaml)

        Returns:
            bool: True if navigation was successful, False otherwise
            str: Message indicating the status of the navigation
        """
        conditional_print(
            message=f"Received nav target request for - {nav_target}",
            verbose=self.verbose,
        )

        if nav_target is not None:
            # Get the nav target coordinates
            try:
                nav_target_tuple = nav_target_from_waypoint(
                    nav_target, self.waypoints_yaml_dict
                )
            except Exception:
                message = (
                    f"Failed - nav target {nav_target} not found - use the exact name"
                )
                conditional_print(message=message, verbose=self.verbose)
                return False, message
        else:
            msg = "No nav target specified, skipping nav"
            return False, msg

        nav_x, nav_y, nav_theta = nav_target_tuple
        status, message = self.nav(nav_x, nav_y, nav_theta)
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    @multimethod  # type: ignore
    def nav(self, x: float, y: float, theta=float) -> Tuple[bool, str]:  # noqa
        """
        Perform the nav action on the navigation target specified as a metric location

        Args:
            x (float): x coordinate of the nav target (in meters) specified in the world frame
            y (float): y coordinate of the nav target (in meters) specified in the world frame
            theta (float): yaw for the nav target (in radians) specified in the world frame

        Returns:
            bool: True if navigation was successful, False otherwise
            str: Message indicating the status of the navigation
        """

        conditional_print(
            message=f"Navigating to x, y, theta : {x}, {y}, {theta}",
            verbose=self.verbose,
        )

        result = None
        nav_target_tuple = None
        try:
            nav_target_tuple = (x, y, theta)
            result = self.nav_controller.execute([nav_target_tuple])
        except Exception:
            message = "Error encountered while navigating"
            conditional_print(message=message, verbose=self.verbose)
            return False, message

        # Make the angle from rad to deg
        _nav_target_pose_deg = (
            nav_target_tuple[0],
            nav_target_tuple[1],
            np.rad2deg(nav_target_tuple[2]),
        )
        check_navigation_suc = is_pose_within_bounds(
            result[0][-1].get("pose"),
            _nav_target_pose_deg,
            self.nav_config.SUCCESS_DISTANCE,
            self.nav_config.SUCCESS_ANGLE_DIST,
        )

        # Check for success and return appropriately
        status = False
        message = "Navigation failed to reach the target pose"
        if check_navigation_suc:
            status = True
            message = "Successfully reached the target pose by default"
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    def pick(self, pick_target: str = None) -> Tuple[bool, str]:
        """
        Perform the pick action on the pick target specified as string

        Args:
            pick_target (str): Descriptive name of the pick target (eg: ball_plush)

        Returns:
            bool: True if pick was successful, False otherwise
            str: Message indicating the status of the pick
        """
        conditional_print(
            message=f"Received pick target request for - {pick_target}",
            verbose=self.verbose,
        )

        if pick_target is None:
            message = "No pick target specified, skipping pick"
            conditional_print(message=message, verbose=self.verbose)
            return False, message

        conditional_print(message=f"Picking {pick_target}", verbose=self.verbose)

        result = None
        try:
            result = self.gaze_controller.execute([pick_target])
        except Exception:
            message = "Error encountered while picking"
            conditional_print(message=message, verbose=self.verbose)
            return False, message

        # Check for success and return appropriately
        status = False
        message = "Pick failed to pick the target object"
        if result[0].get("success"):
            status = True
            message = "Successfully picked the target object"
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    @multimethod  # type: ignore
    def place(self, place_target: str = None) -> Tuple[bool, str]:  # type: ignore
        """
        Perform the place action on the place target specified as known string

        Args:
            place_target (str): Name of the place target (as stored in waypoints.yaml)

        Returns:
            bool: True if place was successful, False otherwise
            str: Message indicating the status of the place
        """
        conditional_print(
            message=f"Received place target request for - {place_target}",
            verbose=self.verbose,
        )

        if place_target is not None:
            # Get the place target coordinates
            try:
                place_target_location = place_target_from_waypoint(
                    place_target, self.waypoints_yaml_dict
                )
            except Exception:
                message = f"Failed - place target {place_target} not found - use the exact name"
                conditional_print(message=message, verbose=self.verbose)
                return False, message
        else:
            message = "No place target specified, skipping place"
            conditional_print(message=message, verbose=self.verbose)
            return False, message

        place_x, place_y, place_z = place_target_location.astype(np.float64).tolist()
        status, message = self.place(place_x, place_y, place_z)
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    @multimethod  # type: ignore
    def place(self, x: float, y: float, z: float) -> Tuple[bool, str]:  # noqa
        """
        Perform the place action on the place target specified as metric location

        Args:
            x (float): x coordinate of the place target (in meters) specified in the world frame
            y (float): y coordinate of the place target (in meters) specified in the world frame
            z (float): z coordinate of the place target (in meters) specified in the world frame

        Returns:
            bool: True if place was successful, False otherwise
            str: Message indicating the status of the place
        """
        conditional_print(
            message=f"Place target object at x, y, z : {x}, {y}, {z}",
            verbose=self.verbose,
        )

        result = None
        try:
            place_target_tuple = (x, y, z)
            result = self.place_controller.execute([place_target_tuple])
        except Exception:
            message = "Error encountered while placing"
            conditional_print(message=message, verbose=self.verbose)
            return False, message

        # Check for success and return appropriately
        status = False
        message = "Place failed to reach the target position"
        if is_position_within_bounds(
            result[0].get("ee_pos"),
            result[0].get("place_target"),
            self.place_config.SUCC_XY_DIST,
            self.place_config.SUCC_Z_DIST,
            convention="spot",
        ):
            status = True
            message = "Successfully reached the target position"
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    def get_env(self):
        "Get the env for the ease of the access"
        return self.nav_controller.nav_env

    def dock(self):
        # Stow back the arm
        self.get_env().reset_arm()

        status = False
        message = "Dock failed"
        try:
            # Navigate to the dock
            status, message = self.nav("dock")

            # Dock
            self.spot.shutdown(should_dock=True)
        except Exception:
            message = "Error encountered while docking"
            conditional_print(message=message, verbose=self.verbose)
            return status, message

        if status:
            message = "Successfully docked"
        return status, message


if __name__ == "__main__":

    # We initialize the skill using SpotSkillManager.
    # Note that if you want to use mobile gaze for pick,
    # instead of static gaze, you need to do
    # SpotSkillManager(use_mobile_pick=True)
    # spotskillmanager = SpotSkillManager()

    # TODO: the folllowing code will be removed
    spotskillmanager = SpotSkillManager(use_mobile_pick=True)
    spotskillmanager.nav("nyc_mg_pos1")
    spotskillmanager.pick("cup")
    breakpoint()

    # # Nav-Pick-Nav-Place sequence 1
    # spotskillmanager.nav("test_square_vertex1")
    # spotskillmanager.pick("ball_plush")
    # spotskillmanager.nav("test_place_front")
    # spotskillmanager.place("test_place_front")

    # # Nav-Pick-Nav-Place sequence 2
    # spotskillmanager.nav("test_square_vertex3")
    # spotskillmanager.pick("caterpillar_plush")
    # spotskillmanager.nav("test_place_left")
    # spotskillmanager.place("test_place_left")

    # Navigate to dock and shutdown
    spotskillmanager.dock()
