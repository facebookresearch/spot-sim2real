# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import time
from typing import Dict, List, Tuple

import numpy as np
from spot_rl.envs.gaze_env import SpotGazeEnv

# Import Envs
from spot_rl.envs.nav_env import SpotNavEnv
from spot_rl.envs.place_env import SpotPlaceEnv

# Import policies
from spot_rl.real_policy import GazePolicy, MobileGazePolicy, NavPolicy, PlacePolicy

# Import utils and helpers
from spot_rl.utils.construct_configs import (
    construct_config_for_gaze,
    construct_config_for_nav,
    construct_config_for_place,
)
from spot_rl.utils.geometry_utils import (
    generate_intermediate_point,
    get_RPY_from_vector,
    is_pose_within_bounds,
    is_position_within_bounds,
)
from spot_rl.utils.utils import conditional_print, map_user_input_to_boolean

# Import core classes
from spot_wrapper.spot import Spot


class Navigation:
    """
    Navigation is used to navigate the robot to a given waypoint specified as[x, y , yaw]
    in robot's current frame of reference.

    Args:
        spot: Spot object
        config: Config object
        record_robot_trajectories: bool indicating whether to record robot's trajectory

    How to use:
        1. Create Navigation object
        2. Call execute() with nav_targets list as input and get robot's trajectory as output

    Example:
        config = construct_config(opts=[])
        spot = Spot("spot_client_name")
        with spot.get_lease(hijack=True):
            spot.power_on()

            nav_targets_list = [target1, target2, ...]
            nav = Navigation(spot, config)
            robot_trajectories = nav.execute(nav_targets_list)

            spot.shutdown()
    """

    def __init__(
        self, spot: Spot, config=None, record_robot_trajectories=False
    ) -> None:

        # (TODO: Move all to base Skill class)
        if not config:
            config = construct_config_for_nav()
        # super.__init__(spot, config)
        self.spot = spot
        self.config = config
        self.verbose = True

        # Record robot's trajectory (i.e. waypoints)
        self.recording_in_progress = False
        self.record_robot_trajectories = record_robot_trajectories

        # Setup
        self.policy = NavPolicy(self.config.WEIGHTS.NAV, device=self.config.DEVICE)
        self.policy.reset()

        self.env = SpotNavEnv(self.config, self.spot)

    def reset_env_and_policy(self, nav_target: Tuple[float, float, float]):
        """
        Resets the env and policy

        Args:
            nav_target: (x,y,theta) where robot needs to navigate to

        Returns:
            observations: observations from the env

        """
        (goal_x, goal_y, goal_heading) = nav_target
        observations = self.env.reset((goal_x, goal_y), goal_heading)
        self.policy.reset()

        return observations

    def execute_nav(
        self, nav_targets_list: List[Tuple[float, float, float]]
    ) -> List[List[Dict]]:
        """
        @INFO: This function is only to be used locally in spot_rl_experiments
        Executes the navigation to the given nav_targets_list and returns the robot's trajectory
        Args:
            nav_targets_list: List of nav_targets (x,y,theta) where robot needs to navigate to

        Returns:
            robot_trajectories: [[Dict]] where each Dict contains timestamp and pose of the robot, inner list contains trajectory for each nav_target and outer list is a collection of each of the nav_target's trajectory
        """

        robot_trajectories = []  # type: List[List[Dict]]
        for nav_target in nav_targets_list:
            observations = self.reset_env_and_policy(nav_target)
            done = False

            # List of Dicts to store trajectory for
            # each of the nav_targets in nav_targets_list
            robot_trajectory = []  # type: List[Dict]
            time.sleep(1)

            self.env.say(f"Navigating to {nav_target}")

            if self.record_robot_trajectories and not self.recording_in_progress:
                self.recording_in_progress = True
                # Set start time for recording before execution of 1st nav waypoint
                start_time = time.time()

            # Execution Loop
            while not done:
                action = self.policy.act(observations)
                observations, _, done, _ = self.env.step(base_action=action)

                # Record trajectories at every step if True
                if self.record_robot_trajectories:
                    robot_trajectory.append(
                        {
                            "timestamp": time.time() - start_time,
                            "pose": [
                                self.env.x,
                                self.env.y,
                                np.rad2deg(self.env.yaw),
                            ],
                        }
                    )
            # Store the trajectory for each nav_target inside the List[robot_trajectory]
            if self.record_robot_trajectories:
                robot_trajectories.append(robot_trajectory)

        # Turn off recording
        self.recording_in_progress = False

        # Return waypoints back
        return robot_trajectories

    def execute(
        self, nav_target: Tuple[float, float, float]
    ) -> Tuple[bool, str]:  # noqa
        """
        Executes the navigation to the given a nav_target and returns the success status and feedback msg(str)

        Args:
            nav_target: Tuple of (x,y,theta) where robot needs to navigate to

        Returns:
            status (bool): Whether robot was able to succesfully execute the skill or not
            message (str): Message indicating description of success / failure reason
        """

        if nav_target is None:
            message = "No nav target specified, skipping nav"
            conditional_print(message=message, verbose=self.verbose)
            return False, message

        (x, y, theta) = nav_target
        conditional_print(
            message=f"Navigating to x, y, theta : {x}, {y}, {theta}",
            verbose=self.verbose,
        )

        result = None
        nav_target_tuple = None
        try:
            nav_target_tuple = (x, y, theta)
            result = self.execute_nav([nav_target_tuple])
        except Exception as e:
            message = f"Error encountered while navigating : {e}"
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
            self.config.SUCCESS_DISTANCE,
            self.config.SUCCESS_ANGLE_DIST,
        )

        # Check for success and return appropriately
        status = False
        message = "Navigation failed to reach the target pose"
        if check_navigation_suc:
            status = True
            message = "Successfully reached the target pose by default"
        conditional_print(message=message, verbose=self.verbose)
        return status, message


class Pick:
    """
    Pick is used to gaze at, and pick given objects.

    Args:
        spot (Spot): Spot object
        config (Config): Config object

    How to use:
        1. Create a GazeController object
        2. Call execute() method with the target object list

    Example:
        config = construct_config_for_gaze(opts=[])
        spot = Spot("spot_client_name")
        with spot.get_lease(hijack=True):
            spot.power_robot()

            gaze_target_list = ["apple", "banana"]
            gaze_controller = GazeController(config, spot)
            gaze_results = gaze_controller.execute(gaze_target_list)

            spot.shutdown(should_dock=True)
    """

    def __init__(self, spot, config, use_mobile_pick=False):
        # (TODO: Move all to base Skill class)
        if not config:
            config = construct_config_for_gaze()
        # super.__init__(spot, config)
        self.spot = spot
        self.config = config
        self.verbose = True

        self._use_mobile_pick = use_mobile_pick
        if use_mobile_pick:
            self.policy = MobileGazePolicy(
                self.config.WEIGHTS.MOBILE_GAZE,
                device=self.config.DEVICE,
                config=self.config,
            )
        else:
            self.policy = GazePolicy(
                self.config.WEIGHTS.GAZE, device=self.config.DEVICE
            )
        self.policy.reset()

        self.env = SpotGazeEnv(self.config, spot, use_mobile_pick)

    def reset_env_and_policy(self, target_obj_name):
        """
        Resets the env and policy

        Args:
            target_obj_name (str): Name of the target object

        Returns:
            observations: observations from the env

        """
        observations = self.env.reset(target_obj_name=target_obj_name)
        self.policy.reset()

        return observations

    def execute_pick(self, target_object_list, take_user_input=False):
        """
        Gaze at the target object list and pick up the objects if specified in the config

        CAUTION: The robot will drop the object after picking it, please use objects that are not fragile

        Args:
            target_object_list (list): List of target objects to gaze at
            take_user_input (bool): Whether to take user input for the success of the gaze

        Returns:
            gaze_success_list (list): List of dictionaries containing the target object name, time taken and success
        """
        gaze_success_list = []
        print(f"Target object list : {target_object_list}")
        for target_object in target_object_list:
            observations = self.reset_env_and_policy(target_obj_name=target_object)
            done = False
            start_time = time.time()
            self.env.say(f"Gaze at target object - {target_object}")

            while not done:
                action = self.policy.act(observations)
                if self._use_mobile_pick:
                    arm_action, base_action = None, None
                    # first 4 are arm actions, then 2 are base actions & last bit is unused
                    arm_action = action[0:4]
                    base_action = action[4:6]

                    observations, _, done, _ = self.env.step(
                        arm_action=arm_action, base_action=base_action
                    )
                else:
                    observations, _, done, _ = self.env.step(arm_action=action)
            self.env.say("Gaze finished")
            # Ask user for feedback about the success of the gaze and update the "success" flag accordingly
            success_status_from_user_feedback = True
            if take_user_input:
                user_prompt = f"Did the robot successfully pick the right object - {target_object}?"
                success_status_from_user_feedback = map_user_input_to_boolean(
                    user_prompt
                )

            gaze_success_list.append(
                {
                    "target_object": target_object,
                    "time_taken": time.time() - start_time,
                    "success": self.env.grasp_attempted
                    and success_status_from_user_feedback,
                }
            )
        return gaze_success_list

    def execute(self, pick_target: str):
        if pick_target is None:
            message = "No pick target specified, skipping pick"
            conditional_print(message=message, verbose=self.verbose)
            return False, message

        conditional_print(message=f"Picking {pick_target}", verbose=self.verbose)

        result = None
        try:
            result = self.execute_pick([pick_target])
        except Exception as e:
            message = f"Error encountered while picking - {e}"
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


class Place:
    """
    Place controller is used to execute place for given place targets

    Args:
        config: Config object
        spot: Spot object
        use_policies (bool): Whether to use policies or use BD API to execute place

    How to use:
        1. Create PlaceController object
        2. Call execute() with place_target_list as input

    Example:
        config = construct_config_for_place(opts=[])
        spot = Spot("PlaceController")
        with spot.get_lease(hijack=True):
            spot.power_robot()

            place_target_list = [target1, target2, ...]
            place_controller = PlaceController(config, spot, use_policies=True)
            place_result = place_controller.execute(place_target_list, is_local=False)

            spot.shutdown(should_dock=True)
    """

    def __init__(self, spot: Spot, config, use_policies=True):
        # (TODO: Move all to base Skill class)
        if not config:
            config = construct_config_for_place()
        # super.__init__(spot, config)
        self.spot = spot
        self.config = config
        self.verbose = True

        self.use_policies = use_policies
        # Setup
        if self.use_policies:
            self.policy = PlacePolicy(config.WEIGHTS.PLACE, device=config.DEVICE)
            self.policy.reset()

        self.env = SpotPlaceEnv(config, spot)

    def reset_env_and_policy(self, place_target, is_local):
        """
        Resets the env and policy

        Args:
            place_target (np.array([x,y,z])): Place target in either global frame or base frame of the robot
            is_local (bool): Whether the place target is in the base frame of the robot

        Returns:
            observations: Initial observations from the env
        """
        observations = self.env.reset(place_target, is_local)
        self.policy.reset()

        return observations

    def execute_place(self, place_target_list, is_local=False):
        """
        Execute place for each place target in place_target_list

        Args:
            place_target_list (list): List of place targets to go and place
            is_local (bool): Whether the place target is in the local frame of the robot

        Returns:
            success_list (list): List of dicts containing the following keys:
                - time_taken (float): Time taken to place the object
                - success (bool): Whether the place was successful
                - place_target (np.array([x,y,z])): Place target in base frame
                - ee_pos (np.array([x,y,z])): End effector position in base frame
        """
        success_list = []
        for place_target in place_target_list:
            start_time = time.time()

            self.env.say(f"Placing at {place_target}")

            if self.use_policies:
                observations = self.reset_env_and_policy(place_target, is_local)
                done = False

                while not done:
                    action = self.policy.act(observations)
                    observations, _, done, _ = self.env.step(arm_action=action)

            else:
                # TODO: Get reset arm position (Is there a better way to do this without using place env?????????????)
                self.env.reset(place_target, is_local)

                # End effector positions in base frame (as needed by the API)
                curr_ee_pos = self.env.get_gripper_position_in_base_frame_spot()
                goal_ee_pos = self.env.get_base_frame_place_target_spot()
                intr_ee_pos = generate_intermediate_point(curr_ee_pos, goal_ee_pos)

                # Get direction vector from current ee position to goal ee position for EE orientation
                dir_rpy_to_intr = get_RPY_from_vector(goal_ee_pos - curr_ee_pos)

                # Go to intermediate point
                self.spot.move_gripper_to_point(
                    intr_ee_pos,
                    dir_rpy_to_intr,
                    self.config.ARM_TRAJECTORY_TIME_IN_SECONDS,
                    timeout_sec=10,
                )

                # Direct the gripper to face downwards
                dir_rpy_to_goal = [0.0, np.pi / 2, 0.0]

                # Go to goal point
                self.spot.move_gripper_to_point(
                    goal_ee_pos,
                    dir_rpy_to_goal,
                    self.config.ARM_TRAJECTORY_TIME_IN_SECONDS,
                    timeout_sec=10,
                )

            # Record the success
            local_place_target_spot = self.env.get_base_frame_place_target_spot()
            local_ee_pose_spot = self.env.get_gripper_position_in_base_frame_spot()
            success_list.append(
                {
                    "time_taken": time.time() - start_time,
                    "success": is_position_within_bounds(
                        local_place_target_spot,
                        local_ee_pose_spot,
                        self.config.SUCC_XY_DIST,
                        self.config.SUCC_Z_DIST,
                        convention="spot",
                    ),
                    "place_target": local_place_target_spot,
                    "ee_pos": local_ee_pose_spot,
                }
            )

            # Open gripper to drop the object
            self.spot.open_gripper()
            # Add sleep as open_gripper() is a non-blocking call
            time.sleep(1)

            # Reset the arm here
            self.env.reset_arm()

        return success_list

    def execute(
        self, place_target: Tuple[float, float, float], is_local=False
    ) -> Tuple[bool, str]:  # noqa

        if place_target is None:
            message = "No place target specified, skipping nav"
            conditional_print(message=message, verbose=self.verbose)
            return False, message

        (x, y, z) = place_target
        conditional_print(
            message=f"Place target object at x, y, z : {x}, {y}, {z}",
            verbose=self.verbose,
        )

        result = None
        place_target_tuple = None
        try:
            place_target_tuple = (x, y, z)
            result = self.execute_place([place_target_tuple], is_local=is_local)
        except Exception as e:
            message = f"Error encountered while placing : {e}"
            conditional_print(message=message, verbose=self.verbose)
            return False, message

        # Check for success and return appropriately
        status = False
        message = "Place failed to reach the target position"
        if is_position_within_bounds(
            result[0].get("ee_pos"),
            result[0].get("place_target"),
            self.config.SUCC_XY_DIST,
            self.config.SUCC_Z_DIST,
            convention="spot",
        ):
            status = True
            message = "Successfully reached the target position"
        conditional_print(message=message, verbose=self.verbose)
        return status, message
