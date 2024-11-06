# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import quaternion
import rospy
from perception_and_utils.utils.generic_utils import (
    conditional_print,
    map_user_input_to_boolean,
)
from spot_rl.envs.gaze_env import SpotGazeEEEnv, SpotGazeEnv, SpotSemanticGazeEnv

# Import Envs
from spot_rl.envs.nav_env import SpotNavEnv
from spot_rl.envs.open_close_drawer_env import SpotOpenCloseDrawerEnv
from spot_rl.envs.place_env import (
    SpotPlaceEnv,
    SpotSemanticPlaceEEEnv,
    SpotSemanticPlaceEnv,
)

# Import policies
from spot_rl.real_policy import (
    GazePolicy,
    MobileGazeEEPolicy,
    MobileGazePolicy,
    NavPolicy,
    OpenCloseDrawerPolicy,
    PlacePolicy,
    SemanticGazePolicy,
    SemanticPlaceEEPolicy,
    SemanticPlacePolicy,
)

# Import utils and helpers
from spot_rl.utils.construct_configs import (
    construct_config_for_gaze,
    construct_config_for_nav,
    construct_config_for_open_close_drawer,
    construct_config_for_place,
    construct_config_for_semantic_place,
)

# Import Utils
from spot_rl.utils.geometry_utils import (
    euclidean,
    generate_intermediate_point,
    get_RPY_from_vector,
    is_pose_within_bounds,
    is_position_within_bounds,
    wrap_angle_deg,
)
from spot_rl.utils.utils import get_skill_name_and_input_from_ros

# Import core classes
from spot_wrapper.spot import Spot, image_response_to_cv2

###
### Atomic Skills is a skill which requires its own policy and environment and does not depend on other atomic or composite skills
###


# Define Interface skill class
class Skill:
    """
    Base Interface class for all skills

    This class is used to define the structure and interface for all skills
    WARNING: Do not create an object of this class directly, use the derived classes instead

    """

    def __init__(self, spot: Spot, config=None) -> None:
        self.spot = spot
        self.config = config
        self.verbose = True

        # Logger to log data from the most recent robot trajectory
        self.skill_result_log = {}  # type: Dict[str, Any]
        self.start_time = None  # type: float
        self.reset_logger()

    def sanity_check(self, goal_dict: Dict[str, Any]):
        """
        Sanity check for the input goal_dict

        Args:
            goal_dict: Dict[str, Any] containing necessary goal information for skill
                                      Required keys: Please check the skill's docstring for details

        Raises:
            KeyError: If the required keys are not found in the goal_dict
        """
        pass

    def reset_skill(self, goal_dict: Dict[str, Any]):
        """
        Performs:
            Sanity checks the input goal_dict,
            Resets the env and policy,
            Updates the logged data at init,
            Sets the start time for recording before starting skill

        Args:
            goal_dict: Dict[str, Any] containing necessary goal information for skill
                                      Required keys: Please check the skill's docstring for details

        Returns:
            observations: Initial observations from the env
        """
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def reset_logger(self):
        """Resets result log dict and start time for new run"""
        # Clear result log from the previous run
        self.skill_result_log = {
            "time_taken": None,
            "success": False,
            "robot_trajectory": [],
        }  # type: Dict[str, Any]

        # Set start time for recording before starting skill (this also resets the previous start time data)
        self.start_time = time.time()  # type: float

    def execute_rl_loop(self, goal_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """
        !!BLOCKING CALL!!

        Executes the action based on output from policy

        WARNING: This method WILL NOT perform any explicit sanity checks on the input provided.
                 It is the responsibility of the caller to ensure that the input is valid
                 and safe to use or perform sanity checks inside reset_skill() method

        Args:
            goal_dict: Dict[str, Any] containing necessary goal information for the appropriate skill

        Returns:
            status (bool): Whether robot was able to succesfully execute the skill or not
            message (str): Message indicating description of success / failure reason
        """

        # Check the current buffer to see which skill we are using right now
        # This does not have any effect if the initial ros buffter is none
        begin_skill_name, begin_skill_input = get_skill_name_and_input_from_ros()

        # Reset policy,env and update logged data at init
        try:
            observations = self.reset_skill(goal_dict)
        except Exception as e:
            raise e
        done = False

        self.ee_point_before_starting_the_skill = self.spot.get_ee_pos_in_body_frame()[
            0
        ]

        current_human_action = self.env.human_activity_current.copy()  # type: ignore

        # Execution Loop
        while not done:
            # The current formate is timestamp, action, object_name, target_receptacle.
            # Read the human action to interrupt the skill execution
            human_action = "None"
            if self.config.READ_HUMAN_ACTION_FROM_ROS_PARAM:
                human_action = rospy.get_param(
                    "/human_action", f"{str(time.time())},None,None,None"
                )
                print(f"Human action :: {human_action}")
            else:
                current_human_action = self.env.human_activity_current.copy()  # type: ignore
                print(f"human_action: {human_action}")
                previous_human_action = getattr(self, "previous_human_activity", None)
                if previous_human_action != current_human_action:
                    human_action = "human_action_detected"
                else:
                    human_action = "None"

            action = self.policy.act(observations)  # type: ignore
            action_dict = self.split_action(action)
            if "should_dock" in goal_dict:
                action_dict["should_dock"] = goal_dict["should_dock"]
            prev_pose = [
                self.env.x,  # type: ignore
                self.env.y,  # type: ignore
            ]
            observations, _, done, info = self.env.step(action_dict=action_dict)  # type: ignore

            if "None" not in human_action:
                done = True

            curr_pose = [
                self.env.x,  # type: ignore
                self.env.y,  # type: ignore
            ]
            # Record trajectories at every step
            self.skill_result_log["robot_trajectory"].append(
                {
                    "timestamp": time.time() - self.start_time,
                    "pose": [
                        self.env.x,  # type: ignore
                        self.env.y,  # type: ignore
                        np.rad2deg(self.env.yaw),  # type: ignore
                    ],
                }
            )
            self.skill_result_log["num_steps"] = info["num_steps"]
            if "distance_travelled" not in self.skill_result_log:
                self.skill_result_log["distance_travelled"] = 0
            self.skill_result_log["distance_travelled"] += euclidean(
                curr_pose, prev_pose
            )

            # Check if we still want to use the same skill
            # We terminate the skill if skill name or input is differnt from the one at the beginning.
            cur_skill_name, cur_skill_input = get_skill_name_and_input_from_ros()
            if (
                cur_skill_name != begin_skill_name
                or cur_skill_input != begin_skill_input
            ):
                done = True
            self.previous_human_activity = current_human_action
        # Update logged data after finishing execution and get feedback (status & msg)
        return self.update_and_check_status(goal_dict)

    def execute(self, goal_dict: Dict[str, Any]) -> Tuple[bool, str]:  # noqa
        """
        Executes the skill for the new_target and returns the success status and feedback msg(str)

        Args:
            goal_dict: Dict[str, Any] containing necessary goal information for skill
                                      Required keys: Please check the skill's docstring for details

        Returns:
            status (bool): Whether robot was able to succesfully execute the skill or not
            message (str): Message indicating description of success / failure reason
        """
        status = False
        message = ""
        try:
            status, message = self.execute_rl_loop(goal_dict=goal_dict)
            print(f"Feedback from skill: {message}")
        except Exception as e:
            message = f"Error encountered in skill execution - {e}"
            conditional_print(message=message, verbose=self.verbose)

        return status, message

    def update_and_check_status(self, goal_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Updates the logged data at the end of the skill execution loop
        Checks the execution results and provides feedback

        Args:
            goal_dict: Dict[str, Any] containing necessary goal information for skill
                                      Required keys: Please check the skill's docstring for details

        Returns:
            status (bool): Whether robot was able to succesfully execute the skill or not
            message (str): Message indicating description of success / failure reason
        """
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def get_most_recent_result_log(self):
        """
        Returns the most recently logged data from the skill execution

        Returns:
            skill_result_log : Dict[str,Any] Result log containing following keys
                - time_taken : float Time taken to run skill
                - success : bool Whether the skill was successful
                - robot_trajectory :List[Dict] List of dict ((timestamp, pose)
                                          where pose is [x, y, theta(deg) in deg])
                                          for most recent run
        """
        return self.skill_result_log

    def split_action(self, action: np.ndarray) -> Dict[str, Any]:
        """
        Split action output from policy into an action dictionary

        Action dictionary should contain:
            - base_action: For base motion
            - arm_action: For arm motion

        Args:
            action (np.ndarray): Action array as outputted by policy

        Returns:
            action_dict (Dict[str, Any]): Dictionary containing the appropriate actions
        """
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )


"""
Process to add a new atomic skill : Please follow the steps described in the file skills/README.md

RULES:
1. All atomic skills should inherit from Skill class
2. All policy based atomic skills should have their own policy and environment and should not use other atomic skill objects for their execution
3. All atomic skills should ONLY process 1 goal at a time
4. All policy based atomic skills should use the execute_rl_loop() method for the execution while loop
"""


class Navigation(Skill):
    """
    Navigation is used to navigate the robot to a given waypoint specified as[x, y , yaw]
    in robot's current frame of reference.

    Expected goal_dict input:
        goal_dict = {
            "nav_target": (x, y, theta), # (Necessary) Tuple of (x,y,theta) where robot needs to navigate to.
        }

    Args:
        spot: Spot object
        config: Config object

    How to use:
        1. Create Navigation object
        2. Call execute(goal_dict) with "nav_target" as tuple in input goal_dict
        3. Call get_most_recent_result_log() to get the robot trajectory from the most recent navigation run

    Example:
        config = construct_config(opts=[])
        spot = Spot("spot_client_name")
        with spot.get_lease(hijack=True):
            spot.power_on()

            nav = Navigation(spot, config)
            trajectories = []
            nav_targets_list = [target1, target2, ...]
            for nav_target in nav_targets_list:
                goal_dict = {"nav_target": (x, y, theta)}
                status, feedback = nav.execute(goal_dict)
                robot_trajectory = nav.get_most_recent_result_log().get("robot_trajectory")
                trajectories.append(robot_trajectory)
            spot.shutdown()
    """

    def __init__(self, spot: Spot, config=None) -> None:
        if not config:
            config = construct_config_for_nav()
        super().__init__(spot, config)

        # Setup
        self.policy = NavPolicy(
            self.config.WEIGHTS.NAV, device=self.config.DEVICE, config=self.config
        )
        self.policy.reset()

        self.env = SpotNavEnv(self.config, self.spot)

    def sanity_check(self, goal_dict: Dict[str, Any]):
        """Refer to class Skill for documentation"""
        nav_target = goal_dict.get(
            "nav_target", None
        )  # type: Tuple[float, float, float]
        if nav_target is None:
            raise KeyError(
                "Error in Navigation.sanity_check(): nav_target key not found in goal_dict"
            )

        conditional_print(message="SanityCheck passed for nav", verbose=self.verbose)

    def reset_skill(self, goal_dict: Dict[str, Any]) -> Any:
        """Refer to class Skill for documentation"""
        try:
            self.sanity_check(goal_dict)
        except Exception as e:
            raise e

        nav_target = goal_dict.get("nav_target")
        (x, y, theta) = nav_target
        conditional_print(
            message=f"Navigating to x, y, theta : {x}, {y}, {theta}",
            verbose=self.verbose,
        )

        # Reset the env and policy
        (goal_x, goal_y, goal_heading) = nav_target
        dynamic_yaw = goal_dict.get("dynamic_yaw")
        dynamic_yaw = False if dynamic_yaw is None else dynamic_yaw
        observations = self.env.reset((goal_x, goal_y), goal_heading, dynamic_yaw)
        self.policy.reset()
        # Logging and Debug
        self.env.say(f"Navigating to {goal_dict['nav_target']}")

        # Reset logged data at init
        self.reset_logger()

        return observations

    def update_and_check_status(self, goal_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """Refer to class Skill for documentation"""
        self.env.say("Navigation Skill finished .. checking status")

        nav_target = goal_dict["nav_target"]  # safe to access as sanity check passed
        # Make the angle from rad to deg
        _nav_target_pose_deg = (
            nav_target[0],
            nav_target[1],
            np.rad2deg(nav_target[2]),
        )
        current_pose = self.skill_result_log.get("robot_trajectory")[-1].get("pose")
        check_navigation_success = is_pose_within_bounds(
            current_pose,
            _nav_target_pose_deg,
            (
                self.config.SUCCESS_DISTANCE_FOR_DYNAMIC_YAW_NAV
                if self.env._enable_dynamic_yaw
                else self.config.SUCCESS_DISTANCE
            ),
            (
                self.config.SUCCESS_ANGLE_DIST_FOR_DYNAMIC_YAW_NAV
                if self.env._enable_dynamic_yaw
                else self.config.SUCCESS_ANGLE_DIST
            ),
        )

        # Update result log

        self.skill_result_log["distance_to_goal"] = {
            "linear": euclidean(current_pose[:2], _nav_target_pose_deg[:2]),
            "angular": abs(
                wrap_angle_deg(current_pose[2])
                - wrap_angle_deg(_nav_target_pose_deg[2])
            ),
        }
        self.skill_result_log["time_taken"] = time.time() - self.start_time
        self.skill_result_log["success"] = check_navigation_success

        # Check for success and return appropriately
        status = False
        message = "Navigation failed to reach the target pose"
        if check_navigation_success:
            status = True
            message = "Successfully reached the target pose by default"
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    def split_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Refer to class Skill for documentation"""
        # For navigation, all of input actions are base actions
        action_dict = {
            "arm_action": None,
            "base_action": action,
        }

        return action_dict


class Pick(Skill):
    """
    Pick is used to gaze at, and pick given objects.

    CAUTION: The robot will drop the object after picking it, please use objects that are not fragile

    Expected goal_dict input:
        goal_dict = {
            "target_object": "apple", # (Necessary) Name of the target object to pick
            "take_user_input": False, # (Optional) Whether to take user input for verifying success of the gaze
        }

    Args:
        spot (Spot): Spot object
        config (Config): Config object

    How to use:
        1. Create a Pick object
        2. Call execute(goal_dict) method with "target"object" as a str in input goal_dict
        3. Call get_most_recent_result_log() to get the result from the most recent pick operation

    Example:
        config = construct_config_for_gaze(opts=[])
        spot = Spot("spot_client_name")
        with spot.get_lease(hijack=True):
            spot.power_robot()

            gaze_target_list = ["apple", "banana"]
            results = []
            pick = Pick(spot, config)
            for target_object in gaze_target_list:
                goal_dict = {"target_object": target_object}
                status, feedback = pick.execute(goal_dict=goal_dict)
                results.append(pick.get_most_recent_result_log())
            spot.shutdown(should_dock=True)
    """

    def __init__(self, spot, config=None, use_mobile_pick=False) -> None:
        if not config:
            config = construct_config_for_gaze()
        super().__init__(spot, config)

        # Setup
        self._use_mobile_pick = use_mobile_pick
        if use_mobile_pick:
            self.policy = MobileGazePolicy(
                self.config.WEIGHTS.MOBILE_GAZE,
                device=self.config.DEVICE,
                config=self.config,
            )
        else:
            self.policy = GazePolicy(
                self.config.WEIGHTS.GAZE,
                device=self.config.DEVICE,
                config=self.config,
            )
        self.policy.reset()
        self.enable_pose_estimation: bool = False
        self.enable_pose_correction: bool = False
        self.enable_force_control: bool = False

        self.env = SpotGazeEnv(self.config, spot, use_mobile_pick)

    def set_pose_estimation_flags(
        self,
        enable_pose_estimation: bool = False,
        enable_pose_correction: bool = False,
    ) -> None:
        self.enable_pose_estimation = enable_pose_estimation
        self.enable_pose_correction = enable_pose_correction

    def set_force_control(self, enable_force_control: bool = False):
        self.enable_force_control = enable_force_control

    def set_grasp_type(self, grasp_mode: str = "any"):
        self.grasp_mode = grasp_mode

    def sanity_check(self, goal_dict: Dict[str, Any]):
        """Refer to class Skill for documentation"""
        target_obj_name = goal_dict.get("target_object", None)  # type: str
        if target_obj_name is None:
            raise KeyError(
                "Error in Pick.sanity_check(): target_object key not found in goal_dict"
            )

        conditional_print(message="SanityCheck passed for pick", verbose=self.verbose)

    def reset_skill(self, goal_dict: Dict[str, Any]) -> Any:
        """Refer to class Skill for documentation"""
        try:
            self.sanity_check(goal_dict)
        except Exception as e:
            raise e

        target_obj_name = goal_dict.get("target_object", None)
        take_user_input = goal_dict.get("take_user_input", False)  # type: bool
        conditional_print(
            message=f"Gaze at object : {target_obj_name}  - {'WILL' if take_user_input else 'WILL NOT'} take user input at the end for verification of pick",
            verbose=self.verbose,
        )

        # Reset the env and policy
        observations = self.env.reset(target_obj_name=target_obj_name)
        self.policy.reset()

        # Logging and Debug
        self.env.say(f"Gaze at target object - {target_obj_name}")
        print(
            "The robot will drop the object after picking it, please use objects that are not fragile"
        )

        # Reset logged data at init
        self.reset_logger()

        return observations

    def update_and_check_status(self, goal_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """Refer to class Skill for documentation"""
        self.env.say("Pick Skill finished.. checking status")

        target_object = goal_dict.get("target_object")
        take_user_input = goal_dict.get("take_user_input", False)

        # Ask user for feedback about the success of the gaze and update the "success" flag accordingly
        success_status_from_user_feedback = True
        if take_user_input:
            user_prompt = (
                f"Did the robot successfully pick the right object - {target_object}?"
            )
            success_status_from_user_feedback = map_user_input_to_boolean(user_prompt)

        # Get the images from the env and check if the image is being block by the object
        time.sleep(0.5)
        # Get the hand depth images to test if the gripper is holding something
        hand_image_responses = self.spot.get_hand_image()
        imgs_list = [image_response_to_cv2(r) for r in hand_image_responses]
        hand_depth = imgs_list[1] / 1000.0  # from mm to meter

        # Check the value -- the gripper close state is about 0.48
        block_ratio = (
            np.sum(hand_depth < self.config.BLOCK_VALUE_THRESHOLD) / hand_depth.size
        )
        is_object_block_camera = block_ratio >= self.config.BLOCK_PERCENTAGE_THRESHOLD

        # Check the openness of the gripper
        # This value is between 0 (close) and 100 (open)
        _gripper_open_percentage = (
            self.spot.robot_state_client.get_robot_state().manipulator_state.gripper_open_percentage
        )
        is_gripper_open_slightly = (
            _gripper_open_percentage
            > self.config.GRIPPER_OPEN_PERCENTAGE_THRESHOLD_FOR_GRASPING
        )

        print(
            f"is_object_block_camera: {is_object_block_camera} with ratio {block_ratio} and threshold {self.config.BLOCK_PERCENTAGE_THRESHOLD}"
        )
        print(
            f"is_gripper_open_slightly: {is_gripper_open_slightly} with gripper open {_gripper_open_percentage}% and threshold {self.config.GRIPPER_OPEN_PERCENTAGE_THRESHOLD_FOR_GRASPING}"
        )

        check_pick_success = (
            self.env.grasp_attempted
            and success_status_from_user_feedback
            and is_object_block_camera
            and is_gripper_open_slightly
        )

        # Update result log
        self.skill_result_log["time_taken"] = time.time() - self.start_time
        self.skill_result_log["success"] = check_pick_success

        # Check for success and return appropriately
        status = False
        message = "Pick failed to pick the target object"
        if check_pick_success:
            status = True
            message = "Successfully picked the target object"
        else:
            # If pick fails, we open the gripper again to ensure the gripper state is correct
            self.spot.open_gripper()
            time.sleep(0.5)

        conditional_print(message=message, verbose=self.verbose)
        return status, message

    def split_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Refer to class Skill for documentation"""
        # Mobile pick uses both base & arm but static pick only uses arm
        action_dict = None

        if self._use_mobile_pick:
            # first 4 are arm actions, then 2 are base actions & last bit is unused
            action_dict = {
                "arm_action": action[0:4],
                "base_action": action[4:6],
                "enable_pose_estimation": self.enable_pose_estimation,
                "enable_pose_correction": self.enable_pose_correction,
                "enable_force_control": self.enable_force_control,
                "grasp_mode": self.grasp_mode,
            }
        else:
            action_dict = {
                "arm_action": action,
                "base_action": None,
                "enable_pose_estimation": self.enable_pose_estimation,
                "enable_pose_correction": self.enable_pose_correction,
                "enable_force_control": self.enable_force_control,
                "grasp_mode": self.grasp_mode,
            }

        return action_dict


class SemanticPick(Pick):
    """
    Semantic Pick is used to gaze at, and pick given objects.

    CAUTION: The robot will drop the object after picking it, please use objects that are not fragile

    Expected goal_dict input:
        goal_dict = {
            "target_object": "apple", # (Necessary) Name of the target object to pick
            "take_user_input": False, # (Optional) Whether to take user input for verifying success of the gaze
        }

    Args:
        spot (Spot): Spot object
        config (Config): Config object

    How to use:
        1. Create a Pick object
        2. Call execute(goal_dict) method with "target"object" as a str in input goal_dict
        3. Call get_most_recent_result_log() to get the result from the most recent pick operation

    Example:
        config = construct_config_for_gaze(opts=[])
        spot = Spot("spot_client_name")
        with spot.get_lease(hijack=True):
            spot.power_robot()

            gaze_target_list = ["apple", "banana"]
            results = []
            pick = Pick(spot, config)
            for target_object in gaze_target_list:
                goal_dict = {"target_object": target_object}
                status, feedback = pick.execute(goal_dict=goal_dict)
                results.append(pick.get_most_recent_result_log())
            spot.shutdown(should_dock=True)
    """

    def __init__(self, spot, config=None) -> None:
        if not config:
            config = construct_config_for_gaze()
        super().__init__(spot, config)

        self.policy = SemanticGazePolicy(
            self.config.WEIGHTS.SEMANTIC_GAZE,
            device=self.config.DEVICE,
            config=self.config,
        )

        self.policy.reset()

        self.env = SpotSemanticGazeEnv(self.config, spot)

    def reset_skill(self, goal_dict: Dict[str, Any]) -> Any:
        """Refer to class Skill for documentation"""
        try:
            self.sanity_check(goal_dict)
        except Exception as e:
            raise e

        target_obj_name = goal_dict.get("target_object", None)
        take_user_input = goal_dict.get("take_user_input", False)  # type: bool
        grasping_type = goal_dict.get("grasping_type", "topdown")
        conditional_print(
            message=f"Gaze at object : {target_obj_name}  - {'WILL' if take_user_input else 'WILL NOT'} take user input at the end for verification of pick",
            verbose=self.verbose,
        )

        # Reset the env and policy
        observations = self.env.reset(
            target_obj_name=target_obj_name, grasping_type=grasping_type
        )
        self.policy.reset()

        # Logging and Debug
        self.env.say(
            f"Gaze at target object - {target_obj_name} with {grasping_type} grasping"
        )
        print(
            "The robot will drop the object after picking it, please use objects that are not fragile"
        )

        # Reset logged data at init
        self.reset_logger()

        return observations

    def split_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Refer to class Skill for documentation"""
        # Mobile pick uses both base & arm but static pick only uses arm
        action_dict = None

        # first 4 are arm actions, then 2 are base actions & last bit is unused
        action_dict = {
            "arm_action": action[0:4],
            "base_action": action[4:6],
        }
        return action_dict


class MobilePickEE(Pick):
    """
    Semantic place ee controller is used to execute place for given place targets
    """

    def __init__(self, spot: Spot, config, use_mobile_pick=True):
        if not config:
            config = construct_config_for_gaze()
        super().__init__(spot, config, use_mobile_pick=True)

        self.policy = MobileGazeEEPolicy(
            self.config.WEIGHTS.MOBILE_GAZE,
            device=self.config.DEVICE,
            config=self.config,
        )
        self.policy.reset()

        self.env = SpotGazeEEEnv(config, spot, use_mobile_pick)

    def split_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Refer to class Skill for documentation"""
        action_dict = {
            "arm_ee_action": action[:6],
            "base_action": action[7:9],
            "enable_pose_estimation": self.enable_pose_estimation,
            "enable_pose_correction": self.enable_pose_correction,
            "enable_force_control": self.enable_force_control,
            "grasp_mode": self.grasp_mode,
        }

        return action_dict


class Place(Skill):
    """
    Place controller is used to execute place for given place targets

    CAUTION: The robot will DROP the object from the place_target location, please use objects that are not fragile!

    Expected goal_dict input:
        goal_dict = {
            "place_target": (x, y, z), # (Necessary) Tuple of (x,y,z) where robot needs to place at
            "is_local": False,         # (Optional) Whether the place target is in the local frame of the robot
        }

    Args:
        config: Config object
        spot: Spot object
        use_policies (bool): Whether to use policies or use BD API to execute place

    How to use:
        1. Create Place object
        2. Call execute(goal_dict) with "place_target" as tuple and "is_local" as bool in input goal_dict
        3. Call get_most_recent_result_log() to get the result from the most recent place operation

    Example:
        config = construct_config_for_place(opts=[])
        spot = Spot("spot_client_name")
        with spot.get_lease(hijack=True):
            spot.power_robot()

            place_target_list = [target1, target2, ...]
            results = []
            place = Place(spot, config, use_policies=True)
            for place_target in place_target_list:
                goal_dict = {"place_target": place_target, "is_local": True/False}
                status, feedback = place.execute(goal_dict=goal_dict)
                results.append(place.get_most_recent_result_log())
            spot.shutdown(should_dock=True)
    """

    def __init__(self, spot: Spot, config=None, use_policies=True) -> None:
        if not config:
            config = construct_config_for_place()
        super().__init__(spot, config)

        # Setup
        self.policy = None
        self.use_policies = use_policies
        if self.use_policies:
            self.policy = PlacePolicy(
                config.WEIGHTS.PLACE, device=config.DEVICE, config=config
            )
            self.policy.reset()

        self.env = SpotPlaceEnv(config, spot)

    def sanity_check(self, goal_dict: Dict[str, Any]):
        """Refer to class Skill for documentation"""
        place_target = goal_dict.get(
            "place_target", None
        )  # type: Tuple[float, float, float]
        if place_target is None:
            raise KeyError(
                "Error in Place.sanity_check(): place_target key not found in goal_dict"
            )

        conditional_print(message="SanityCheck passed for place", verbose=self.verbose)

    def reset_skill(self, goal_dict: Dict[str, Any]) -> Any:
        """Refer to class Skill for documentation"""
        try:
            self.sanity_check(goal_dict)
        except Exception as e:
            raise e

        place_target = goal_dict.get("place_target")
        is_local = goal_dict.get("is_local", False)
        ee_orientation_at_grasping = goal_dict.get("ee_orientation_at_grasping", None)

        (x, y, z) = place_target
        conditional_print(
            message=f"Place target object at x, y, z : {x}, {y}, {z} in {'spot' if is_local else 'spots world'} frame.",
            verbose=self.verbose,
        )

        # Reset the env and policy
        observations = self.env.reset(
            place_target,
            is_local,
            ee_orientation_at_grasping=ee_orientation_at_grasping,
        )
        if self.policy is not None:
            self.policy.reset()

        # Logging and Debug
        self.env.say(f"Placing at {place_target}")
        print(
            "CAUTION: The robot will DROP the object from the place_target location, please use objects that are not fragile!"
        )

        # Reset logged data at init
        self.reset_logger()

        return observations

    def execute(self, goal_dict: Dict[str, Any]) -> Tuple[bool, str]:  # noqa
        """
        Place the object in hand list at the given place target
        Refer to class Skill for documentation
        CAUTION: The robot will DROP the object from the place_target location, please use objects that are not fragile!
        """
        status = False
        message = ""
        try:
            if self.use_policies:
                status, message = self.execute_rl_loop(goal_dict=goal_dict)
            else:
                status, message = self.move_arm_to_goal_via_intermediate_point(
                    goal_dict=goal_dict
                )
            print(f"Feedback from place: {message}")
        except Exception as e:
            message = f"Error encountered while placing : {e}"
            conditional_print(message=message, verbose=self.verbose)

        return status, message

    def move_arm_to_goal_via_intermediate_point(
        self, goal_dict: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Does not use policies to execute place operation

        Move arm to place_target via intermediate point using BD APIs.
        This function is used when we set use_policies to False

        It will also clear the result data from previous place operation and updates
        the result with data from the current place operation once it finishes

        Args:
            goal_dict: Dict[str, Any] containing necessary goal information for placing
                                      Required key: "place_target" (x,y,z) where robot needs to place at
                                                    "is_local" (bool) Whether the place target is in the local frame of the robot

        Returns:
            status (bool): Whether robot was able to succesfully execute the skill or not
            message (str): Message indicating description of success / failure reason
        """
        # Reset the arm (via env.reset) and update logged data at init
        try:
            self.reset_skill(goal_dict)
        except Exception as e:
            raise e

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

        # Update logged data after finishing execution and get feedback (status & msg)
        return self.update_and_check_status(goal_dict)

    def update_and_check_status(self, goal_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """Refer to class Skill for documentation"""
        self.env.say("Place Skill finished.. checking status")

        # Record the success
        local_place_target_spot = self.env.get_base_frame_place_target_spot()
        local_ee_pose_spot = self.env.get_gripper_position_in_base_frame_spot()
        check_place_success = is_position_within_bounds(
            local_place_target_spot,
            local_ee_pose_spot,
            self.config.SUCC_XY_DIST,
            self.config.SUCC_Z_DIST,
            convention="spot",
        )

        # Update result log
        self.skill_result_log["time_taken"] = time.time() - self.start_time
        self.skill_result_log["success"] = check_place_success

        # Open gripper to drop the object
        self.spot.open_gripper()
        # Add sleep as open_gripper() is a non-blocking call
        time.sleep(1)

        # Reset the arm here
        self.env.reset_arm()

        # Check for success and return appropriately
        status = False
        message = "Place failed to reach the target position"
        if check_place_success:
            status = True
            message = "Successfully reached the target position"
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    def split_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Refer to class Skill for documentation"""
        # For place, all of input actions are arm actions
        action_dict = {
            "arm_action": action,
            "base_action": None,
        }

        return action_dict


class SemanticPlace(Place):
    """
    Semantic place controller is used to execute place for given place targets
    """

    def __init__(self, spot: Spot, config):
        if not config:
            config = construct_config_for_semantic_place()
        super().__init__(spot, config)

        self.policy = SemanticPlacePolicy(
            config.WEIGHTS.SEMANTIC_PLACE, device=config.DEVICE, config=config
        )
        self.policy.reset()

        self.env = SpotSemanticPlaceEnv(config, spot)

    def execute_rl_loop(self, goal_dict: Dict[str, Any]) -> Tuple[bool, str]:
        # Set the robot inital pose
        self.env.initial_pose = self.spot.get_ee_pos_in_body_frame()[-1]
        return super().execute_rl_loop(goal_dict)

    def split_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Refer to class Skill for documentation"""
        # For semantic place, TODO: add
        action_dict = {
            "arm_action": action[:5],
            "base_action": None,
            "grip_action": action[5],
        }

        return action_dict

    def update_and_check_status(self, goal_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """Refer to class Skill for documentation"""
        self.env.say("Place Skill finished.. checking status")

        # Record the success
        check_place_success = self.env.get_success({})

        # Update result log
        self.skill_result_log["time_taken"] = time.time() - self.start_time
        self.skill_result_log["success"] = check_place_success

        # Open gripper to drop the object
        self.spot.open_gripper()
        # Add sleep as open_gripper() is a non-blocking call
        time.sleep(1)

        # Reset the arm here
        self.env.reset_arm()

        # Check for success and return appropriately
        status = False
        message = "Place failed to reach the target position"
        if check_place_success:
            status = True
            message = "Successfully reached the target position"
        conditional_print(message=message, verbose=self.verbose)
        return status, message


class SemanticPlaceEE(SemanticPlace):
    """
    Semantic place ee controller is used to execute place for given place targets
    """

    def __init__(self, spot: Spot, config, use_semantic_place=False):
        if not config:
            config = construct_config_for_semantic_place()
        super().__init__(spot, config)

        self.policy = SemanticPlaceEEPolicy(
            config.WEIGHTS.SEMANTIC_PLACE_EE, device=config.DEVICE, config=config
        )
        self.policy.reset()

        self.env = SpotSemanticPlaceEEEnv(config, spot, use_semantic_place)

    def split_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Refer to class Skill for documentation"""
        action_dict = {
            "arm_ee_action": action[:6],
            "base_action": action[7:9],
            "grip_action": action[6],
        }

        return action_dict


class OpenCloseDrawer(Skill):
    """
    Open close drawer controller is used to execute open/close drawers
    """

    def __init__(self, spot, config=None) -> None:
        if not config:
            config = construct_config_for_open_close_drawer()
        super().__init__(spot, config)

        # Setup
        self.policy = OpenCloseDrawerPolicy(
            self.config.WEIGHTS.OPEN_CLOSE_DRAWER,
            device=self.config.DEVICE,
            config=self.config,
        )
        self.policy.reset()

        self.env = SpotOpenCloseDrawerEnv(self.config, spot)

    def reset_skill(self, goal_dict: Dict[str, Any]) -> Any:
        """Refer to class Skill for documentation"""
        # Reset the env and policy
        observations = self.env.reset(goal_dict)
        self.policy.reset()

        # Reset logged data at init
        self.reset_logger()

        return observations

    def update_and_check_status(self, goal_dict: Dict[str, Any]) -> Tuple[bool, str]:
        # Check for success and return appropriately
        status = False
        message = "Open/close failed to open/close the drawer"
        check_open_close_success = self.env.get_success()
        if check_open_close_success:
            status = True
            message = "Successfully opened/closed the drawer"
        conditional_print(message=message, verbose=self.verbose)
        return status, message

    def split_action(self, action: np.ndarray) -> Dict[str, Any]:
        # Assign the action into action dict
        action_dict = {
            "arm_action": action[0:4],
            "base_action": action[5:7],  # None
            "close_gripper": action[4],
        }
        return action_dict
