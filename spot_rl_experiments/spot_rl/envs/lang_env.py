# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import subprocess
import time
from collections import Counter
from typing import Any, Dict

import magnum as mn
import numpy as np
import rospy
from hydra import compose, initialize
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.envs.gaze_env import SpotGazeEnv
from spot_rl.llm.src.rearrange_llm import RearrangeEasyChain
from spot_rl.models.sentence_similarity import SentenceSimilarity
from spot_rl.real_policy import GazePolicy, MixerPolicy, NavPolicy, PlacePolicy
from spot_rl.utils.geometry_utils import is_position_within_bounds
from spot_rl.utils.remote_spot import RemoteSpot
from spot_rl.utils.utils import (
    closest_clutter,
    construct_config,
    get_clutter_amounts,
    get_default_parser,
    get_waypoint_yaml,
    nav_target_from_waypoint,
    object_id_to_nav_waypoint,
    place_target_from_waypoint,
)
from spot_rl.utils.whisper_translator import WhisperTranslator
from spot_wrapper.spot import Spot

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 549))


def main(spot, use_mixer, config, out_path=None):

    if use_mixer:
        policy = MixerPolicy(
            config.WEIGHTS.MIXER,
            config.WEIGHTS.NAV,
            config.WEIGHTS.GAZE,
            config.WEIGHTS.PLACE,
            device=config.DEVICE,
            config=config,
        )
        env_class = SpotMobileManipulationBaseEnv
    else:
        policy = SequentialExperts(
            config.WEIGHTS.NAV,
            config.WEIGHTS.GAZE,
            config.WEIGHTS.PLACE,
            device=config.DEVICE,
            config=config,
        )
        env_class = SpotMobileManipulationSeqEnv

    env = env_class(config, spot)

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
    while True:
        audio_transcription_success = False
        while not audio_transcription_success:
            try:
                input("Are you ready?")
                audio_to_text.record()
                instruction = audio_to_text.translate()
                print("Transcribed instructions : ", instruction)

                # Use LLM to convert user input to an instructions set
                # Eg: nav_1, pick, nav_2 = 'bowl_counter', "container", 'coffee_counter'
                nav_1, pick, nav_2, _ = llm.parse_instructions(instruction)
                print("PARSED", nav_1, pick, nav_2)

                # Find closest nav_targets to the ones robot knows locations of
                nav_1 = sentence_similarity.get_most_similar_in_list(
                    nav_1, list(waypoints_yaml_dict["nav_targets"].keys())
                )
                nav_2 = sentence_similarity.get_most_similar_in_list(
                    nav_2, list(waypoints_yaml_dict["nav_targets"].keys())
                )
                print("MOST SIMILAR: ", nav_1, pick, nav_2)
                audio_transcription_success = True
            except Exception as e:
                print(f"Exception encountered in Speech to text : {e} \n\n Retrying...")

        # Used for Owlvit
        rospy.set_param("object_target", pick)

        # Used for Visualizations
        rospy.set_param("viz_pick", nav_1)
        rospy.set_param("viz_object", pick)
        rospy.set_param("viz_place", nav_2)

        # Ensure the parameters are correct
        rospy.set_param("is_gripper_blocked", 0)
        rospy.set_param("is_whiten_black", True)

        spot.power_robot()
        # time.sleep(1)
        out_data = []

        waypoint = nav_target_from_waypoint(nav_1, waypoints_yaml=waypoints_yaml_dict)
        observations = env.reset(waypoint=waypoint)

        policy.reset()
        done = False
        if use_mixer:
            expert = None
        else:
            expert = Tasks.NAV
        env.stopwatch.reset()
        while not done:
            out_data.append((time.time(), env.x, env.y, env.yaw))
            base_action, arm_action = policy.act(observations, expert=expert)
            nav_silence_only = True
            env.stopwatch.record("policy_inference")
            action_dict = {
                "base_action": base_action,
                "arm_action": arm_action,
            }  # type: Dict[str, Any]
            observations, _, done, info = env.step(
                action_dict=action_dict,
                nav_silence_only=nav_silence_only,
            )

            if use_mixer and info.get("grasp_success", False):
                policy.policy.prev_nav_masks *= 0

            if not use_mixer:
                expert = info["correct_skill"]
            print("Expert:", expert)

            # We reuse nav, so we have to reset it before we use it again.
            if not use_mixer and expert != Tasks.NAV:
                policy.nav_policy.reset()

            env.stopwatch.print_stats(latest=True)

        # Go to the dock
        # TODO: This needs to be cleaned up
        env.say(f"Finished object rearrangement. RETURN_TO_BASE - {return_to_base}.")
        if return_to_base:
            waypoint = nav_target_from_waypoint(
                "dock", waypoints_yaml=waypoints_yaml_dict
            )
            observations = env.reset(waypoint=waypoint)
            expert = Tasks.NAV

            while True:
                base_action, arm_action = policy.act(observations, expert=expert)
                nav_silence_only = True
                env.stopwatch.record("policy_inference")
                action_dict = {
                    "base_action": base_action,
                    "arm_action": arm_action,
                }  # type: Dict[str, Any]
                observations, _, done, info = env.step(
                    action_dict=action_dict,
                    nav_silence_only=nav_silence_only,
                )
                try:
                    spot.dock(dock_id=DOCK_ID)
                    break
                except Exception:
                    print("Dock not found... trying again")
                    time.sleep(0.1)
        else:
            env.say(
                "Since RETURN_TO_BASE was set to false in config.yaml, will sit down."
            )
            time.sleep(2)
            spot.sit()

        print("Done!")

        out_data.append((time.time(), env.x, env.y, env.yaw))

        if out_path is not None:
            data = (
                "\n".join(
                    [",".join([str(i) for i in t_x_y_yaw]) for t_x_y_yaw in out_data]
                )
                + "\n"
            )
            with open(out_path, "w") as f:
                f.write(data)


class Tasks:
    r"""Enumeration of types of tasks."""

    NAV = "nav"
    GAZE = "gaze"
    PLACE = "place"


class SequentialExperts:
    def __init__(
        self, nav_weights, gaze_weights, place_weights, device="cuda", config=None
    ):
        print("Loading nav_policy...")
        self.nav_policy = NavPolicy(nav_weights, device, config)
        print("Loading gaze_policy...")
        self.gaze_policy = GazePolicy(gaze_weights, device, config)
        print("Loading place_policy...")
        self.place_policy = PlacePolicy(place_weights, device, config)
        print("Done loading all policies!")

    def reset(self):
        self.nav_policy.reset()
        self.gaze_policy.reset()
        self.place_policy.reset()

    def act(self, observations, expert):
        base_action, arm_action = None, None
        if expert == Tasks.NAV:
            base_action = self.nav_policy.act(observations)
        elif expert == Tasks.GAZE:
            arm_action = self.gaze_policy.act(observations)
        elif expert == Tasks.PLACE:
            arm_action = self.place_policy.act(observations)

        return base_action, arm_action


class SpotMobileManipulationBaseEnv(SpotGazeEnv):
    node_name = "SpotMobileManipulationBaseEnv"

    def __init__(self, config, spot: Spot):
        super().__init__(config, spot)

        # Nav
        self.goal_xy = None
        self.goal_heading = None
        self.succ_distance = config.SUCCESS_DISTANCE
        self.succ_angle = np.deg2rad(config.SUCCESS_ANGLE_DIST)
        self.gaze_nav_target = None
        self.place_nav_target = None
        self.rho = float("inf")
        self.heading_err = float("inf")

        # Gaze
        self.locked_on_object_count = 0
        self.target_obj_name = None

        # Place
        self.place_target = None
        self.ee_gripper_offset = mn.Vector3(config.EE_GRIPPER_OFFSET)
        self.place_target_is_local = False

        # General
        self.max_episode_steps = 1000
        self.navigating_to_place = False

    def reset(self, waypoint=None, *args, **kwargs):
        # Move arm to initial configuration (w/ gripper open)
        self.spot.set_arm_joint_positions(
            positions=np.deg2rad(self.config.GAZE_ARM_JOINT_ANGLES), travel_time=0.75
        )
        # Wait for arm to arrive to position
        # import pdb; pdb.set_trace()
        time.sleep(0.75)
        print("open gripper called in SpotMobileManipulationBaseEnv")
        self.spot.open_gripper()

        # Nav
        if waypoint is None:
            self.goal_xy = None
            self.goal_heading = None
        else:
            self.goal_xy, self.goal_heading = (waypoint[:2], waypoint[2])

        # Place
        self.place_target = mn.Vector3(-1.0, -1.0, -1.0)

        # General
        self.navigating_to_place = False

        return SpotBaseEnv.reset(self)

    def step(self, action_dict: Dict[str, Any], *args, **kwargs):
        # import pdb; pdb.set_trace()
        gripper_pos_in_base_frame = self.get_gripper_position_in_base_frame_hab()
        place_target_in_base_frame = self.get_base_frame_place_target_hab()
        place = is_position_within_bounds(
            gripper_pos_in_base_frame,
            place_target_in_base_frame,
            self.config.SUCC_XY_DIST,
            self.config.SUCC_Z_DIST,
            convention="habitat",
        )
        if place:
            print("place is true")

        if self.grasp_attempted:
            grasp = False
        else:
            grasp = self.should_grasp()

        if self.grasp_attempted:
            self.max_joint_movement_scale = self.config.MAX_JOINT_MOVEMENT_2
        else:
            self.max_joint_movement_scale = self.config.MAX_JOINT_MOVEMENT

        # Update the  action_dict with grasp and place flags
        action_dict["grasp"] = grasp
        action_dict["place"] = place

        # Slow the base down if we are close to the nav target for grasp to limit blur
        if (
            not self.grasp_attempted
            and self.rho < 0.5
            and abs(self.heading_err) < np.rad2deg(45)
        ):
            self.slowdown_base = 0.5  # Hz
            print("!!!!!!Slow mode!!!!!!")
        else:
            self.slowdown_base = -1
        disable_oa = False if self.rho > 0.3 and self.config.USE_OA_FOR_NAV else None
        observations, reward, done, info = SpotBaseEnv.step(
            self,
            action_dict=action_dict,
            disable_oa=disable_oa,
            *args,
            **kwargs,
        )
        if done:
            print("done is true")

        if self.grasp_attempted and not self.navigating_to_place:
            # Get the waypoints from waypoints.yaml
            waypoints_yaml_dict = get_waypoint_yaml()

            # Determine where to go based on what object we've just grasped
            waypoint_name = rospy.get_param("/viz_place")
            waypoint = nav_target_from_waypoint(waypoint_name, waypoints_yaml_dict)

            self.say("Navigating to " + waypoint_name)

            self.place_target = place_target_from_waypoint(
                waypoint_name, waypoints_yaml_dict
            )
            self.goal_xy, self.goal_heading = (waypoint[:2], waypoint[2])
            self.navigating_to_place = True
            info["grasp_success"] = True

        return observations, reward, done, info

    def get_observations(self):
        observations = self.get_nav_observation(self.goal_xy, self.goal_heading)
        rho = observations["target_point_goal_gps_and_compass_sensor"][0]
        self.rho = rho
        goal_heading = observations["goal_heading"][0]
        self.heading_err = goal_heading
        self.use_mrcnn = True
        observations.update(super().get_observations())
        observations["obj_start_sensor"] = self.get_place_sensor()

        return observations

    def get_success(self, observations):
        return self.place_attempted


class SpotMobileManipulationSeqEnv(SpotMobileManipulationBaseEnv):
    node_name = "SpotMobileManipulationSeqEnv"

    def __init__(self, config, spot: Spot):
        super().__init__(config, spot)
        self.current_task = Tasks.NAV
        self.timeout_start = float("inf")

    def reset(self, *args, **kwargs):
        observations = super().reset(*args, **kwargs)
        self.current_task = Tasks.NAV
        self.target_obj_name = 0
        self.timeout_start = float("inf")

        return observations

    def step(self, *args, **kwargs):
        pre_step_navigating_to_place = self.navigating_to_place
        observations, reward, done, info = super().step(*args, **kwargs)

        if self.current_task != Tasks.GAZE:
            # Disable target searching if we are not gazing
            self.last_seen_objs = []

        if self.current_task == Tasks.NAV and self.get_nav_success(
            observations, self.succ_distance, self.succ_angle
        ):
            if not self.grasp_attempted:
                self.current_task = Tasks.GAZE
                self.timeout_start = time.time()
                self.target_obj_name = None
            else:
                self.current_task = Tasks.PLACE
                self.say("Starting place")
                self.timeout_start = time.time()

        if self.current_task == Tasks.PLACE and time.time() > self.timeout_start + 10:
            # call place after 10s of trying
            print("Place failed to reach target")
            self.spot.rotate_gripper_with_delta(wrist_roll=1.57)
            spot.open_gripper()
            time.sleep(0.75)
            done = True

        if not pre_step_navigating_to_place and self.navigating_to_place:
            # This means that the Gaze task has just ended
            self.current_task = Tasks.NAV

        info["correct_skill"] = self.current_task

        self.use_mrcnn = self.current_task == Tasks.GAZE

        #

        return observations, reward, done, info


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("-m", "--use-mixer", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = construct_config(opts=args.opts)
    spot = (RemoteSpot if config.USE_REMOTE_SPOT else Spot)("RealSeqEnv")
    if config.USE_REMOTE_SPOT:
        try:
            main(spot, args.use_mixer, config, args.output)
        finally:
            spot.power_off()
    else:
        with spot.get_lease(hijack=True):
            try:
                main(spot, args.use_mixer, config, args.output)
            finally:
                spot.shutdown(should_dock=True)
