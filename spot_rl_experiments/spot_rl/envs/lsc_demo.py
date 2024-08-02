# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import subprocess
import time
from collections import Counter
from typing import Any, Dict

import numpy as np
import rospy
from hydra import compose, initialize
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.llm.src.rearrange_llm import RearrangeEasyChain
from spot_rl.models.sentence_similarity import SentenceSimilarity
from spot_rl.utils.utils import construct_config, get_default_parser, get_waypoint_yaml
from spot_rl.utils.whisper_translator import WhisperTranslator

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 549))


def main(spot, config):
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
            try:
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
            except Exception as e:
                print(f"Exception encountered in Speech to text : {e} \n\n Retrying...")
        # Used for Owlvit
        rospy.set_param("object_target", pick)

        # Used for Visualizations
        rospy.set_param("viz_pick", nav_1)
        rospy.set_param("viz_object", pick)
        rospy.set_param("viz_place", nav_2)

        if pre_in_dock:
            spot.spot.power_robot()

        # Do LSC by calling skills
        slow_down = 0.5  # slow down time to increase the skill execution stability
        spot.nav(nav_1)
        time.sleep(slow_down)
        spot.pick(pick)
        time.sleep(slow_down)
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
    spotskillmanager = SpotSkillManager(use_mobile_pick=True)
    main(spotskillmanager, config)
