import os
import time
from collections import Counter

import magnum as mn
import numpy as np
from spot_wrapper.spot import Spot
import rospy
from hydra import compose, initialize

from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.envs.gaze_env import SpotGazeEnv
from spot_rl.envs.mobile_manipulation_env import SequentialExperts, SpotMobileManipulationBaseEnv, SpotMobileManipulationSeqEnv, Tasks
from spot_rl.real_policy import GazePolicy, MixerPolicy, NavPolicy, PlacePolicy
from spot_rl.utils.remote_spot import RemoteSpot
from spot_rl.utils.utils import (
    WAYPOINTS,
    closest_clutter,
    construct_config,
    get_clutter_amounts,
    get_default_parser,
    nav_target_from_waypoints,
    object_id_to_nav_waypoint,
    place_target_from_waypoints,
)

from spot_rl.utils.whisper_translator import WhisperTranslator
from spot_rl.models.sentence_similarity import SentenceSimilarity
from spot_rl.llm.src.rearrange_llm import RearrangeEasyChain


DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))


def main(spot, use_mixer, config, out_path=None):
    if use_mixer:
        policy = MixerPolicy(
            config.WEIGHTS.MIXER,
            config.WEIGHTS.NAV,
            config.WEIGHTS.GAZE,
            config.WEIGHTS.PLACE,
            device=config.DEVICE,
        )
        env_class = SpotMobileManipulationBaseEnv
    else:
        policy = SequentialExperts(
            config.WEIGHTS.NAV,
            config.WEIGHTS.GAZE,
            config.WEIGHTS.PLACE,
            device=config.DEVICE,
        )
        env_class = SpotMobileManipulationSeqEnv

    env = env_class(config, spot)

    # Reset the viz params
    rospy.set_param('/viz_pick', 'None')
    rospy.set_param('/viz_object', 'None')
    rospy.set_param('/viz_place', 'None')

    audio_to_text = WhisperTranslator()
    sentence_similarity = SentenceSimilarity()
    with initialize(config_path='../llm/src/conf'):
       llm_config = compose(config_name='config')
    llm = RearrangeEasyChain(llm_config)

    print('I am ready to take instructions!\n Sample Instructions : take the rubik cube from the dining table to the hamper')
    print('-'*100)
    input('Are you ready?')
    audio_to_text.record()
    instruction = audio_to_text.translate()
    print('Transcribed instructions : ', instruction)

    # Use LLM to convert user input to an instructions set
    # Eg: nav_1, pick, nav_2 = 'bowl_counter', "container", 'coffee_counter'
    nav_1, pick, nav_2, _ = llm.parse_instructions(instruction)
    print('PARSED', nav_1, pick, nav_2)

    # Find closest nav_targets to the ones robot knows locations of
    nav_1 = sentence_similarity.get_most_similar_in_list(nav_1, list(WAYPOINTS['nav_targets'].keys()))
    nav_2 = sentence_similarity.get_most_similar_in_list(nav_2, list(WAYPOINTS['nav_targets'].keys()))
    print('MOST SIMILAR: ', nav_1, pick, nav_2)

    # Used for Owlvit
    rospy.set_param('object_target', pick)

    # Used for Visualizations
    rospy.set_param('viz_pick', nav_1)
    rospy.set_param('viz_object', pick)
    rospy.set_param('viz_place', nav_2)

    env.power_robot()
    time.sleep(1)
    count = Counter()
    out_data = []

    waypoint = nav_target_from_waypoints(nav_1)
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
        observations, _, done, info = env.step(
            base_action=base_action,
            arm_action=arm_action,
            nav_silence_only=nav_silence_only,
            use_default_target_obj_destination=False,
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
    env.say("Finished object rearrangement. Heading to dock.")
    waypoint = nav_target_from_waypoints("dock")
    observations = env.reset(waypoint=waypoint)
    expert = Tasks.NAV

    while True:
        base_action, arm_action = policy.act(observations, expert=expert)
        nav_silence_only = True
        env.stopwatch.record("policy_inference")
        observations, _, done, info = env.step(
            base_action=base_action,
            arm_action=arm_action,
            nav_silence_only=nav_silence_only,
        )
        try:
            spot.dock(dock_id=DOCK_ID, home_robot=True)
            break
        except:
            print("Dock not found... trying again")
            time.sleep(0.1)

    print("Done!")

    out_data.append((time.time(), env.x, env.y, env.yaw))

    if out_path is not None:
        data = (
            "\n".join([",".join([str(i) for i in t_x_y_yaw]) for t_x_y_yaw in out_data])
            + "\n"
        )
        with open(out_path, "w") as f:
            f.write(data)


if __name__ == "__main__":
    parser = get_default_parser()
    parser.add_argument("-m", "--use-mixer", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = construct_config(args.opts)
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
                spot.power_off()