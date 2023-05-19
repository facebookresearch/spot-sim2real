import os
import time

import numpy as np
from spot_wrapper.spot import Spot

from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.real_policy import NavPolicy
from spot_rl.utils.utils import (
    WAYPOINTS,
    construct_config,
    get_default_parser,
    nav_target_from_waypoints,
)
from spot_rl.utils.whisper_translator import WhisperTranslator
from spot_rl.models.sentence_similarity import SentenceSimilarity
from spot_rl.llm.src.rearrange_llm import RearrangeEasyChain
from hydra import compose, initialize

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))


def main(spot):
    parser = get_default_parser()
    parser.add_argument("-g", "--goal")
    parser.add_argument("-w", "--waypoint")
    parser.add_argument("-d", "--dock", action="store_true")
    args = parser.parse_args()
    config = construct_config(args.opts)

    # Don't need gripper camera for Nav
    config.USE_MRCNN = False

    policy = NavPolicy(config.WEIGHTS.NAV, device=config.DEVICE)
    policy.reset()


    ##############################################
    # Language input code - Start #
    ##############################################

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

    # # Used for Owlvit
    # rospy.set_param('object_target', pick)

    # # Used for Visualizations
    # rospy.set_param('viz_pick', nav_1)
    # rospy.set_param('viz_object', pick)
    # rospy.set_param('viz_place', nav_2)

    ##############################################
    # Language input code - End #
    ##############################################
    env = SpotNavEnv(config, spot)
    env.power_robot()

    try:
        # First nav target
        nav_to_receptacle(env, nav_1, policy)

        # Second nav target
        nav_to_receptacle(env, nav_2, policy)

        # Dock
        nav_to_receptacle(env, "dock", policy)
        spot.dock(dock_id=DOCK_ID, home_robot=True)
        spot.home_robot()
    finally:
        spot.power_off()

    
def nav_to_receptacle(env, receptacle_name, policy):
    goal_x, goal_y, goal_heading = nav_target_from_waypoints(receptacle_name)

    observations = env.reset((goal_x, goal_y), goal_heading)
    done = False
    time.sleep(1)

    while not done:
        action = policy.act(observations)
        observations, _, done, _ = env.step(base_action=action)
        env.print_nav_stats(observations) 


class SpotNavEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        super().__init__(config, spot)
        self.goal_xy = None
        self.goal_heading = None
        self.succ_distance = config.SUCCESS_DISTANCE
        self.succ_angle = np.deg2rad(config.SUCCESS_ANGLE_DIST)

    def reset(self, goal_xy, goal_heading):
        self.goal_xy = np.array(goal_xy, dtype=np.float32)
        self.goal_heading = goal_heading
        observations = super().reset()
        assert len(self.goal_xy) == 2

        return observations

    def get_success(self, observations):
        succ = self.get_nav_success(observations, self.succ_distance, self.succ_angle)
        if succ:
            self.spot.set_base_velocity(0.0, 0.0, 0.0, 1 / self.ctrl_hz)
        return succ

    def get_observations(self):
        return self.get_nav_observation(self.goal_xy, self.goal_heading)


if __name__ == "__main__":
    spot = Spot("RealNavEnv")
    with spot.get_lease(hijack=True):
        main(spot)
