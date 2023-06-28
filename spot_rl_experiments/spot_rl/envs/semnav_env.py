import os
import time
from cv2 import cv2

import numpy as np
from spot_wrapper.spot import Spot
from spot_rl.utils.utils import ros_topics as rt

from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.real_policy import NavPolicy
from spot_rl.utils.utils import (
    construct_config,
    get_default_parser,
    nav_target_from_waypoints,
)

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))


def main(spot):
    parser = get_default_parser()
    parser.add_argument("-g", "--goal")
    parser.add_argument("-w", "--waypoint")
    parser.add_argument("-d", "--dock", action="store_true")
    args = parser.parse_args()
    config = construct_config(args.opts)

    # Don't need gripper camera for Nav
    # config.USE_MRCNN = False

    # policy = NavPolicy(config.WEIGHTS.NAV, device=config.DEVICE)
    # policy.reset()

    env = SpotSemanticNavEnv(config, spot)
    env.power_robot()
    # if args.waypoint is not None:
        # goal_x, goal_y, goal_heading = nav_target_from_waypoints(args.waypoint)
        # env.say(f"Navigating to {args.waypoint}")
    # else:
        # assert args.goal is not None
        # goal_x, goal_y, goal_heading = [float(i) for i in args.goal.split(",")]
    observations = env.reset()
    done = False
    time.sleep(1)
    action = [0,0]
    try:
        while not done:
            # action = policy.act(observations)
            # lin_dist, ang_dist = base_action
            # this is from [-1,1] which scales based on MAX_LIN_DIST and MAX_ANG_DIST in the config
            # it computes speed assuming based on this distance and a control frequence of config.CTRL_HZ (default 2hz)
            observations, _, done, _ = env.step(base_action=action)
            for k,v in observations.items():
                print(k,v.__class__)
            if cv2.waitKey(0) == ord('w'):
                action = [1,0]
            if cv2.waitKey(0) == ord('s'):
                action = [-1,0]
            else:
                action = [0,0]
        if args.dock:
            env.say("Executing automatic docking")
            dock_start_time = time.time()
            while time.time() - dock_start_time < 2:
                try:
                    spot.dock(dock_id=DOCK_ID, home_robot=True)
                except:
                    print("Dock not found... trying again")
                    time.sleep(0.1)
    finally:
        spot.power_off()


class SpotSemanticNavEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        super().__init__(config, spot)
        self.goal_xy = None
        self.goal_heading = None
        self.succ_distance = config.SUCCESS_DISTANCE
        self.succ_angle = np.deg2rad(config.SUCCESS_ANGLE_DIST)

    def reset(self):
        observations = super().reset()
        return observations

    def get_success(self, observations):
        return False

    def get_observations(self):
        observations = self.get_nav_observation(self.goal_xy, self.goal_heading)
        observations['arm_depth'] = self.msg_to_cv2(self.filtered_hand_depth, "mono8")
        observations['hand_rgb'] = self.msg_to_cv2(self.hand_rgb, "rgb8")
        return observations

    def hand_rgb(self):
        return self.msgs[rt.HAND_RGB]

if __name__ == "__main__":
    spot = Spot("RealNavEnv")
    with spot.get_lease(hijack=True):
        main(spot)
