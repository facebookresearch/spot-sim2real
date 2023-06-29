import os
import time
from cv2 import cv2

import numpy as np
from spot_wrapper.spot import Spot
from spot_rl.utils.utils import ros_topics as rt
import einops
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
    # open the gripper so the camera is not occluded
    spot.open_gripper()
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
            # default is one step can go 0.5 m/s for 0.5s and rotate at 30deg/s for 0.5s
            observations, _, done, _ = env.step(base_action=action)
            for k,v in observations.items():
                print(k,v.__class__)
            print('pos: ', observations['position'], 'yaw: ',observations['yaw'])
            # import pdb; pdb.set_trace()
            depth = observations['hand_depth']
            # depth = observations['hand_depth_raw']
            vis_depth = einops.repeat(depth,'r c -> r c 3')
            vis_im = np.concatenate((observations['hand_rgb'],vis_depth),1)

            cv2.imshow("vis",vis_im)
            key = cv2.waitKey(1)
            # forward
            if key == ord('w'):
                action = [1,0]
            # back
            elif key == ord('s'):
                action = [-1,0]
            # rotate right
            elif key == ord('a'):
                action = [0,1]
            # rotate left
            elif key == ord('d'):
                action = [0,-1]
            elif key == ord('z'):
                done = True
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
        super().__init__(config, spot,no_raw=False)
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
        observations = {}
        # Get visual observations
        front_depth = self.msg_to_cv2(self.filtered_head_depth, "mono8")
        front_depth = cv2.resize(
            front_depth, (120 * 2, 212), interpolation=cv2.INTER_AREA
        )
        front_depth = np.float32(front_depth) / 255.0
        # Add dimension for channel (unsqueeze)
        front_depth = front_depth.reshape(*front_depth.shape[:2], 1)
        observations["spot_right_depth"], observations["spot_left_depth"] = np.split(
            front_depth, 2, 1
        )
        # Get rho theta observation
        curr_xy = np.array([self.x, self.y], dtype=np.float32)
        observations['position'] = curr_xy
        observations['yaw'] = self.yaw
        observations['hand_depth'] = self.msg_to_cv2(self.filtered_hand_depth, "mono8")
        observations['hand_depth_raw'] = self.msg_to_cv2(self.raw_hand_depth, "mono8")
        observations['hand_rgb'] = self.msg_to_cv2(self.hand_rgb, "rgb8")
        return observations

    @property
    def hand_rgb(self):
        return self.msgs[rt.HAND_RGB]
    
    @property
    def raw_hand_depth(self):
        return self.msgs[rt.HAND_DEPTH]

if __name__ == "__main__":
    spot = Spot("RealNavEnv")
    with spot.get_lease(hijack=True):
        main(spot)
