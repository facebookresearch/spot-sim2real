import os
import time

import numpy as np
from spot_wrapper.spot import Spot

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
    config.USE_MRCNN = False

    policy = NavPolicy(config.WEIGHTS.NAV, device=config.DEVICE)
    policy.reset()

    env = SpotNavEnv(config, spot)
    env.power_robot()
    
    should_dock = args.dock
    if args.waypoint is not None:
        goal_x, goal_y, goal_heading = nav_target_from_waypoints(args.waypoint)
        env.say(f"Navigating to {args.waypoint}")
        if 'dock' == args.waypoint:
            should_dock=True
    else:
        assert args.goal is not None
        goal_x, goal_y, goal_heading = [float(i) for i in args.goal.split(",")]
    observations = env.reset((goal_x, goal_y), goal_heading)
    done = False
    time.sleep(1)
    try:
        while not done:
            action = policy.act(observations)
            observations, _, done, _ = env.step(base_action=action)
            if should_dock:
                result = try_docking(spot)
                if result:
                    print("Docked successfully, homing robot")
                    spot.home_robot()
                    break

    finally:
        spot.power_off()

def try_docking(spot):
    try:
        spot.dock(dock_id=DOCK_ID, home_robot=True)
        return True
    except:
        return False

class SpotNavEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        super().__init__(config, spot)
        self.goal_xy = None
        self.goal_heading = None
        self.succ_distance = config.SUCCESS_DISTANCE
        self.succ_angle = np.deg2rad(config.SUCCESS_ANGLE_DIST)
        self.docking = False

    def reset(self, goal_xy, goal_heading, docking=False):
        self.goal_xy = np.array(goal_xy, dtype=np.float32)
        self.goal_heading = goal_heading
        self.docking = docking
        observations = super().reset()
        assert len(self.goal_xy) == 2

        return observations
    
    # def step(self, *args, **kwargs):
    #     ret = self.step(*args, **kwargs)
    #     if self.docking:
    #         try:
    #             self.spot.dock(dock_id=DOCK_ID, home_robot=True)
    #             ret[2] = True  # done
    #         except:
    #             pass
    #     return ret

    
    # def reset(self, waypoint=None):
    # # def reset(self, goal_xy, goal_heading):
    #     # Nav
    #     if waypoint is None:
    #         self.goal_xy = None
    #         self.goal_heading = None
    #     else:
    #         self.goal_xy, self.goal_heading = (waypoint[:2], waypoint[2])

    #     self.goal_xy = np.array(self.goal_xy, dtype=np.float32)
    #     self.goal_heading = self.goal_heading
    #     observations = super().reset()
    #     assert len(self.goal_xy) == 2

    #     return observations

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
