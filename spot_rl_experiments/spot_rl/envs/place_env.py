import magnum as mn
import numpy as np
from spot_wrapper.spot import Spot

from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.real_policy import PlacePolicy
from spot_rl.utils.utils import (
    construct_config,
    get_default_parser,
    place_target_from_waypoints,
)


def main(spot):
    parser = get_default_parser()
    parser.add_argument("-p", "--place_target")
    parser.add_argument("-w", "--waypoint")
    parser.add_argument("-l", "--target_is_local", action="store_true")
    args = parser.parse_args()
    config = construct_config(args.opts)

    # Don't need cameras for Place
    config.USE_HEAD_CAMERA = False
    config.USE_MRCNN = False

    if args.waypoint is not None:
        assert not args.target_is_local
        place_target = place_target_from_waypoints(args.waypoint)
    else:
        assert args.place_target is not None
        place_target = [float(i) for i in args.place_target.split(",")]
    env = SpotPlaceEnv(config, spot, place_target, args.target_is_local)
    env.power_robot()
    policy = PlacePolicy(config.WEIGHTS.PLACE, device=config.DEVICE)
    policy.reset()
    observations = env.reset()
    done = False
    env.say("Starting episode")
    while not done:
        action = policy.act(observations)
        observations, _, done, _ = env.step(arm_action=action)
    if done:
        while True:
            env.reset()
            spot.set_base_velocity(0, 0, 0, 1.0)


class SpotPlaceEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot, place_target, target_is_local=False):
        super().__init__(config, spot)
        self.place_target = np.array(place_target)
        self.place_target_is_local = target_is_local
        self.ee_gripper_offset = mn.Vector3(config.EE_GRIPPER_OFFSET)
        self.placed = False

    def reset(self, *args, **kwargs):
        # Move arm to initial configuration
        cmd_id = self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=0.75
        )
        self.spot.block_until_arm_arrives(cmd_id, timeout_sec=2)

        observations = super(SpotPlaceEnv, self).reset()
        self.placed = False
        return observations

    def step(self, place=False, *args, **kwargs):
        _, xy_dist, z_dist = self.get_place_distance()
        place = xy_dist < self.config.SUCC_XY_DIST and z_dist < self.config.SUCC_Z_DIST
        return super().step(place=place, *args, **kwargs)

    def get_success(self, observations):
        return self.place_attempted

    def get_observations(self):
        observations = {
            "joint": self.get_arm_joints(),
            "obj_start_sensor": self.get_place_sensor(),
        }

        return observations


if __name__ == "__main__":
    spot = Spot("RealPlaceEnv")
    with spot.get_lease(hijack=True):
        main(spot)
