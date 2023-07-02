import os
import time

import _pickle as cPickle
import numpy as np
import quaternion
import skimage
import skimage.morphology
import spot_rl.utils.pose as pu
import torch
import torch.nn as nn
from semantic_exploration.agents.sem_exp import Sem_Exp_Env_Agent
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.models.semantic_map import Semantic_Mapping
from spot_rl.real_policy import NavPolicy
from spot_rl.utils.utils import (
    construct_config,
    get_default_parser,
    nav_target_from_waypoints,
)
from spot_wrapper.spot import Spot, wrap_heading

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

    env = SpotSemExpEnv(config, spot)
    env.power_robot()
    if args.waypoint is not None:
        goal_x, goal_y, goal_heading = nav_target_from_waypoints(args.waypoint)
        env.say(f"Navigating to {args.waypoint}")
    else:
        assert args.goal is not None
        goal_x, goal_y, goal_heading = [float(i) for i in args.goal.split(",")]

    # Reset the info to get the first observation
    observations = env.reset((goal_x, goal_y), goal_heading)
    # Once we have the first observation, we then can process the input
    env.map_init()

    done = False
    time.sleep(1)

    try:
        while not done:
            # Policy control method to drive the robot movement
            # action = policy.act(observations)
            # observations, state, done = env.step(base_action=action)

            # Direct control the robot
            # action: 1: forward; action 2: left; action: 3: right
            if env.info["action"] == 1:
                vel = [config.BASE_LIN_VEL, 0.0, 0.0]  # go forward
            elif env.info["action"] == 2:
                vel = [0.0, 0.0, config.BASE_ANGULAR_VEL]  # turn left
            elif env.info["action"] == 3:
                vel = [0.0, 0.0, -config.BASE_ANGULAR_VEL]  # turn right
            else:
                vel = [0.0, 0.0, 0.0]
            # Control the spot robot
            spot.set_base_velocity(
                x_vel=vel[0],
                y_vel=vel[1],
                ang_vel=vel[2],
                vel_time=config.UPDATE_PERIOD * 2,
            )
            # Update the observation without giving input
            observations, state, done = env.step()

            # Update the map here
            env.map_update()

        if args.dock:
            env.say("Executing automatic docking")
            dock_start_time = time.time()
            while time.time() - dock_start_time < 2:
                try:
                    spot.dock(dock_id=DOCK_ID, home_robot=True)
                except Exception:
                    print("Dock not found... trying again")
                    time.sleep(0.1)
    finally:
        spot.power_off()


class SpotSemExpEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        super().__init__(config, spot)
        self.goal_xy = None
        self.goal_heading = None
        self.succ_distance = config.SUCCESS_DISTANCE
        self.succ_angle = np.deg2rad(config.SUCCESS_ANGLE_DIST)
        self.cur_observation = None
        self.num_sem_categories = config.NUM_SEM_CATEGORIES
        self.last_sim_location = None
        self.config = config
        self.num_scenes = 1
        self.info = {}  # type: ignore
        self.agent = Sem_Exp_Env_Agent(config=config)
        self.i_step = 0
        self.g_step = (
            self.i_step // self.config.NUM_LOCAL_STEPS
        ) % self.config.NUM_GLOBAL_STEPS
        self.l_step = self.i_step % self.config.NUM_LOCAL_STEPS

    def map_init(self):
        """Function to initialize the map"""
        # Initialize map variables:
        # Full map consists of multiple channels containing the following:
        # 1. Obstacle Map
        # 2. Exploread Area
        # 3. Current Agent Location
        # 4. Past Agent Locations
        # 5,6,7,.. : Semantic Categories
        nc = self.num_sem_categories + 1

        # Calculating full and local map sizes
        map_size = self.config.MAP_SIZE_CM // self.config.MAP_RESOLUTION
        self.full_w, self.full_h = map_size, map_size
        self.local_w = int(self.full_w / self.config.GLOBAL_DOWNSCALING)
        self.local_h = int(self.full_h / self.config.GLOBAL_DOWNSCALING)

        # Initializing full and local map
        self.full_map = (
            torch.zeros(self.num_scenes, nc, self.full_w, self.full_h)
            .float()
            .to(self.config.DEVICE)
        )
        self.local_map = (
            torch.zeros(self.num_scenes, nc, self.local_w, self.local_h)
            .float()
            .to(self.config.DEVICE)
        )

        # Initial full and local pose
        self.full_pose = torch.zeros(self.num_scenes, 3).float().to(self.config.DEVICE)
        self.local_pose = torch.zeros(self.num_scenes, 3).float().to(self.config.DEVICE)

        # Origin of local map
        self.origins = np.zeros((self.num_scenes, 3))

        # Local Map Boundaries
        self.lmb = np.zeros((self.num_scenes, 4)).astype(int)

        # Planner pose inputs has 7 dimensions
        # 1-3 store continuous global agent location
        # 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((self.num_scenes, 7))

        # Iinitialize the map and the pose
        self.full_map.fill_(0.0)
        self.full_pose.fill_(0.0)
        self.full_pose[:, :2] = self.config.MAP_SIZE_CM / 100.0 / 2.0

        locs = self.full_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = self.locs
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / self.config.MAP_RESOLUTION),
                int(c * 100.0 / self.config.MAP_RESOLUTION),
            ]

            self.full_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0

            self.lmb[e] = self.get_local_map_boundaries(
                (loc_r, loc_c), (self.local_w, self.local_h), (self.full_w, self.full_h)
            )

            self.planner_pose_inputs[e, 3:] = self.lmb[e]
            self.origins[e] = [
                self.lmb[e][2] * self.config.MAP_RESOLUTION / 100.0,
                self.lmb[e][0] * self.config.MAP_RESOLUTION / 100.0,
                0.0,
            ]

        for e in range(self.num_scenes):
            self.local_map[e] = self.full_map[
                e, :, self.lmb[e, 0] : self.lmb[e, 1], self.lmb[e, 2] : self.lmb[e, 3]
            ]
            self.local_pose[e] = (
                self.full_pose[e]
                - torch.from_numpy(self.origins[e]).to(self.config.DEVICE).float()
            )

        # Global policy observation space
        ngc = 8 + self.num_sem_categories

        # Semantic Mapping
        self.sem_map_module = Semantic_Mapping(self.config).to(self.config.DEVICE)
        self.sem_map_module.eval()

        self.global_input = torch.zeros(
            self.num_scenes, ngc, self.local_w, self.local_h
        )
        self.global_orientation = torch.zeros(self.num_scenes, 1).long()
        self.intrinsic_rews = torch.zeros(self.num_scenes).to(self.config.DEVICE)
        self.extras = torch.zeros(self.num_scenes, 2)

        # Predict semantic map from frame 1
        poses = (
            torch.from_numpy(
                np.asarray(
                    [self.info["sensor_pose"] for env_idx in range(self.num_scenes)]
                )
            )
            .float()
            .to(self.config.DEVICE)
        )

        _, self.local_map, _, self.local_pose = self.sem_map_module(
            self.agent._preprocess_obs(self.info["state"]),
            poses,
            self.local_map,
            self.local_pose,
        )

        # Compute Global policy input
        locs = self.local_pose.cpu().numpy()
        self.global_input = torch.zeros(
            self.num_scenes, ngc, self.local_w, self.local_h
        )
        self.global_orientation = torch.zeros(self.num_scenes, 1).long()

        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / self.config.MAP_RESOLUTION),
                int(c * 100.0 / self.config.MAP_RESOLUTION),
            ]

            self.local_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0
            self.global_orientation[e] = int((locs[e, 2] + 180.0) / 5.0)

            # Set a disk around the agent to explore
            try:
                radius = self.config.frontier_explore_radius
                explored_disk = skimage.morphology.disk(radius)
                self.local_map[
                    e,
                    1,
                    int(r - radius) : int(r + radius + 1),
                    int(c - radius) : int(c + radius + 1),
                ][explored_disk == 1] = 1
            except IndexError:
                pass

        self.global_input[:, 0:4, :, :] = self.local_map[:, 0:4, :, :].detach()
        self.global_input[:, 4:8, :, :] = nn.MaxPool2d(self.config.GLOBAL_DOWNSCALING)(
            self.full_map[:, 0:4, :, :]
        )
        self.global_input[:, 8:, :, :] = self.local_map[:, 4:, :, :].detach()

        goal_cat_id = torch.from_numpy(
            np.asarray([self.config.GOAL_CAT_ID for env_idx in range(self.num_scenes)])
        )

        self.extras = torch.zeros(self.num_scenes, 2)
        self.extras[:, 0] = self.global_orientation[:, 0]
        self.extras[:, 1] = goal_cat_id

        self.goal_maps = [
            np.zeros((self.local_w, self.local_h)) for _ in range(self.num_scenes)
        ]

        planner_inputs = [{} for e in range(self.num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input["map_pred"] = self.local_map[e, 0, :, :].cpu().numpy()
            p_input["exp_pred"] = self.local_map[e, 1, :, :].cpu().numpy()
            p_input["pose_pred"] = self.planner_pose_inputs[e]
            p_input["goal"] = self.goal_maps[e]  # global_goals[e]
            p_input["new_goal"] = 1
            p_input["found_goal"] = 0
            p_input["wait"] = False
            if self.config.VISUALIZE:
                self.local_map[e, -1, :, :] = 1e-5
                p_input["sem_map_pred"] = (
                    self.local_map[e, 4:, :, :].argmax(0).cpu().numpy()
                )

        # Reset the agent
        self.agent.reset(self.info["state"].shape)
        # We have to feed the info information in to the planner agent
        # Update the info and get the observation input for the map
        # self.info now contains action info
        self.obs_map_input, _, done, self.info = self.agent.plan_act_and_preprocess(
            planner_inputs, self.info
        )

    def udpate_step_counter(self):
        # Update the step variable
        self.i_step += 1
        self.g_step = (
            self.i_step // self.config.NUM_LOCAL_STEPS
        ) % self.config.NUM_GLOBAL_STEPS
        self.l_step = self.i_step % self.config.NUM_LOCAL_STEPS

    def map_update(self):
        # Semantic Mapping Module
        poses = (
            torch.from_numpy(
                np.asarray(
                    [self.info["sensor_pose"] for env_idx in range(self.num_scenes)]
                )
            )
            .float()
            .to(self.config.DEVICE)
        )

        # Set people as not obstacles
        self.local_map[:, 0, :, :] *= 1 - self.local_map[:, 19, :, :]

        # Update the smeantic map
        _, self.local_map, _, self.local_pose = self.sem_map_module(
            self.obs_map_input, poses, self.local_map, self.local_pose
        )

        # Set people as not obstacles for planning
        e = 0
        people_mask = (
            skimage.morphology.binary_dilation(
                self.local_map[e, 15, :, :].cpu().numpy(), skimage.morphology.disk(10)
            )
        ) * 1.0
        self.local_map[e, 0, :, :] *= 1 - torch.from_numpy(people_mask).to(
            self.config.DEVICE
        )

        locs = self.local_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = locs + self.origins
        self.local_map[:, 2, :, :].fill_(0.0)  # Resetting current location channel
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [
                int(r * 100.0 / self.config.MAP_RESOLUTION),
                int(c * 100.0 / self.config.MAP_RESOLUTION),
            ]
            self.local_map[e, 2:4, loc_r - 2 : loc_r + 3, loc_c - 2 : loc_c + 3] = 1.0

        # Update the global policy
        if self.l_step == self.config.NUM_LOCAL_STEPS - 1:
            # For every global step, update the full and local maps
            for e in range(self.num_scenes):
                self.update_intrinsic_rew(e)

                self.full_map[
                    e,
                    :,
                    self.lmb[e, 0] : self.lmb[e, 1],
                    self.lmb[e, 2] : self.lmb[e, 3],
                ] = self.local_map[e]
                self.full_pose[e] = (
                    self.local_pose[e]
                    + torch.from_numpy(self.origins[e]).to(self.config.DEVICE).float()
                )

                locs = self.full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [
                    int(r * 100.0 / self.config.MAP_RESOLUTION),
                    int(c * 100.0 / self.config.MAP_RESOLUTION),
                ]

                self.lmb[e] = self.get_local_map_boundaries(
                    (loc_r, loc_c),
                    (self.local_w, self.local_h),
                    (self.full_w, self.full_h),
                )

                self.planner_pose_inputs[e, 3:] = self.lmb[e]
                self.origins[e] = [
                    self.lmb[e][2] * self.config.MAP_RESOLUTION / 100.0,
                    self.lmb[e][0] * self.config.MAP_RESOLUTION / 100.0,
                    0.0,
                ]

                self.local_map[e] = self.full_map[
                    e,
                    :,
                    self.lmb[e, 0] : self.lmb[e, 1],
                    self.lmb[e, 2] : self.lmb[e, 3],
                ]
                self.local_pose[e] = (
                    self.full_pose[e]
                    - torch.from_numpy(self.origins[e]).to(self.config.DEVICE).float()
                )

            locs = self.local_pose.cpu().numpy()
            for e in range(self.num_scenes):
                self.global_orientation[e] = int((locs[e, 2] + 180.0) / 5.0)
            self.global_input[:, 0:4, :, :] = self.local_map[:, 0:4, :, :]
            self.global_input[:, 4:8, :, :] = nn.MaxPool2d(
                self.config.GLOBAL_DOWNSCALING
            )(self.full_map[:, 0:4, :, :])
            self.global_input[:, 8:, :, :] = self.local_map[:, 4:, :, :].detach()
            goal_cat_id = torch.from_numpy(
                np.asarray(
                    [self.config.GOAL_CAT_ID for env_idx in range(self.num_scenes)]
                )
            )
            self.extras[:, 0] = self.global_orientation[:, 0]
            self.extras[:, 1] = goal_cat_id

        # Update long-term goal if target object is found
        self.found_goal = [0 for _ in range(self.num_scenes)]
        self.goal_maps = [
            np.zeros((self.local_w, self.local_h)) for _ in range(self.num_scenes)
        ]

        for e in range(self.num_scenes):
            cn = self.config.GOAL_CAT_ID + 4
            if self.local_map[e, cn, :, :].sum() != 0.0:
                cat_semantic_map = self.local_map[e, cn, :, :].cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.0
                self.goal_maps[e] = cat_semantic_scores
                self.found_goal[e] = 1

        # Take action and get next observation
        planner_inputs = [{} for e in range(self.num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            p_input["map_pred"] = self.local_map[e, 0, :, :].cpu().numpy()
            p_input["exp_pred"] = self.local_map[e, 1, :, :].cpu().numpy()
            p_input["pose_pred"] = self.planner_pose_inputs[e]
            p_input["goal"] = self.goal_maps[e]  # global_goals[e]
            p_input["new_goal"] = self.l_step == self.config.NUM_LOCAL_STEPS - 1
            p_input["found_goal"] = self.found_goal[e]
            p_input["wait"] = False
            if self.config.VISUALIZE:
                self.local_map[e, -1, :, :] = 1e-5
                p_input["sem_map_pred"] = (
                    self.local_map[e, 4:, :, :].argmax(0).cpu().numpy()
                )

        self.obs_map_input, _, done, infos = self.agent.plan_act_and_preprocess(
            planner_inputs
        )

    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if self.config.GLOBAL_DOWNSCALING > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def init_map_and_pose_for_env(self, e):
        self.full_map[e].fill_(0.0)
        self.full_pose[e].fill_(0.0)
        self.full_pose[e, :2] = self.config.MAP_SIZE_CM / 100.0 / 2.0

        locs = self.full_pose[e].cpu().numpy()
        self.planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [
            int(r * 100.0 / self.config.MAP_RESOLUTION),
            int(c * 100.0 / self.config.MAP_RESOLUTION),
        ]

        self.full_map[e, 2:4, loc_r - 1 : loc_r + 2, loc_c - 1 : loc_c + 2] = 1.0

        self.lmb[e] = self.get_local_map_boundaries(
            (loc_r, loc_c), (self.local_w, self.local_h), (self.full_w, self.full_h)
        )

        self.planner_pose_inputs[e, 3:] = self.lmb[e]
        self.origins[e] = [
            self.lmb[e][2] * self.config.MAP_RESOLUTION / 100.0,
            self.lmb[e][0] * self.config.MAP_RESOLUTION / 100.0,
            0.0,
        ]

        self.local_map[e] = self.full_map[
            e, :, self.lmb[e, 0] : self.lmb[e, 1], self.lmb[e, 2] : self.lmb[e, 3]
        ]
        self.local_pose[e] = (
            self.full_pose[e]
            - torch.from_numpy(self.origins[e]).to(self.config.DEVICE).float()
        )

    def update_intrinsic_rew(self, e):
        prev_explored_area = self.full_map[e, 1].sum(1).sum(0)
        self.full_map[
            e, :, self.lmb[e, 0] : self.lmb[e, 1], self.lmb[e, 2] : self.lmb[e, 3]
        ] = self.local_map[e]
        curr_explored_area = self.full_map[e, 1].sum(1).sum(0)
        self.intrinsic_rews[e] = curr_explored_area - prev_explored_area
        self.intrinsic_rews[e] *= (self.config.MAP_RESOLUTION / 100.0) ** 2  # to m^2

    def reset(self, goal_xy, goal_heading):
        self.goal_xy = np.array(goal_xy, dtype=np.float32)
        self.goal_heading = goal_heading
        observations = super().reset()
        assert len(self.goal_xy) == 2

        # Get pose change
        dx, dy, do = self.get_pose_change()

        # Added observation
        self.info = {}
        self.info["sensor_pose"] = [dx, dy, do]
        rgb = observations["hand_rgb"].astype(np.uint8)
        depth = observations["hand_depth"]
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.info["state"] = state

        return observations

    def step(self, base_action=None):
        observations, _, done, _ = super().step(base_action=base_action)

        # Get pose change
        dx, dy, do = self.get_pose_change()

        # Added observation
        # When step is called, it will update the info
        # to get the most up-to-date infomation
        self.info = {}
        self.info["sensor_pose"] = [dx, dy, do]
        rgb = observations["hand_rgb"].astype(np.uint8)
        depth = observations["hand_depth"]
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        self.info["state"] = state
        # Update the step counter
        self.udpate_step_counter()
        return observations, state, done

    def get_success(self, observations):
        succ = False
        return succ

    def get_observations(self):
        self.cur_observation = self.get_sem_exp_observation(
            self.goal_xy, self.goal_heading
        )
        return self.cur_observation

    def get_real_location(self):
        """Returns x, y, o pose of the agent"""
        # Get the location of the agent
        x, y, o = (
            self.cur_observation["x"],
            self.cur_observation["y"],
            self.cur_observation["yaw"],
        )
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_real_pose = self.get_real_location()
        dx, dy, do = pu.get_rel_pose_change(curr_real_pose, self.last_real_location)
        self.last_real_location = curr_real_pose
        return dx, dy, do


if __name__ == "__main__":
    spot = Spot("RealNavEnv")
    with spot.get_lease(hijack=True):
        main(spot)
