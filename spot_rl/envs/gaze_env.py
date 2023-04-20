import time

import cv2
import numpy as np
from spot_wrapper.spot import Spot, wrap_heading

from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.real_policy import GazePolicy
from spot_rl.utils.utils import construct_config, get_default_parser
import os

DEBUG = False

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))

def run_env(spot, config, target_obj_id=None, orig_pos=None):

    # Don't need head cameras for Gaze
    config.USE_HEAD_CAMERA = False

    env = SpotGazeEnv(config, spot)
    # Here, we assign the target text name of the object
    env.owlvit_pick_up_object_name = "ball"
    env.power_robot()
    policy = GazePolicy(config.WEIGHTS.GAZE, device=config.DEVICE)
    policy.reset()
    observations = env.reset(target_obj_id=target_obj_id)
    done = False
    env.say("Starting episode")
    if orig_pos is None:
        orig_pos = (float(env.x), float(env.y), np.pi)
    while not done:
        action = policy.act(observations)
        observations, _, done, _ = env.step(arm_action=action)
    # print("Returning to original position...")
    # baseline_navigate(spot, orig_pos, limits=False)
    # print("Returned.")
    if done:
        # spot.dock(dock_id=DOCK_ID, home_robot=True)
        # import pdb; pdb.set_trace()
        while True:
            spot.set_base_velocity(0, 0, 0, 1.0)
    return done


def baseline_navigate(spot, waypoint, limits=True, **kwargs):
    goal_x, goal_y, goal_heading = waypoint
    if limits:
        cmd_id = spot.set_base_position(
            x_pos=goal_x,
            y_pos=goal_y,
            yaw=goal_heading,
            end_time=100,
            max_fwd_vel=0.5,
            max_hor_vel=0.05,
            max_ang_vel=np.deg2rad(30),
        )
    else:
        cmd_id = spot.set_base_position(
            x_pos=goal_x, y_pos=goal_y, yaw=goal_heading, end_time=100
        )
    cmd_status = None
    success = False
    traj = []
    st = time.time()
    while not success and time.time() < st + 20:
        if cmd_status != 1:
            traj.append((time.time(), *spot.get_xy_yaw()))
            time.sleep(0.5)
            feedback_resp = spot.get_cmd_feedback(cmd_id)
            cmd_status = (
                feedback_resp.feedback.synchronized_feedback.mobility_command_feedback
            ).se2_trajectory_feedback.status
        else:
            if limits:
                cmd_id = spot.set_base_position(
                    x_pos=goal_x,
                    y_pos=goal_y,
                    yaw=goal_heading,
                    end_time=100,
                    max_fwd_vel=0.5,
                    max_hor_vel=0.05,
                    max_ang_vel=np.deg2rad(30),
                )
            else:
                cmd_id = spot.set_base_position(
                    x_pos=goal_x, y_pos=goal_y, yaw=goal_heading, end_time=100
                )

        x, y, yaw = spot.get_xy_yaw()
        dist = np.linalg.norm(np.array([x, y]) - np.array([goal_x, goal_y]))
        heading_diff = abs(wrap_heading(goal_heading - yaw))
        success = dist < 0.3 and heading_diff < np.deg2rad(5)

    return traj


def close_enough(pos1, pos2):
    dist = np.linalg.norm(np.array([pos1[0] - pos2[0], pos1[1] - pos2[1]]))
    theta = abs(wrap_heading(pos1[2] - pos2[2]))
    return dist < 0.1 and theta < np.deg2rad(2)


class SpotGazeEnv(SpotBaseEnv):
    def reset(self, target_obj_id=None, *args, **kwargs):
        # Move arm to initial configuration
        cmd_id = self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=1
        )
        self.spot.block_until_arm_arrives(cmd_id, timeout_sec=1)
        print("Open gripper called in Gaze")
        self.spot.open_gripper()

        observations = super().reset(target_obj_id=target_obj_id, *args, **kwargs)

        # Reset parameters
        self.locked_on_object_count = 0
        if target_obj_id is None:
            self.target_obj_name = self.config.TARGET_OBJ_NAME

        return observations

    def step(self, base_action=None, arm_action=None, grasp=False, place=False):
        grasp = self.should_grasp()

        observations, reward, done, info = super().step(
            base_action, arm_action, grasp, place
        )

        return observations, reward, done, info

    def get_observations(self):
        arm_depth, arm_depth_bbox = self.get_gripper_images()
        if DEBUG:
            img = np.uint8(arm_depth_bbox * 255).reshape(*arm_depth_bbox.shape[:2])
            img2 = np.uint8(arm_depth * 255).reshape(*arm_depth.shape[:2])
            cv2.imwrite(f"arm_bbox_{self.num_steps:03}.png", img)
            cv2.imwrite(f"arm_depth_{self.num_steps:03}.png", img2)
        observations = {
            "joint": self.get_arm_joints(),
            "arm_depth": arm_depth,
            "arm_depth_bbox": arm_depth_bbox,
        }

        return observations

    def get_success(self, observations):
        return self.grasp_attempted


if __name__ == "__main__":
    spot = Spot("RealGazeEnv")
    parser = get_default_parser()
    parser.add_argument("--target-object", "-t")
    args = parser.parse_args()
    config = construct_config(args.opts)
    with spot.get_lease(hijack=True):
        run_env(spot, config, target_obj_id=args.target_object)
