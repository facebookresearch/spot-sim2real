
import os
import time
from cv2 import cv2
import glob
import numpy as np
from spot_wrapper.spot import Spot
from spot_rl.utils.utils import ros_topics as rt
import einops
from spot_rl.envs.base_env import SpotBaseEnv
from spot_rl.real_policy import NavPolicy
import rospy
from spot_rl.utils.utils import (
    construct_config,
    get_default_parser,
    nav_target_from_waypoints,
)

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))
from spot_rl.utils.robot_subscriber import SpotRobotSubscriberMixin

def main(spot):
    parser = get_default_parser()
    reader = SpotRobotSubscriberMixin(spot=spot)
    rate = rospy.Rate(3)
    done=False
    frames=[]
    recording = False
    write_dir = "/home/akshara/trajs"
    while not done:
        # action = policy.act(observations)
        # lin_dist, ang_dist = base_action
        # this is from [-1,1] which scales based on MAX_LIN_DIST and MAX_ANG_DIST in the config
        # it computes speed assuming based on this distance and a control frequence of config.CTRL_HZ (default 2hz)
        # default is one step can go 0.5 m/s for 0.5s and rotate at 30deg/s for 0.5s
        # import pdb; pdb.set_trace()
            

        
        filtered_depth = reader.msg_to_cv2(reader.msgs["/filtered_hand_depth"])
        rgb = reader.msg_to_cv2(reader.msgs["/hand_rgb"])
        keys = ['x','y','yaw','current_arm_pose','link_wr1_position','link_wr1_rotation']
        raw_depth = reader.msg_to_cv2(reader.msgs["/raw_hand_depth"])
        # import pdb; pdb.set_trace()
        state = {k: getattr(reader,k) for k in keys}
        vis_depth = einops.repeat(filtered_depth,'r c -> r c 3')
        vis_im = np.concatenate((rgb,vis_depth),1)
        state['rgb'] = rgb
        state['filtered_depth'] = filtered_depth
        state['raw_depth'] = raw_depth
        cv2.imshow("vis",vis_im)
        if recording:
            frames.append(state)

        key = cv2.waitKey(1)
        # forward
        if key == ord('w'):
            pass
        elif key == ord('r'):
            recording = True
            frames= []
            print("start recording")
        elif key == ord('s') and recording:
            written = glob.glob(f"{write_dir}/*.npy")
            ind = len(written)
            np.save(f"{write_dir}/{ind}.npy",frames)
            print(f"recording finished, saved {ind}.npy")
            frames = []
            recording = False
        elif key == ord('z'):
            done = True
        print(len(frames))
        rate.sleep()

UPDATE_PERIOD = 0.2
def cement_arm_joints(spot):
    arm_proprioception = spot.get_arm_proprioception()
    current_positions = np.array(
        [v.position.value for v in arm_proprioception.values()]
    )
    spot.set_arm_joint_positions(positions=current_positions, travel_time=UPDATE_PERIOD)


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

    def initialize_arm(self):
        INITIAL_POINT = np.array([0.5, 0.0, 0.7])
        INITIAL_RPY = np.deg2rad([0.0, 20.0, 0.0])
        point = INITIAL_POINT
        rpy = INITIAL_RPY
        cmd_id = spot.move_gripper_to_point(point, rpy)
        spot.block_until_arm_arrives(cmd_id, timeout_sec=1.5)
        cement_arm_joints(spot)
        return point, rpy

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
    main(spot)
