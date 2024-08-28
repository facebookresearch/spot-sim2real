# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# mypy: ignore-errors
import argparse
import atexit
import os
import os.path as osp
import signal
import sys
import time
from copy import deepcopy

import magnum as mn
import numpy as np
import rospy
from perception_and_utils.utils.generic_utils import map_user_input_to_boolean
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.utils import ros_topics as rt
from spot_wrapper.utils import get_angle_between_two_vectors

NUM_REPEAT = 1
WAYPOINT_TEST = [([3.6, -4.2], [3.6, -4.2])] * NUM_REPEAT  # x, y
OBJECT_TO_PICK = "tomato"

metrics_list = None
metrics = None


def print_logs():
    print(f"Current metrics while exiting {metrics_list}, Metrics: {metrics}")


class SpotRosSkillExecutor:
    """This class reads the ros butter to execute skills"""

    def __init__(self):
        self.spotskillmanager = None
        self._cur_skill_name_input = None

    def compute_metrics(self, traj, target_point, name_key="nav1"):
        """ "Compute the metrics"""
        num_steps = len(traj)
        final_point = np.array(traj[-1]["pose"][0:2])
        distance = np.linalg.norm(target_point - final_point)
        # Compute the angle
        vector_robot_to_target = target_point - final_point
        vector_robot_to_target = vector_robot_to_target / np.linalg.norm(
            vector_robot_to_target
        )
        vector_forward_robot = np.array(
            self.spotskillmanager.get_env().curr_transform.transform_vector(
                mn.Vector3(1, 0, 0)
            )
        )[[0, 1]]
        vector_forward_robot = vector_forward_robot / np.linalg.norm(
            vector_forward_robot
        )
        dot_product_facing_target = abs(
            np.dot(vector_robot_to_target, vector_forward_robot)
        )
        angle_facing_target = abs(
            get_angle_between_two_vectors(vector_robot_to_target, vector_forward_robot)
        )
        return {
            f"num_steps_{name_key}": num_steps,
            f"distance_{name_key}": distance,
            f"dot_product_facing_target_{name_key}": dot_product_facing_target,
            f"angle_facing_target_{name_key}": angle_facing_target,
        }

    def benchmark(self):
        """ "Run the benchmark code to test skills"""
        global metrics_list
        global metrics

        metrics_list = []
        for waypoint in WAYPOINT_TEST:
            # Call the skill manager
            self.spotskillmanager = SpotSkillManager(
                use_mobile_pick=True, use_semantic_place=True
            )
            # parse two nav waypoints
            waypoint0, waypoint1 = waypoint
            x1, y1 = waypoint0
            x2, y2 = waypoint1
            # do first nav
            suc, _ = self.spotskillmanager.nav(x1, y1)
            # Compute the metrics
            traj = (
                self.spotskillmanager.nav_controller.get_most_recent_result_log().get(
                    "robot_trajectory"
                )
            )
            metrics = self.compute_metrics(traj, np.array([x1, y1]))
            metrics["nav1_suc"] = suc

            # iff nav succeeds then we do pick
            if suc:
                suc, _ = self.spotskillmanager.pick(OBJECT_TO_PICK)
            else:
                suc = False
                self.spotskillmanager.spot.open_gripper()
            metrics["pick_suc"] = suc

            # iff pick succeeds nav to place target
            if suc:
                suc, _ = self.spotskillmanager.nav(x2, y2)
                # Compute the metrics
                traj = self.spotskillmanager.nav_controller.get_most_recent_result_log().get(
                    "robot_trajectory"
                )
                metrics2 = self.compute_metrics(traj, np.array([x2, y2]), "nav2")
                metrics.update(metrics2)
            else:
                suc = False
                self.spotskillmanager.spot.open_gripper()
            metrics["nav2_suc"] = suc

            # iff nav to place succeeds then we do semantic place
            if suc:
                suc, _ = self.spotskillmanager.place(
                    None, is_local=True, visualize=False
                )
                rospy.set_param("is_gripper_blocked", 0)

            else:
                suc = False
            self.spotskillmanager.spot.open_gripper()
            metrics["place_suc"] = suc

            metrics_list.append(metrics)
            # Reset
            self.spotskillmanager.dock()

        # Compute the final number
        for mm in metrics:
            data = [metrics[mm] for metrics in metrics_list]
            _mean = round(np.mean(data), 2)
            _std = round(np.std(data), 2)
            print(f"{mm}: {_mean} +- {_std}")


atexit.register(print_logs)
# Register the signal handler for termination signals
signal.signal(signal.SIGINT, print_logs)
signal.signal(signal.SIGTERM, print_logs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--useful_parameters", action="store_true")
    _ = parser.parse_args()

    executor = SpotRosSkillExecutor()
    executor.benchmark()


if __name__ == "__main__":
    main()
