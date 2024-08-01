# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import time

import magnum as mn
import numpy as np
import rospy
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.utils import ros_topics as rt
from spot_wrapper.utils import get_angle_between_two_vectors

NUM_REPEAT = 5
WAYPOINT_TEST = [[1.8, 1.1, 0.0]] * NUM_REPEAT  # x, y, yaw


class SpotRosSkillExecutor:
    """This class reads the ros butter to execute skills"""

    def __init__(self):
        self.spotskillmanager = None
        self._cur_skill_name_input = None

    def compute_metrics(self, traj, target_point):
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
            "num_steps": num_steps,
            "distance": distance,
            "dot_product_facing_target": dot_product_facing_target,
            "angle_facing_target": angle_facing_target,
        }

    def benchmark(self):
        """ "Run the benchmark code to test skills"""

        metrics_list = []
        for waypoint in WAYPOINT_TEST:
            # Call the skill manager
            self.spotskillmanager = SpotSkillManager(
                use_mobile_pick=True, use_semantic_place=False
            )
            x, y, theta = waypoint
            suc, _ = self.spotskillmanager.nav(x, y, theta)
            # Compute the metrics
            traj = (
                self.spotskillmanager.nav_controller.get_most_recent_result_log().get(
                    "robot_trajectory"
                )
            )
            metrics = self.compute_metrics(traj, np.array([x, y]))
            metrics["suc"] = suc
            metrics_list.append(metrics)
            # Reset
            self.spotskillmanager.dock()

        # Compute the final number
        for mm in metrics:
            data = [metrics[mm] for metrics in metrics_list]
            _mean = np.mean(data)
            _std = np.std(data)
            print(f"{mm}: {_mean} +- {_std}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--useful_parameters", action="store_true")
    _ = parser.parse_args()

    executor = SpotRosSkillExecutor()
    executor.benchmark()


if __name__ == "__main__":
    main()
