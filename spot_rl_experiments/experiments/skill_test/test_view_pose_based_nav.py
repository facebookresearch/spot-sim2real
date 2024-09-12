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
from spot_rl.utils.retrieve_robot_poses_from_cg import get_view_poses
from spot_rl.utils.utils import ros_topics as rt
from spot_rl.utils.waypoint_estimation_based_on_robot_poses_from_cg import (
    get_navigation_points,
)
from spot_wrapper.utils import get_angle_between_two_vectors

NUM_REPEAT = 1
WAYPOINT_TEST = [([4.3, -2.0, 0.6], [0.9, 0.7, 0.3], ["sink"])] * NUM_REPEAT  # x, y
receptacle_name = "sink"
VISUALIZE = True
PATH_PLANNING_VISUALIZATION_FOLDER = "path_planning_vis_for_cg"
os.makedirs(PATH_PLANNING_VISUALIZATION_FOLDER, exist_ok=True)


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

            bbox_center, bbox_extent, query_class_names = waypoint
            bbox_center = np.array(bbox_center)
            bbox_extent = np.array(bbox_extent)

            # Get the view poses
            view_poses = get_view_poses(
                bbox_center, bbox_extent, query_class_names, VISUALIZE
            )
            # Get the robot x, y, yaw
            x, y, _ = self.spotskillmanager.spot.get_xy_yaw()
            # Get the navigation points
            nav_pts = get_navigation_points(
                view_poses,
                bbox_center,
                bbox_extent,
                [x, y],
                VISUALIZE,
                osp.join(PATH_PLANNING_VISUALIZATION_FOLDER, f"{receptacle_name}.png"),
            )
            # breakpoint()
            # Sequentially give the points
            if len(nav_pts) > 0:
                final_pt_i = len(nav_pts) - 1
                agg_metrics = {
                    "num_steps": 0,
                    "distance": -1,
                    "dot_product_facing_target": -1,
                    "angle_facing_target": -1,
                }
                for pt_i, pt in enumerate(nav_pts):
                    x, y, yaw = pt
                    if pt_i == final_pt_i:
                        # Do normal point nav with yaw for the final location
                        succeded, msg = self.spotskillmanager.nav(x, y, yaw, False)
                    else:
                        # Do dynamic point yaw here for the intermediate points
                        succeded, msg = self.spotskillmanager.nav(x, y)
                    # Compute the metrics
                    traj = self.spotskillmanager.nav_controller.get_most_recent_result_log().get(
                        "robot_trajectory"
                    )
                    metrics = self.compute_metrics(traj, np.array([x, y]))
                    agg_metrics["num_steps"] += metrics["num_steps"]
                    agg_metrics["distance"] = metrics["distance"]
                    agg_metrics["dot_product_facing_target"] = metrics[
                        "dot_product_facing_target"
                    ]
                    agg_metrics["angle_facing_target"] = metrics["angle_facing_target"]
                    agg_metrics["suc"] = succeded
                metrics_list.append(agg_metrics)

            # breakpoint()
            # Reset
            self.spotskillmanager.dock()

        # Compute the final number
        for mm in agg_metrics:
            data = [agg_metrics[mm] for agg_metrics in metrics_list]
            _mean = round(np.mean(data), 2)
            _std = round(np.std(data), 2)
            print(f"{mm}: {_mean} +- {_std}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--useful_parameters", action="store_true")
    _ = parser.parse_args()

    executor = SpotRosSkillExecutor()
    executor.benchmark()


if __name__ == "__main__":
    main()
