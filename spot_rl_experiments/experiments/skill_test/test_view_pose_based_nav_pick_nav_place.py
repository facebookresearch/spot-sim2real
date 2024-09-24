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
from skill_execution_with_benchmark import SpotSkillExecuterWithBenchmark
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.retrieve_robot_poses_from_cg import get_view_poses
from spot_rl.utils.utils import ros_topics as rt
from spot_rl.utils.waypoint_estimation_based_on_robot_poses_from_cg import (
    get_navigation_points,
)
from spot_wrapper.utils import get_angle_between_two_vectors

NUM_REPEAT = 1
# Center, extent, object_tags
WAYPOINT_TEST = [
    (
        {
            "id": 53,
            "bbox_extent": [0.7, 0.5, 0.5],
            "bbox_center": [3.5, -1.2, 0.3],
            "object_tag": "chair",
        },
        {
            "id": 71,
            "bbox_extent": [0.9, 0.9, 0.6],
            "bbox_center": [1.1, 3.3, -0.1],
            "object_tag": "coffee table",
        },
    )
] * NUM_REPEAT
object_to_pickup = "bottle"
VISUALIZE = False
PATH_PLANNING_VISUALIZATION_FOLDER = "path_planning_vis_for_cg"
os.makedirs(PATH_PLANNING_VISUALIZATION_FOLDER, exist_ok=True)


class SpotRosSkillExecutor(SpotSkillExecuterWithBenchmark):
    def nav(self, bbox_center, bbox_extent, query_class_names, metrics_list):
        bbox_center = np.array(bbox_center)
        bbox_extent = np.array(bbox_extent)
        # Get the view poses
        view_poses = get_view_poses(
            bbox_center, bbox_extent, query_class_names, None, False
        )
        # Get the robot x, y, yaw
        x, y, _ = self.spotskillmanager.spot.get_xy_yaw()
        # Get the navigation points
        nav_pts = get_navigation_points(
            view_poses,
            bbox_center,
            bbox_extent,
            [x, y],
            False,
            osp.join(PATH_PLANNING_VISUALIZATION_FOLDER, "nav_receptacle.png"),
        )

        # Sequentially give the points
        if len(nav_pts) > 0:
            final_pt_i = len(nav_pts) - 1
            agg_metrics = {
                "num_steps": 0,
                "distance": -1,
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
                metrics = self.compute_metrics(traj, np.array([x, y]), name_key="")
                agg_metrics["num_steps"] += metrics["num_steps"]
                agg_metrics["distance"] = metrics["distance"]
                agg_metrics["suc"] = succeded
            metrics_list.append(agg_metrics)
        return metrics_list, succeded

    def benchmark(self):
        """ "Run the benchmark code to test skills"""

        metrics_list = []
        for waypoint in WAYPOINT_TEST:
            # Call the skill manager
            self.spotskillmanager = SpotSkillManager(
                use_mobile_pick=True, use_semantic_place=True
            )
            pick_location_address, plac_location_address = waypoint

            _, bbox_extent, bbox_center, query_class_names = list(
                pick_location_address.values()
            )

            id, bbox_extent_place, bbox_center_place, query_class_place = list(
                plac_location_address.values()
            )

            if isinstance(query_class_names, str):
                query_class_names = [query_class_names]
            if isinstance(query_class_place, str):
                query_class_place = [query_class_place]

            metric_list, nav_suc = self.nav(
                bbox_center, bbox_extent, query_class_names, metrics_list
            )
            if nav_suc:
                self.spotskillmanager.pick(object_to_pickup)
                metric_list, nav_suc = self.nav(
                    bbox_center_place, bbox_extent_place, query_class_place, metric_list
                )
                if nav_suc:
                    self.spotskillmanager.place(
                        f"{id}_{query_class_place[0]}",
                        is_local=True,
                        visualize=False,
                        enable_waypoint_estimation=True,
                    )

            # Reset
            self.spotskillmanager.dock()

        if len(metric_list) > 0:
            agg_metrics = metric_list[0]
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
