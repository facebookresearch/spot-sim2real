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

NUM_REPEAT = 5
WAYPOINT_TEST = [[1.8, 1.1]] * NUM_REPEAT  # x, y


class SpotRosSkillExecutor(SpotSkillExecuterWithBenchmark):
    def benchmark(self):
        """ "Run the benchmark code to test skills"""

        metrics_list = []
        for waypoint in WAYPOINT_TEST:
            # Call the skill manager
            self.spotskillmanager = SpotSkillManager(
                use_mobile_pick=True, use_semantic_place=False
            )
            x, y = waypoint
            suc, _ = self.spotskillmanager.nav(x, y)
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
