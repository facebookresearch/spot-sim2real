# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import time

import numpy as np
from spot_rl.utils.utils import (
    get_default_parser,
    get_waypoint_yaml,
    nav_target_from_waypoint,
)
from spot_wrapper.spot import Spot

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))


def main(spot):
    parser = get_default_parser()
    parser.add_argument("-g", "--goal")
    parser.add_argument("-w", "--waypoint")
    parser.add_argument("-d", "--dock", action="store_true")
    parser.add_argument("-l", "--limit", action="store_true")
    args = parser.parse_args()

    waypoint_yaml = get_waypoint_yaml()

    if args.waypoint is not None:
        goal_x, goal_y, goal_heading = nav_target_from_waypoint(
            args.waypoint, waypoints_yaml=waypoint_yaml
        )
    else:
        assert args.goal is not None
        goal_x, goal_y, goal_heading = [float(i) for i in args.goal.split(",")]

    if args.limit:
        kwargs = {
            "max_fwd_vel": 0.5,
            "max_hor_vel": 0.05,
            "max_ang_vel": np.deg2rad(30),
        }
    else:
        kwargs = {}

    spot.power_on()
    spot.blocking_stand()
    try:
        cmd_id = spot.set_base_position(
            x_pos=goal_x,
            y_pos=goal_y,
            yaw=goal_heading,
            end_time=100,
            **kwargs,
        )
        cmd_status = None
        while cmd_status != 1:
            time.sleep(0.1)
            feedback_resp = spot.get_cmd_feedback(cmd_id)
            cmd_status = (
                feedback_resp.feedback.synchronized_feedback.mobility_command_feedback
            ).se2_trajectory_feedback.status
        if args.dock:
            dock_start_time = time.time()
            while time.time() - dock_start_time < 2:
                try:
                    spot.dock(dock_id=DOCK_ID, home_robot=True)
                except Exception:
                    print("Dock not found... trying again")
                    time.sleep(0.1)
    finally:
        spot.power_off()


if __name__ == "__main__":
    spot = Spot("GoToWaypoint")
    with spot.get_lease(hijack=True):
        main(spot)
