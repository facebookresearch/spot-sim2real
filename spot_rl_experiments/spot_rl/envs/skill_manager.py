# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import time
from typing import Tuple

import numpy as np
import rospy
from spot_rl.envs.gaze_env import GazeController, construct_config_for_gaze
from spot_rl.envs.nav_env import WaypointController, construct_config_for_nav
from spot_rl.envs.place_env import PlaceController, construct_config_for_place
from spot_rl.utils.geometry_utils import (
    is_pose_within_bounds,
    is_position_within_bounds,
)
from spot_rl.utils.utils import nav_target_from_waypoints, place_target_from_waypoints
from spot_wrapper.spot import Spot

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))


# SpotSkillManager? SpotSkillExecutor? SpotSkillController?
class SpotSkillManager:
    def __init__(self):
        # We will probably receive a config as part of the constructor
        # And eventually that specifies the set of skills that we can instantiate and execute
        self.__init_spot()
        self.__initiate_controllers()

        # Power on the robot
        # TODO: Check if this can be moved outside of env
        self.nav_controller.nav_env.power_robot()

        # TODO: Re-think this
        # self.reset()

        self.waypoints_yaml_dict = get_waypoint_yaml()
        # ...

    # TODO: What is the best thing to do here in terms of shutdown?
    # def __del__(self):
    #     self.dock()

    # #     self.power_off()

    def __init_spot(self):
        # Create Spot object
        self.spot = Spot("RealSeqEnv")

        # Acquire spot's lease
        self.lease = self.spot.get_lease(hijack=True)
        if not self.lease:
            print("Failed to get lease for Spot. Exiting!")
            exit(1)

        # Construct configs for nav, gaze, and place
        self.nav_config = construct_config_for_nav()
        self.pick_config = construct_config_for_gaze(
            max_episode_steps=35
        )  # TODO: Find a better episode cap
        self.place_config = construct_config_for_place()

    def __initiate_controllers(self):
        self.nav_controller = WaypointController(
            config=self.nav_config, spot=self.spot, should_record_trajectories=False
        )
        self.gaze_controller = GazeController(config=self.pick_config, spot=self.spot)
        self.place_controller = PlaceController(
            config=self.place_config, spot=self.spot, use_policies=True
        )

    def reset(self):
        # Reset the policies and environments via the controllers
        raise NotImplementedError

    def nav(self, nav_target: str = None) -> Tuple[bool, str]:

        if nav_target is not None:
            try:
                nav_target_list = [nav_target_from_waypoints(nav_target)]
            except Exception:
                return (
                    False,
                    f"Failed - nav target {nav_target} not found - use the exact name",
                )
        else:
            return False, "No nav target specified, skipping nav"

        self.nav_controller.say(f"Navigating to {nav_target}")

        result = None
        try:
            result = self.nav_controller.execute(nav_target_list)
        except Exception:
            return False, "Error encountered while navigating"

        # TODO: Please check if this is correct formulation of success.
        if is_pose_within_bounds(
            result[0][-1].get("pose"),
            nav_target_list[0],
            self.nav_config.SUCCESS_DISTANCE,
            self.nav_config.SUCCESS_ANGLE_DIST,
        ):
            return True, "Success"
        else:
            return False, "Navigation failed to reach the target pose"

    def pick(self, pick_target: str = None) -> Tuple[bool, str]:

        if pick_target is None:
            print("No pick target specified, skipping pick")
            return False, "No pick target specified, skipping pick"

        self.gaze_controller.say(f"Picking {pick_target}")

        result = None
        try:
            result = self.gaze_controller.execute([pick_target])
        except Exception:
            return False, "Error encountered while picking"

        if result[0].get("success"):
            return True, "Success"
        else:
            return False, "Pick failed to pick the target object"

    def place(self, place_target: str = None) -> Tuple[bool, str]:

        if place_target is not None:
            # Get the place target coordinates
            try:
                place_target_list = [place_target_from_waypoints(place_target)]
            except Exception:
                return (
                    False,
                    f"Failed - nav target {place_target} not found - use the exact name",
                )
        else:
            # TODO: We should put the arm back here in the stow position if it is not already there, otherwise docking fails XXXX?
            return False, "No place target specified, skipping place"

        self.place_controller.say(
            f"Place target object at {place_target} i.e. {place_target_list}"
        )

        result = None
        try:
            result = self.place_controller.execute(place_target_list)
        except Exception:
            return False, "Error encountered while placing"

        # TODO: RESET THE Place ENVIRONMENT HERE or inside place_controller.execute()?
        self.reset()

        # TODO: Please check if this is correct formulation of success.
        if result[0].get("success") and is_position_within_bounds(
            result[0].get("ee_pos"),
            result[0].get("place_target"),
            self.place_config.SUCC_XY_DIST,
            self.place_config.SUCC_Z_DIST,
        ):
            return True, "Success"
        else:
            return False, "Place failed to reach the target position"

    def dock(self):
        # TODO: Stow back the arm

        # Navigate to the dock
        self.nav("dock")

        # Reset observation????

        # Dock  ---- TODO: Move this method to spot.py, also rename it to `dock_and_power_off()`
        self.nav_controller.shutdown()

        # WHY RESET policies here?
        self.reset()
        return None

    # TODO: DO WE NEED THIS?
    # def power_off(self):
    #     # Power off spot
    #     self.spot.power_off()


if __name__ == "__main__":
    spotskillmanager = SpotSkillManager()
    # try:
    spotskillmanager.nav("test_receptacle")
    spotskillmanager.pick("penguin")
    spotskillmanager.place("test_receptacle")

    spotskillmanager.dock()

    # spotskillmanager.nav("living_table")
    # spotskillmanager.pick("ball")
    # spotskillmanager.place("couch")

    # spotskillmanager.nav('counter')
    # spotskillmanager.pick('ball')
    # spotskillmanager.place('chair2')

    # except KeyboardInterrupt as e:
    #     print(f"Received keyboard interrupt - {e}. Going to dock")
    # except Exception as e:
    #     print(f"Encountered exception - {e}. Going to dock")
    # finally:
    #     spotskillmanager.dock()
