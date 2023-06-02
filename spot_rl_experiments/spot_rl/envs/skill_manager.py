import os
import time

from spot_wrapper.spot import Spot

from spot_rl.envs.nav_env import SpotNavEnv
from spot_rl.envs.place_env import SpotPlaceEnv
from spot_rl.envs.gaze_env import SpotGazeEnv

from spot_rl.real_policy import NavPolicy, PlacePolicy, GazePolicy

from spot_rl.utils.utils import (
    construct_config,
    get_default_parser,
    nav_target_from_waypoints,
)

DOCK_ID = int(os.environ.get("SPOT_DOCK_ID", 520))

# SpotSkillManager? SpotSkillExecutor? SpotSkillController?
class SpotSkillManager():
    def __init__(self):
        # We will probably receive a config as part of the constructor
        # And eventually that specifies the set of skills that we can instantiate and execute
        self.__init_spot()
        self.__initiate_policies()
        self.__initialize_environments()
        self.reset()

        # Power on the robot
        # TODO: Check if this can be moved outside of env
        self.nav_env.power_robot()
        #...

    # def __del__(self):
    #     # Power off the robot
    #     self.power_off()

    def __init_spot(self):
        # Create Spot object
        self.spot = Spot("RealSeqEnv")

        # Acquire spot's lease
        self.lease = self.spot.get_lease(hijack=True)
        if not self.lease:
            print("Failed to get lease for Spot. Exiting!")
            exit(1)

        # Get configs
        parser = get_default_parser()
        args = parser.parse_args()
        self.config = construct_config(args.opts) # TODO: Get config from constructor

        # TODO: The way configs are handled in the original code is a bit messy ... fix this
        # Example:
        # Don't need gripper camera for Nav
        self.config.USE_MRCNN = False

        # Don't need head cameras for Gaze
        self.config.USE_HEAD_CAMERA = False

        # Don't need cameras for Place
        # self.config.USE_HEAD_CAMERA = False
        # self.config.USE_MRCNN = False

    def __initiate_policies(self):
        # Initialize the nav, place, and pick policies (NavPolicy, PlacePolicy, GazePolicy)
        self.nav_policy = NavPolicy(self.config.WEIGHTS.NAV, device=self.config.DEVICE)
        self.pick_policy = GazePolicy(self.config.WEIGHTS.GAZE, device=self.config.DEVICE)

    def __initialize_environments(self):
        # Initialize the nav, place, and pick environments (SpotNavEnv, SpotPlaceEnv, SpotGazeEnv)
        self.nav_env = SpotNavEnv(self.config, self.spot)
        self.pick_env = SpotGazeEnv(self.config, self.spot)

    def reset(self):
        # Reset the the policies
        self.nav_policy.reset()
        self.pick_policy.reset()

    def nav(self, nav_target: str=None) -> bool:
        # use the logic of current skill to get nav_target (nav_target_from_waypoints)
        # reset the nav environment with the current target (or use the ros param)
        # run the nav policy until success
        # reset (policies and nav environment)
        if nav_target is not None:
            try:
                goal_x, goal_y, goal_heading = nav_target_from_waypoints(nav_target)
            except KeyError:
                print(f"Nav target: {nav_target} does not exist in waypoints.yaml")
                return False
        else:
            print("No nav target specified, skipping nav")
            return False

        self.nav_env.say(f"Navigating to {nav_target}")

        observations = self.nav_env.reset((goal_x, goal_y), goal_heading)
        done = False
        time.sleep(1)
        try:
            while not done:
                # Get best action using nav policy
                action = self.nav_policy.act(observations)

                # Execute action
                observations, _, done, _ = self.nav_env.step(base_action=action)
        except KeyboardInterrupt:
            print("Keyboard interrupt detected, stopping navigation")
            return False

        return True

    def pick(self, pick_target: str) -> str:
        # The current pick target is simply a string (received on the variable pick_target)
        # Reset the gaze Set the ros param so that the gaze environment looks for the pick target
        # run the gaze policy until success
        # reset (policies and gaze environment)

        # if pick_target is not None:
        #     try:
        #         goal_x, goal_y, goal_heading = nav_target_from_waypoints(nav_target)
        #     except KeyError:
        #         print(f"Nav target: {nav_target} does not exist in waypoints.yaml")
        #         return False
        # else:
        #     print("No pick target specified, skipping pick")
        #     return False
        self.pick_env.say(f"Picking to {pick_target}")

        observations = self.pick_env.reset(target_obj_id=pick_target)
        done = False
        time.sleep(1)
        try:
            while not done:
                # Get best action using pick policy
                action = self.pick_policy.act(observations)

                # Execute action
                observations, _, done, _ = self.pick_env.step(arm_action=action)
        except KeyboardInterrupt:
            print("Keyboard interrupt detected, stopping picking")
            return False

        return True

    def place(self, place_target: str) -> str:
        # use the logic of current skill to get place_target (place_target_from_waypoints)
        # reset the nav environment with the current target (or use the ros param)
        # run the nav policy until success
        # reset (policies and place environment)
        return None

    def dock(self):
        # Navigate to the dock
        self.nav('dock')

        # Reset observation????

        # Dock
        self.nav_env.say("Executing automatic docking")
        dock_start_time = time.time()
        while time.time() - dock_start_time < 2:
            try:
                self.spot.dock(dock_id=DOCK_ID, home_robot=True)
            except:
                print("Dock not found... trying again")
                time.sleep(0.1)
        return None

    def power_off(self):
        # Power off spot
        self.spot.power_off()


if __name__ == "__main__":
    spotskillmanager = SpotSkillManager()
    spotskillmanager.nav('sofa')
    # spotskillmanager.pick('ball')
    spotskillmanager.nav('coffee_table')
    # spotskillmanager.place('hall_table')
    spotskillmanager.dock()
