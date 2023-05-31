from spot_wrapper.spot import Spot


# SpotSkillManager? SpotSkillExecutor? SpotSkillController?
class SpotSkillManager():
    def __init__(self, ):
        self.spot = Spot("RealSeqEnv")
        self.__init_spot()
        self.__initiate_policies()
        self.__initialize_environments()
        self.reset()
        #...

    def __init_spot(self):
        # Run anything to start spot like lease, or power_robot

    def __initiate_policies(self):
        # Initialize the nav, place, and pick policies (NavPolicy, PlacePolicy, SpoGazePolicy)

    def __initialize_environments(self):
        # Initialize the nav, place, and pick environments (SpotNavEnv, SpotPlaceEnv, SpotGazeEnv)

    def reset(self):
        # Reset the the policies
    
    def nav(self, nav_target):
        # use the logic of current skill to get nav_target (nav_target_from_waypoints)
        # reset the nav environment with the current target (or use the ros param)
        # run the nav policy until success
        # reset (policies and nav environment)

    def place(self, place_target):
        # use the logic of current skill to get place_target (place_target_from_waypoints)
        # reset the nav environment with the current target (or use the ros param)
        # run the nav policy until success
        # reset (policies and place environment)

    def pick(self, pick_target):
        # The current pick target is simply a string (received on the variable pick_target)
        # Reset the gaze Set the ros param so that the gaze environment looks for the pick target
        # run the gaze policy until success
        # reset (policies and gaze environment)

    def dock(self):
        # Dock
    
    def power_off(self):
        # Power off spot


if __name__ == "__main__":
    spot = SpotSkillController()
    spot.nav('kitchen_counter')
    spot.pick('ball')
    spot.nav('hall_table')
    spot.place('hall_table')
    spot.dock()
