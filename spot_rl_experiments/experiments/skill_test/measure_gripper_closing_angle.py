import time

import numpy as np
from perception_and_utils.utils.generic_utils import map_user_input_to_boolean
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_wrapper.spot import Spot

if __name__ == "__main__":
    spot: SpotSkillManager = SpotSkillManager()
    # Check the openness of the gripper
    # This value is between 0 (close) and 100 (open)
    run = True
    closing_angle_samples = []
    spot.spot.open_gripper()
    time.sleep(1.0)
    spot.spot.close_gripper()
    time.sleep(1.0)
    while True:
        if run:
            _gripper_open_percentage = (
                spot.spot.robot_state_client.get_robot_state().manipulator_state.gripper_open_percentage
            )
            closing_angle_samples.append(_gripper_open_percentage)
            print(f"Gripper Open Percentage {_gripper_open_percentage}")

        if len(closing_angle_samples) % 10 == 0:
            print(
                "Max:",
                max(closing_angle_samples),
                "Mean",
                np.mean(closing_angle_samples),
                "min:",
                min(closing_angle_samples),
            )
        run = map_user_input_to_boolean("Do it again ?y/n")
        spot.spot.open_gripper()
        time.sleep(1.0)
        spot.spot.close_gripper()
        time.sleep(1.0)
