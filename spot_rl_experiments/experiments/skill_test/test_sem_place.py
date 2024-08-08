# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import rospy
from perception_and_utils.utils.generic_utils import map_user_input_to_boolean
from spot_rl.envs.skill_manager import SpotSkillManager

if __name__ == "__main__":
    # Know which location we are doing experiments
    in_fre_lab = map_user_input_to_boolean("Are you Tushar in FRE? Y/N ")
    enable_estimation_before_place = map_user_input_to_boolean(
        "Enable estimation before place? Y/N "
    )

    if in_fre_lab:
        # at FRE
        place_target = "kitchen_right_inside"
    else:
        # at NYC
        place_target = "test_desk"

    spotskillmanager = SpotSkillManager(use_mobile_pick=False, use_semantic_place=True)

    is_local = False
    if enable_estimation_before_place:
        # place_target = None
        is_local = True

    # Start testing
    contnue = True
    while contnue:
        rospy.set_param("is_gripper_blocked", 0)
        # spotskillmanager.pick("bottle")
        # spotskillmanager.nav(place_target)
        spotskillmanager.place(
            None if enable_estimation_before_place else place_target,
            is_local=is_local,
            visualize=True,
        )
        # spotskillmanager.contrainedplace("bowl", is_local=is_local, proposition="left")
        contnue = map_user_input_to_boolean("Do you want to do it again ? Y/N ")

# The following is a helpful tip to debug the arm
# We get Spot class
# spot = spotskillmanager.spot
# We can move the gripper to a point with x,y,z and roll, pitch, yaw
# spot.move_gripper_to_point((0.55, 0., 0.26), np.deg2rad(np.array([0,0,0])))
# We can also set the robot arm joints
# config = construct_config()
# spot.set_arm_joint_positions(np.deg2rad(config.INITIAL_ARM_JOINT_ANGLES))

# In addition, if you want to use semantic place skill based on the grasping orientation, you can do
# spotskillmanager.nav("black_case")
# spotskillmanager.pick("bottle")
# # Fetch the arm joint at grasping location
# ee_orientation_at_grasping = spotskillmanager.gaze_controller.env.ee_orientation_at_grasping
# spotskillmanager.nav("test_desk")
# spotskillmanager.place("test_desk", orientation_at_grasping) # This controls the arm initial orientation
