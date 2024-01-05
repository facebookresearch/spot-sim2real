import time
from copy import deepcopy

import cv2
import magnum as mn
import numpy as np
from spot_rl.envs.gaze_env import construct_config_for_gaze
from spot_rl.envs.nav_env import construct_config_for_nav
from spot_rl.envs.place_env import construct_config_for_place
from spot_rl.envs.skill_manager import SpotSkillManager
from spot_rl.utils.heuristic_nav import ImageSearch
from spot_rl.utils.utils import map_user_input_to_boolean
from spot_wrapper.spot import image_response_to_cv2, scale_depth_img

if __name__ == "__main__":
    run_the_loop = True  # Don't break the test loop
    is_in_position = False  # is it before the receptacle
    # Construct all three configs
    nav_config, pick_config, place_config = (
        construct_config_for_nav(),
        construct_config_for_gaze(),
        construct_config_for_place(),
    )
    # Modify any values in the config
    # nav_config.SUCCESS_DISTANCE = 0.10
    # nav_config.SUCCESS_ANGLE_DIST = 1
    place_config.SUCCESS_DISTANCE = 0.10
    pick_config.SUCCESS_DISTANCE = 0.10
    pick_config.MAX_EPISODE_STEPS = 350
    image_searcher = ImageSearch("ball")
    while run_the_loop:
        # Send the modified configs in SpotSkillManager
        spotskillmanager = SpotSkillManager(nav_config, pick_config, place_config)
        if not is_in_position:
            spotskillmanager.nav("position_of_ball")
            x, y, yaw = spotskillmanager.spot.get_xy_yaw()
            print(f"Current x,y, yaw {x, y, yaw}")

            # spotskillmanager.nav(6.735982126785333, 0.07834622761482896, 0.01598833462395932)
            spotskillmanager.spot.open_gripper()

            gaze_arm_angles = deepcopy(
                spotskillmanager.pick_config.GAZE_ARM_JOINT_ANGLES
            )

            spotskillmanager.spot.set_arm_joint_positions(np.deg2rad(gaze_arm_angles))
            base_shoulder_position = spotskillmanager.spot.get_arm_joint_positions()[0]

            rate = 20
            # print(np.arange(-90, 110, 20))
            for ai, angle in enumerate(np.arange(-90, 110, 20)):
                print(f"Searching in {angle}")
                angle_time = int(np.abs(gaze_arm_angles[0] - angle) / rate)
                gaze_arm_angles[0] = angle
                spotskillmanager.spot.set_arm_joint_positions(
                    np.deg2rad(gaze_arm_angles), angle_time
                )
                imgs = spotskillmanager.spot.get_hand_image()
                rgb_img = image_response_to_cv2(imgs[0])
                unscaled_dep_img = image_response_to_cv2(imgs[1])
                dep_img = scale_depth_img(
                    unscaled_dep_img,
                    max_depth=unscaled_dep_img.max() * 0.001,
                    as_img=False,
                )
                intrinsics = imgs[0].source.pinhole.intrinsics
                vision_T_hand: mn.Matrix4 = spotskillmanager.spot.get_spot_a_T_b(
                    "vision",
                    "hand_color_image_sensor",
                    imgs[0].shot.transforms_snapshot,
                )
                body_T_hand: mn.Matrix4 = spotskillmanager.spot.get_spot_a_T_b(
                    "body", "hand_color_image_sensor", imgs[0].shot.transforms_snapshot
                )
                found, (x, y, theta), rgb_img = image_searcher.is_object_found(
                    rgb_img,
                    unscaled_dep_img,
                    dep_img,
                    vision_T_hand,
                    body_T_hand,
                    intrinsics,
                    spotskillmanager.spot,
                )
                # print(rgb_img.dtype, rgb_img.shape, rgb_img.max(), rgb_img.min())
                cv2.imwrite(f"imagesearch_{angle}.png", rgb_img)
                if found:
                    print(f"found object at {(x,y,theta)}")
                    angle_time = int(
                        np.abs(
                            gaze_arm_angles[0]
                            - spotskillmanager.pick_config.GAZE_ARM_JOINT_ANGLES[0]
                        )
                        / rate
                    )
                    spotskillmanager.spot.set_arm_joint_positions(
                        np.deg2rad(spotskillmanager.pick_config.GAZE_ARM_JOINT_ANGLES),
                        angle_time,
                    )
                    break
                time.sleep(0.5)
            spotskillmanager.nav(x, y, theta)
            exit()

            # enter static gaze
        # Reset Arm before gazing starts
        spotskillmanager.get_env().reset_arm()
        # spotskillmanager.gaze_controller.reset_env_and_policy("ball")
        # pick_stats = spotskillmanager.pick("ball")
        # # print(pick_stats)
        # spotskillmanager.get_env().reset_arm()
        # spotskillmanager.gaze_controller.reset_env_and_policy("ball")
        # is_in_position = False
        # # Navigate to Test Receotacle
        # spotskillmanager.nav("pick_table_05_45")
        is_in_position = True

        run_the_loop = map_user_input_to_boolean(
            "Do you want to continue to next test or dock & exit ?"
        )
        if not run_the_loop:
            spotskillmanager.nav_controller.nav_env._enable_nav_goal_change = False
            # spotskillmanager.nav("temp_home")
            spotskillmanager.dock()
