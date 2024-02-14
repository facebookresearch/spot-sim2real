import os
import time
from copy import deepcopy
from glob import glob

import cv2
import magnum as mn
import numpy as np
from spot_rl.utils.heuristic_nav import (
    ImageSearch,
    get_arguments_for_image_search,
    pull_back_point_along_theta_by_offset,
)
from spot_wrapper.spot import Spot


def heurisitic_object_search(
    x: float,
    y: float,
    theta: float,
    object_target: str,
    image_search: ImageSearch = None,
    save_cone_search_images: bool = True,
    pull_back: bool = True,
    skillmanager=None,
    angle_start=-90,
    angle_end=110,
    angle_interval=20,
):
    """
    Args:
            x (float): x coordinate of the nav target (in meters) specified in the world frame
            y (float): y coordinate of the nav target (in meters) specified in the world frame
            theta (float): yaw for the nav target (in radians) specified in the world frame
            object_target: str object to search
            image_search : spot_rl.utils.heuristic_nav.ImageSearch, Optional, default=None, ImageSearch (object detector wrapper), if none creates a new one for you uses OwlVit
            save_cone_search_images: bool, optional, default= True, saves image with detections in each search cone
            pull_back : bool, optional, default=True, pulls back x,y along theta direction
            skill_manager: skill_manager object to perform low level skills
        Returns:
            bool: True if navigation was successful, False otherwise, if True you are good to fire .pick metho

    """
    if save_cone_search_images:
        previously_saved_images = glob("imagesearch*.png")
        for f in previously_saved_images:
            os.remove(f)

    if image_search is None:
        image_search = ImageSearch(
            corner_static_offset=0.5,
            use_yolov8=False,
            visualize=save_cone_search_images,
            version=2,
        )

    (x, y) = (
        pull_back_point_along_theta_by_offset(x, y, theta, 0.2) if pull_back else (x, y)
    )
    # print(f"Nav targets adjusted on the theta direction ray {x, y, np.degrees(theta)}")
    # skillmanager.nav(x, y, theta)
    # skillmanager.nav_controller.nav_env.enable_nav_by_hand()

    spot: Spot = skillmanager.spot
    spot.open_gripper()
    gaze_arm_angles = deepcopy(skillmanager.pick_config.GAZE_ARM_JOINT_ANGLES)
    spot.set_arm_joint_positions(np.deg2rad(gaze_arm_angles), 1)
    time.sleep(1.5)
    found, (x, y, theta), point3d_in_vision, visulize_img = image_search.search(
        object_target, *get_arguments_for_image_search(spot)
    )
    rate = angle_interval  # control time taken to rotate the arm, higher the rotation higher is the time
    if not found:
        # start semi circle search
        semicircle_range = np.arange(angle_start, angle_end, angle_interval)
        for i_a, angle in enumerate(semicircle_range):
            print(f"Searching in {angle} cone")
            angle_time = int(np.abs(gaze_arm_angles[0] - angle) / rate)
            gaze_arm_angles[0] = angle
            spot.set_arm_joint_positions(np.deg2rad(gaze_arm_angles), angle_time)
            time.sleep(1.5)
            (
                found,
                (x, y, theta),
                point3d_in_vision,
                visulize_img,
            ) = image_search.search(  # type : ignore
                object_target, *get_arguments_for_image_search(spot)
            )
            if save_cone_search_images:
                cv2.imwrite(f"imagesearch_{angle}.png", visulize_img)
            if found:
                print(f"In Cone Search object found at {(x,y,theta)}")
                break

    else:
        if save_cone_search_images:
            cv2.imwrite("imagesearch_looking_forward.png", visulize_img)
    angle_time = int(
        np.abs(gaze_arm_angles[0] - skillmanager.pick_config.GAZE_ARM_JOINT_ANGLES[0])
        / rate
    )
    spot.set_arm_joint_positions(
        np.deg2rad(skillmanager.pick_config.GAZE_ARM_JOINT_ANGLES),
        angle_time,
    )
    if found:
        print(f"Nav goal after cone search {x, y, np.degrees(theta)}")
        point_in_home = np.array([x, y, point3d_in_vision[-1]])
        # backup_steps = skillmanager.nav_controller.nav_env.max_episode_steps
        # skillmanager.nav_controller.nav_env.max_episode_steps = 50
        # skillmanager.nav(x, y, theta)
        # skillmanager.nav_controller.nav_env.max_episode_steps = backup_steps
    # skillmanager.nav_controller.nav_env.disable_nav_by_hand()
    return found, point_in_home, (x, y, theta)
