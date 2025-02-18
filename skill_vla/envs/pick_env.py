# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import sys
import time
from typing import Any, Dict

import magnum as mn
import numpy as np
import rospy
import cv2
from envs.base_env import SpotBaseEnv, pad_action, rescale_actions
from spot_wrapper.spot import Spot, wrap_heading


class SpotPickEnv(SpotBaseEnv):
    def __init__(self, config, spot: Spot):
        super().__init__(
            config,
            spot,
        )
        self.grasp_attempted = False
        self.initial_arm_joint_angles = np.deg2rad(self.config.PICK_ARM_JOINT_ANGLES)

    def reset(self, *args, **kwargs):
        # Move arm to initial configuration
        cmd_id = self.spot.set_arm_joint_positions(
            positions=self.initial_arm_joint_angles, travel_time=1
        )

        # Block until arm arrives with incremental timeout for 3 attempts
        timeout_sec = 1.0
        max_allowed_timeout_sec = 3.0
        status = False
        print("Opening gripper for pick")
        self.spot.open_gripper()
        while status is False and timeout_sec <= max_allowed_timeout_sec:
            status = self.spot.block_until_arm_arrives(cmd_id, timeout_sec=timeout_sec)
            timeout_sec += 1.0

        # Update target object name as provided in config
        observations = self.get_observations()
        rospy.set_param("is_gripper_blocked", 0)
        self.grasp_attempted = False
        return observations

    def rescale_depth(self, depth_image, max_depth=10.0):
        return (depth_image / 255.0)*max_depth
    
    def find_smallest_value_pixel_in_largest_contour(self, depth_image, threshold_min, threshold_max):
        print('depth image min max: ', np.min(depth_image), np.max(depth_image), np.mean(depth_image))
        # Apply threshold to the depth image (get pixels in the range)
        thresholded = cv2.inRange(depth_image, threshold_min, threshold_max)

        # Find contours of the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            print("No contours found.")
            return None
        print('contours: ', len(contours))
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the pixels within the largest contour
        contour_pixels = []
        for point in largest_contour:
            x, y = point[0]  # Extract x, y from contour points
            contour_pixels.append((x, y))

        # Find the pixel with the smallest depth value within these contour pixels
        min_depth_value = float('inf')
        min_pixel = None

        for x, y in contour_pixels:
            depth_value = depth_image[y, x]
            if depth_value < min_depth_value:
                min_depth_value = depth_value
                min_pixel = [x, y]

        if min_pixel is None:
            print("No pixels found in the contour.")
            return None
        
        # Optionally, mark the smallest value pixel on the image for visualization
        depth_image_with_min_pixel = (depth_image.copy() / 3.0)*255
        cv2.circle(depth_image_with_min_pixel, min_pixel, 5, (0, 0, 255), -1)  # Red circle at the smallest depth pixel
        cv2.imwrite(f"imgs/filtered_hand_depth_before_{int(time.time()*10000)}.png", depth_image_with_min_pixel)
        return min_pixel

    def find_largest_contour_center(self, depth_image, threshold_min, threshold_max):
        print('depth image min max: ', np.min(depth_image), np.max(depth_image), np.mean(depth_image))
        # Apply threshold to the depth image (get pixels in the range)
        thresholded = cv2.inRange(depth_image, threshold_min, threshold_max)

        # Find contours of the thresholded image
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            print("No contours found.")
            return None
        
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the center of the largest contour (centroid)
        moments = cv2.moments(largest_contour)
        
        if moments["m00"] == 0:
            print("Contour area is zero, unable to find centroid.")
            return None

        # Calculate the centroid of the largest contour
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
        
        # Optionally, mark the center and contour on the image for visualization
        depth_image_with_contour = (depth_image.copy() / 3.0)*255
        cv2.drawContours(depth_image_with_contour, [largest_contour], -1, (0, 255, 0), 2)  # Draw contour in green
        cv2.circle(depth_image_with_contour, (center_x, center_y), 5, (0, 0, 255), -1)  # Red circle at the center
        cv2.imwrite(f"imgs/filtered_hand_depth_before_{int(time.time()*10000)}.png", depth_image_with_contour)
        min_pixel = [center_x, center_y]
        return min_pixel
    
    def step(self, action_dict: Dict[str, Any]):
        base_action, arm_action = super().pre_step(
            action_dict=action_dict,
        )
        grasp_action = action_dict["grasp_action"]
        observations = self.get_observations()
        rescaled_depth = self.rescale_depth(observations["arm_depth"], max_depth=3.0)
        min_pixel = self.find_smallest_value_pixel_in_largest_contour(
            rescaled_depth, 0.7, 1.0
        )
        print('min_pixel: ', min_pixel)
        time.sleep(1)
        
        # if grasp_action == 1 or 
        if min_pixel is not None:
            h, w = observations['arm_rgb'].shape[:2]
            min_pixel = [h//2, w//2]
            self.attempt_grasp(min_pixel)

            # get center of image
            while not self.grasp_success:
                # observations = self.get_observations()
                # rescaled_depth = self.rescale_depth(observations["arm_depth"], max_depth=3.0)
                # new_min_pixel = self.find_smallest_value_pixel_in_largest_contour(
                #     rescaled_depth, 0.7, 1.0
                # )
                # if new_min_pixel is not None:
                #     min_pixel = new_min_pixel
                print('min_pixel: ', min_pixel)
                self.attempt_grasp(min_pixel)

            print('grasp_success: ', self.grasp_success)
        elif base_action is None:
            self.spot.set_arm_joint_positions(
                positions=arm_action,
                travel_time=1 / self.ctrl_hz * 0.9,
            )
        elif arm_action is None:
            self.spot.set_base_velocity(
                *base_action,
                5,
                disable_obstacle_avoidance=False,
            )
        else:
            self.spot.set_base_vel_and_arm_pos(
                *base_action,
                arm_action,
                travel_time=self.config.ARM_TRAJECTORY_TIME_IN_SECONDS,
                disable_obstacle_avoidance=self.config.DISABLE_OBSTACLE_AVOIDANCE,
            )

        observations, reward, done, info = super().post_step()
        return observations, reward, done, info

    def get_observations(self):
        observations = {
            "joint": self.get_arm_joints(),
            "arm_rgb": self.get_gripper_images("filtered_hand_rgb"),
            "arm_depth": self.get_gripper_images("filtered_hand_depth"),
        }

        return observations

    def get_success(self, observations):
        return self.grasp_success
