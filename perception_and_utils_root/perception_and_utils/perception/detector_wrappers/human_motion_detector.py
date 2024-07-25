# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from typing import Any, Dict, Tuple

import numpy as np
import sophus as sp
from perception_and_utils.perception.detector_wrappers.generic_detector_interface import (
    GenericDetector,
)
from perception_and_utils.utils.image_utils import label_img
from perception_and_utils.utils.data_frame import DataFrame

class HumanMotionDetector(GenericDetector):
    """
    """

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger("HumanMotionDetector")

    def _init_human_motion_detector(self, velocity_threshold=0.75, time_horizon_ns=1000000000):
        """
        Initialize the human motion detector
        """
        self._logger.info("Initializing Human Motion Detector")
        self.enable_detector()
        self._velocity_threshold = velocity_threshold

        self._time_horizon_ns = time_horizon_ns # 1 second
        self._previous_frames = []
        self._len_prev_frame_cache = 100 # Keep this value larger than fps

    def process_frame(
        self,
        frame: DataFrame,
    ) -> Tuple[np.ndarray, str]:
        # Do nothing if detector is not enabled
        if self.is_enabled is False:
            self._logger.warning(
                "Human Motion detector is disabled... Skipping processing of current frame."
            )
            return None, "Disabled"

        # Calculate the current position from the transformation matrix
        current_position = frame._deviceWorld_T_camera_rgb.matrix()[:3, 3]

        # Add the current frame to the cache
        self._previous_frames.append((current_position, frame._timestamp_s))

        if len(self._previous_frames) < self._len_prev_frame_cache:
            return None, "Standing"

        # If the cache is full, remove the oldest frame
        if len(self._previous_frames) > self._len_prev_frame_cache:
            self._previous_frames.pop(0)

        # Find an index in "_previous_positions" which is 1 sec before current value
        index = len(self._previous_frames) - 60
        # If the index is out of range, return "Standing"
        if index < 0:
            return None, "Standing"

        # Calculate the Euclidean distance between now & then
        print(f"Current position: {current_position}   |   Previous position: {self._previous_frames[index][0]}")
        distance = np.linalg.norm(current_position - self._previous_frames[index][0])
        time = frame._timestamp_s - self._previous_frames[index][1] # as we are considering x frames per second so our window is 1 sec
        print(f"Distance: {distance} m   |   Time: {time} sec")
        avg_velocity = distance / time

        print(f"Avg velocity: {avg_velocity} m/s")
        # Determine the activity based on the velocity threshold
        if avg_velocity > self._velocity_threshold:
            return None, "Walking"

        return None, "Standing"

    def get_outputs(
        self,
        img_frame: np.ndarray,
        outputs: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """
        """
        # Decorate image with text for visualization
        label_img(
            img=img_frame,
            text=f"Activity: {outputs['activity']}",
            org=(50, 200),
        )
        return img_frame, outputs
