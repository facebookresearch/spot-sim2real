# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp

from perception_and_utils.perception.detector_wrappers.generic_detector_interface import (
    GenericDetector,
)
from perception_and_utils.utils.data_frame import DataFrame
from perception_and_utils.utils.image_utils import decorate_img_with_fps, label_img


class HumanMotionDetector(GenericDetector):
    """ """

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger("HumanMotionDetector")

    def _init_human_motion_detector(
        self,
        # velocity_threshold=1.2,
        velocity_threshold=0.4,  # hacking this
        time_horizon_ns=1000000000,
    ):
        """
        Initialize the human motion detector

        Args:
            velocity_threshold (float, optional): Velocity threshold for determining walking vs standing. Defaults to 1.2 metres/sec
            time_horizon_ns (int, optional): Time horizon for calculating velocity. Defaults to 1 second (in nanosec)
        """
        self._logger.info("Initializing Human Motion Detector")
        self.enable_detector()
        self._velocity_threshold = velocity_threshold

        self._time_horizon_ns = time_horizon_ns  # 1 second
        self._previous_frames = []  # type: List[Tuple[np.ndarray, float]]
        self._len_prev_frame_cache = 100  # Keep this value larger than fps

        self._human_motion_history = []  # type: List[Tuple[float, str]]

    def process_frame(
        self,
        frame: DataFrame,  # new data frame
    ) -> Tuple[str, float]:
        # Do nothing if detector is not enabled
        if self.is_enabled is False:
            self._logger.warning(
                "Human Motion detector is disabled... Skipping processing of current frame."
            )
            return "Disabled", None

        # Calculate the current position from the transformation matrix
        current_position = frame._deviceWorld_T_camera_rgb.matrix()[:3, 3]

        # Add the current frame to the cache
        self._previous_frames.append((current_position, frame._timestamp_s))

        # If the cache is just 1 element or less, return "Standing"
        lookback_index = 10
        if len(self._previous_frames) < lookback_index + 1:
            return "Standing", None

        # If the cache is full, remove the oldest frame
        if len(self._previous_frames) > self._len_prev_frame_cache:
            self._previous_frames.pop(0)

        # Calculate the Euclidean distance between now & then
        # print(
        #     f"Current position: {current_position} |Previous position: {self._previous_frames[lookback_index][0]}"
        # )
        distance = np.linalg.norm(
            current_position - self._previous_frames[lookback_index][0]
        )
        time = (
            frame._timestamp_s - self._previous_frames[lookback_index][1]
        )  # as we are considering x frames per second so our window is 1 sec
        # print(f"Distance: {distance} m |Time: {time} sec")
        avg_velocity = distance / time

        # print(f"Avg velocity: {avg_velocity} m/s")
        state = "Standing"
        # Determine the activity based on the velocity threshold
        if avg_velocity > self._velocity_threshold:
            state = "Walking"

        self.update_human_motion_history(state, frame._timestamp_s)
        return state, avg_velocity

    def get_outputs(
        self,
        img_frame: np.ndarray,
        outputs: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """ """
        viz_img = img_frame.copy()
        # Decorate image with text for visualization
        label_img(
            img=viz_img,
            text=f"Activity: {outputs['activity']}",
            org=(50, 200),
        )
        label_img(
            img=viz_img,
            text=f"Velocity: {outputs['velocity']}",
            org=(50, 250),
        )
        decorate_img_with_fps(viz_img, outputs["data_frame_fps"], pos=(50, 400))
        return viz_img, outputs

    def update_human_motion_history(self, state, timestamp):
        # If current state of human is different from last state in history list, only then update history
        if (
            len(self._human_motion_history) == 0
            or self._human_motion_history[-1][1] != state
        ):
            self._human_motion_history.append((timestamp, state))

    def get_human_motion_history(self):
        return self._human_motion_history
