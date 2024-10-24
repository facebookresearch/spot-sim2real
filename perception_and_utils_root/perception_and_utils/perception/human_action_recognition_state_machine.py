from os import device_encoding
from typing import Any, Dict, Tuple

import numpy as np

from perception_and_utils.perception import detectron2_ho_detector
from perception_and_utils.perception.detector_wrappers.generic_detector_interface import (
    GenericDetector,
)
from perception_and_utils.utils.data_frame import DataFrame
from perception_and_utils.perception.detectron2_ho_detector import Detectron2HODetector


LHAND_CATEGORY = 0
RHAND_CATEGORY = 1
OBJECT_CATEGORY = 2


class HARStateMachine(GenericDetector):
    def __init__(self, model_path, model_config_path, verbose: bool = False):
        super().__init__()
        self.hand_object_detector = Detectron2HODetector(model_path, model_config_path)

        # state-machine setup
        self.ALL_STATES = ["holding", "not_holding"]
        self.FRAME_THRESHOLD = 10  # roughly equal to 1 second
        self._num_hand_frames = 0
        self._num_object_frames = 0
        self.state_machine = {
            "holding": self.holding_state_tick,
            "not_holding": self.not_holding_state_tick,
        }

        # initialize
        self.current_state = "not_holding"

    def holding_state_tick(self, detection_dict) -> Dict[str, Any]:
        """
        In the holding state, if we see N consecutive frames of only-hand, then we trigger
        state-change to not-holding state. Otherwise we stay in this state.
        """
        if OBJECT_CATEGORY not in detection_dict.pred_classes and (
            LHAND_CATEGORY in detection_dict.pred_classes
            or RHAND_CATEGORY in detection_dict.pred_classes
        ):
            self._num_hand_frames += 1
        if OBJECT_CATEGORY in detection_dict.pred_classes:
            self.reset_hand_count()
        if self._num_hand_frames == self.FRAME_THRESHOLD:
            self.toggle_state()
            detection_dict["action_trigger"] = "place"
            detection_dict["state"] = self.current_state
        return detection_dict

    def not_holding_state_tick(self, detection_dict) -> Dict[str, Any]:
        """
        In the not-holding state, if we see N consecutive frames of hand and object, then
        we trigger state-change to holding state. Otherwise we stay in this state.
        """
        if (
            LHAND_CATEGORY in detection_dict.pred_classes
            and OBJECT_CATEGORY in detection_dict.pred_classes
        ) or (
            RHAND_CATEGORY in detection_dict.pred_classes
            and OBJECT_CATEGORY in detection_dict.pred_classes
        ):
            self._num_object_frames += 1
        if OBJECT_CATEGORY not in detection_dict.pred_classes:
            self.reset_object_count()
        if self._num_object_frames == self.FRAME_THRESHOLD:
            self.toggle_state()
            detection_dict["action_trigger"] = "pick"
            detection_dict["state"] = self.current_state
        return detection_dict

    def reset_hand_count(self):
        self._num_hand_frames = 0

    def reset_object_count(self):
        self._num_object_frames = 0

    def toggle_state(self):
        if self.current_state not in self.ALL_STATES:
            raise ValueError(
                f"State-keeping corrupted. {self.current_state=} is not supported"
            )
        self.current_state = (
            self.ALL_STATES[0]
            if self.current_state == self.ALL_STATES[1]
            else self.ALL_STATES[1]
        )

    def process_frame(self, frame: DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        out_img, outputs = self.hand_object_detector.process_frame(frame)
        outputs = self.state_machine[self.current_state](outputs)
        return out_img, outputs
