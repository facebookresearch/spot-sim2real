import logging
from typing import Dict, Any, Tuple

import torch
import detectron2
import numpy as np
import os
import json
import cv2
import random

from perception_and_utils.detector_wrappers.generic_detector import GenericDetector
from perception_and_utils.utils.data_frame import DataFrame


# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode


class Detectron2HODetector(GenericDetector):
    def __init__(self, model_path, model_config_path) -> None:
        super().__init__()
        self.cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg.merge_from_file(model_config_path)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = model_path
        self.predictor = DefaultPredictor(self.cfg)
        self.hod_metadata = {
            "thing_classes": ["lhand", "rhand", "object"],
            "thing_colors": [[220, 20, 60], [119, 11, 32], [0, 0, 142]],
        }
        self._logger = logging.getLogger(__name__)
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        self._logger.setLevel(logging.DEBUG)
        self._logger.info("Initialized")

    def process_frame(self, frame: DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        out_image = np.zero(256, 256, 3)
        output_dict = {}
        if not self.is_enabled:
            self._logger.warning(
                "Detector has not been enabled. Returning empty output. Run enable_detector() before trying to process the frame."
            )
            return out_image, output_dict
        input_image = frame._rgb_frame
        output_dict = self._predictor(input_image)
        output_dict["instances"].detach().cpu()
        v = Visualizer(
            input_image[:, :, ::-1],
            self.hod_metadata,
            scale=1.2,
            instance_mode=ColorMode.SEGMENTATION,
        )
        out_image = v.draw_instance_predictions(output_dict["instances"].to("cpu"))
        return out_image, output_dict
