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
from detectron2.utils.visualizer import Visualizer


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

    def process_frame(self, frame: DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        pass
