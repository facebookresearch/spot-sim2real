import os

import model as hand_object_detector
import numpy as np
import torch
from aria_data_utils.detector_wrappers.generic_detector_interface import GenericDetector
from aria_data_utils.perception.object_in_hand_detector import setup_network
from model.faster_rcnn.faster_rcnn import _fasterRCNN


class ObjectInHandDetectorWrapper(GenericDetector):
    def __init__(self, net="res101", class_agnostic=False) -> None:
        super().__init__()
        self._model_module_dir = os.path.abspath(hand_object_detector)
        self._model_path = os.path.join(
            self._model_module_dir,
            "models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth",
        )
        self._net = net
        self._class_agnostic = class_agnostic
        self._POOLING_MODE = None

        self.fasterrcnn: _fasterRCNN = None
        if not os.path.exists(self._model_path):
            raise Exception("There is no model file at " + self._model_path)
        pascal_classes = np.asarray(["__background__", "targetobject", "hand"])
        set_cfgs = ["ANCHOR_SCALES", "[8, 16, 32, 64]", "ANCHOR_RATIOS", "[0.5, 1, 2]"]

        # setup self.fasterRCNN network
        self.fasterRCNN = self._setup_network(pascal_classes, set_cfgs)

    def _setup_network(self, pascal_classes, set_cfgs):
        return setup_network(
            pascal_classes=pascal_classes,
            cfgs=set_cfgs,
            model_path=self._model_path,
            net=self._net,
            class_agnostic=self._class_agnostic,
        )

    def process_frame(self, frame: np.ndarray):
        """
        100DOH detector fires on the input image and returns:
        #TODO:
        """
        pass


if __name__ == "__main__":
    obj = ObjectInHandDetectorWrapper()
