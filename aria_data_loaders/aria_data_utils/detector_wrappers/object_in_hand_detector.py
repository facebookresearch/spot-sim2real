import os

import numpy as np
from aria_data_utils.detector_wrappers.generic_detector_interface import GenericDetector
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16


class ObjectInHandDetectorWrapper(GenericDetector):
    def __init__(self) -> None:
        super().__init__()
        # self._pkg_path = os.path.dirname(a_module.__file__)

    def process_frame(self, frame: np.ndarray):
        """
        100DOH detector fires on the input image and returns:
        #TODO:
        """
        pass


if __name__ == "__main__":
    obj = ObjectInHandDetectorWrapper()
