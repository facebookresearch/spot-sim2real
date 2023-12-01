import numpy as np
from aria_data_utils.detector_wrappers.generic_detector_interface import GenericDetector


class ObjectInHandDetectorWrapper(GenericDetector):
    def __init__(self) -> None:
        super().__init__()

    def process_frame(self, frame: np.ndarray):
        """
        100DOH detector fires on the input image and returns:
        #TODO:
        """
        pass
