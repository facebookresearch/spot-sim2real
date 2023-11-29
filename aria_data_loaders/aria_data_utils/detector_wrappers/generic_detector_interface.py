from typing import Any, Dict

import numpy as np


class GenericDetector:
    def __init__(self):
        self.is_enabled = False

    def enable_detector(self):
        self.is_enabled = True

    def disable_detector(self):
        self.is_enabled = False

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError
