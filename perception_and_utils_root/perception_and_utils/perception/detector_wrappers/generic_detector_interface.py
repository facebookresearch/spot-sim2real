# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, Tuple

import numpy as np
from perception_and_utils.utils.data_frame import DataFrame

class GenericDetector:
    def __init__(self):
        self.is_enabled = False

    def enable_detector(self):
        self.is_enabled = True

    def disable_detector(self):
        self.is_enabled = False

    def process_frame(self, frame: DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError
