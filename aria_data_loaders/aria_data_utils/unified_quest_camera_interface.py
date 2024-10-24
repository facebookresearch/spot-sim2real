# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp
import numpy as np

"""
This file defines the interface to for Quest3 streaming receiver API calls

The implementation for this Interface is available for Meta employees only at the moment

TODO: Define the steps to get implementation file
"""


class UnifiedQuest3CameraInterface:
    def __init__(self):
        pass

    def get_rgb(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def get_depth(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def get_rbg_focal_lengths(self) -> Optional[Tuple[float, float]]:
        raise NotImplementedError

    def get_rgb_principal_point(self) -> Optional[Tuple[float, float]]:
        raise NotImplementedError

    def get_depth_focal_lengths(self) -> Optional[Tuple[float, float]]:
        raise NotImplementedError

    def get_depth_principal_point(self) -> Optional[Tuple[float, float]]:
        raise NotImplementedError

    def get_deviceWorld_T_rgbCamera(self) -> Optional[sp.SE3]:
        raise NotImplementedError

    def get_device_T_rgbCamera(self) -> Optional[sp.SE3]:
        raise NotImplementedError

    def get_deviceWorld_T_depthCamera(self) -> Optional[sp.SE3]:
        raise NotImplementedError

    def get_device_T_depthCamera(self) -> Optional[sp.SE3]:
        raise NotImplementedError

    def get_avg_fps_rgb(self):
        raise NotImplementedError

    def get_avg_fps_depth(self):
        raise NotImplementedError

    def is_connected(self):
        raise NotImplementedError
