from typing import Optional

import numpy as np

try:
    import sophuspy as sp
except Exception as e:
    print(f"Cannot import sophuspy due to {e}. Import sophus instead")
    import sophus as sp


class DataFrame:
    """
    DataFrame class to store instantaneous data from Human Sensor stream.
    It stores the image, timestamp and frame number of each received image
    """

    def __init__(self):
        self._frame_number = -1
        self._timestamp_s = -1
        self._avg_rgb_fps = None  # type: Optional[float]
        self._avg_depth_fps = None  # type: Optional[float]
        self._avg_data_frame_fps = None  # type: Optional[float]
        self._rgb_frame = None  # type: Optional[np.ndarray]
        self._depth_frame = None  # type: Optional[np.ndarray]
        self._aligned_depth_frame = None  # type: Optional[np.ndarray]
        self._deviceWorld_T_camera_rgb = None  # type: Optional[sp.SE3]
        self._deviceWorld_T_camera_depth = None  # type: Optional[sp.SE3]
        self._device_T_camera_rgb = None  # type: Optional[sp.SE3]
        self._device_T_camera_depth = None  # type: Optional[sp.SE3]
