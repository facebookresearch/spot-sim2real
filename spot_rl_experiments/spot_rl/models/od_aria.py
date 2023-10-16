import argparse
import time
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt
from owlvit import OwlVit
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId


class AriaDataLoader:
    def __init__(self, vrsfile_path, closed_loop_trajectory_path):
        self._vrsfile = vrsfile_path
        self._traj_dile = closed_loop_trajectory_path
        self.rgb_stream_id = StreamId("214-1")

        self.data_provider = data_provider.create_vrs_data_provider(vrsfile_path)
        self.mps_trajectory = mps.read_closed_loop_trajectory(
            closed_loop_trajectory_path
        )
        self.xyz_trajectory = np.empty([len(self.mps_trajectory), 3])
        self.quat_trajectory = np.empty([len(self.mps_trajectory), 4])
        self.trajectory_ns = np.empty([len(self.mps_trajectory)])
        self.initialize_trajectory()

    def initialize_trajectory(self):
        for i in range(len(self.mps_trajectory)):
            self.xyz_trajectory[i, :] = self.mps_trajectory[
                i
            ].transform_world_device.translation()
            self.quat_trajectory[i, :] = self.mps_trajectory[
                i
            ].transform_world_device.quaternion()
            self.trajectory_ns[i] = self.mps_trajectory[
                i
            ].tracking_timestamp.total_seconds()

    def plot_trajectory(self, markers: Optional[np.ndarray] = None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            self.xyz_trajectory[:, 0],
            self.xyz_trajectory[:, 1],
            self.xyz_trajectory[:, 2],
        )
        if markers is not None:
            ax.plot(markers[:, 0], markers[:, 1], markers[:, 2], "ro", markersize=12)
        plt.show()

    def plot_rgb_and_trajectory(self, timestamp=None):
        rgb = self.data_provider.get_image_data_by_time_ns(
            self.rgb_stream_id,
            int(timestamp * 1e9),
            TimeDomain.DEVICE_TIME,
            TimeQueryOptions.CLOSEST,
        )[0].to_numpy_array()
        plt.imshow(rgb)
        marker_idx = np.argmin(np.abs(self.trajectory_ns - timestamp))
        markers = self.xyz_trajectory[marker_idx, :].reshape(1, 3)
        self.plot_trajectory(markers=markers)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--file",
    type=str,
    default="/home/priparashar/Documents/aria_recordings_transient/09_12_2023/QrCodeTest00.vrs",
)
parser.add_argument("--score_threshold", type=float, default=0.1)
parser.add_argument("--show_img", type=bool, default=True)
parser.add_argument(
    "--labels",
    type=List[str],
    default=[
        [
            "indoor plant",
            "stained glass",
            "doorframe",
        ]
    ],
)
args = parser.parse_args()

vrs_file = args.file
mps_file = "/home/priparashar/Documents/aria_recordings_transient/09_12_2023/outputs/closed_loop_trajectory.csv"
aria = AriaDataLoader(vrs_file, mps_file)

V = OwlVit(args.labels, args.score_threshold, args.show_img)
img = aria.data_provider.get_image_data_by_time_ns(
    aria.rgb_stream_id,
    int(25000 * 1e9),
    TimeDomain.DEVICE_TIME,
    TimeQueryOptions.CLOSEST,
)[0].to_numpy_array()
results = V.run_inference(img)
# Keep the window open for 10 seconds
time.sleep(10)
