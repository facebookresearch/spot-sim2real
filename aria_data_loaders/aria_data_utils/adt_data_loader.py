import glob
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import yaml
from projectaria_tools.core import calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPathsProvider,
    AriaDigitalTwinDataProvider,
    bbox2d_to_image_coordinates,
)
from projectaria_tools.projects.adt import utils as adt_utils
from tqdm import tqdm

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class ADTSequences:
    """Class to load ADT sequences from a root directory. This class is meant to
    be used as the top-level interface for working with ADT. It is meant to
    use ADTSubsequence as a wrapper class for the actual data.
    Variable members:
        - data_root: path to top-level folder containing ADT sequences
        - verbose: whether to print out debug messages
        - device_data: list of lists of paths to ADT sequences, each list contains
                        paths to sequences that are part of the same ADT sequence
        - multi_device_sequences: list of indices of sequences that contain
                                    data from multiple devices
        - single_device_sequences: list of indices of sequences that contain
                                    data from a single device
        - current_sequence: ADTSubSequence object for the current sequence
    """

    def __init__(self, data_root, verbose=False, is_path=False) -> None:
        """Initializes ADTSequences class, data_root can be either a path to a
        top-level folder containing ADT sequences or path to a specific ADT
        Sequence (is_path should be set to True in this case)
        """
        self.data_root = data_root
        self.verbose = verbose
        self.device_data = []
        self._get_sequences(is_path=is_path)
        self._do_indexing()

    def _sanitize_paths(self) -> None:
        """deletes any files from globbed paths"""
        for path in self.file_paths:
            if os.path.isfile(path):
                if self.verbose:
                    logging.debug(f"Removing file {path}")
                self.file_paths.remove(path)

    def _get_sequences(self, is_path=False) -> None:
        """Loads all ADT sequences from data_root, if is_path is True, then
        data_root is a path to a specific ADT sequence"""
        if not is_path:
            self.file_paths = glob.glob(
                os.path.join(self.data_root, "**"), recursive=False
            )
        else:
            self.file_paths = [self.data_root]
        self._sanitize_paths()
        if self.verbose:
            print(f"Found {len(self.file_paths)} folders")
        for i, path in enumerate(self.file_paths):
            path_provider = AriaDigitalTwinDataPathsProvider(path)
            devices = path_provider.get_device_serial_numbers()
            self.device_data.append([])
            for idx, device in enumerate(devices):
                paths = path_provider.get_datapaths_by_device_num(idx)
                self.device_data[i].append(paths)
                if self.verbose:
                    print("Device ID", idx)
                    print("Paths:\n", paths)
        if self.verbose:
            print(f"Loaded data-paths from {len(self.device_data)} sequences")

    def _do_indexing(self) -> None:
        """Creates lists of indices for sequences that contain data from multiple
        devices and sequences that contain data from a single device"""
        self.multi_device_sequences = []
        self.single_device_sequences = []
        for i, data in enumerate(self.device_data):
            if len(data) > 1:
                self.multi_device_sequences.append(i)
            else:
                self.single_device_sequences.append(i)

    def load_sequence(self, index, device_num=0) -> None:
        """Fetches an ADTSubsequence object for the sequence at index and
        device_num"""
        self.data = ADTSubsequence(
            self.device_data[index][device_num], device_num=device_num
        )
        self.id = index

    def get_all_annotated_objects(self) -> Tuple[Dict[str, Any], list]:
        return self.data.objects, self.data.object_instance_ids

    ## easy interface to low-lying functions
    def get_rectified_2d_bbox(self, bbox_2d) -> Optional[np.ndarray]:
        return self.data.get_rectified_2d_bbox(bbox_2d)

    def create_object_retrieval_benchmark(
        self, num_sequences, num_objects, method="id", llm=None
    ):
        """Creates a retrieval benchmark for ADT sequences"""
        num_tot_sequences = len(self.file_paths)
        sampled_sequences = random.sample(range(num_tot_sequences), num_sequences)
        sequence_to_object_map = {}
        for sequence_idx in sampled_sequences:
            self.load_sequence(sequence_idx)
            sampled_objects = random.sample(self.data.object_instance_ids, num_objects)
            if method == "id":
                sequence_to_object_map[sequence_idx] = sampled_objects
            elif method == "description":
                if llm is None:
                    raise ValueError(
                        "LLM object not provided, can not generate descriptions"
                    )
            else:
                raise ValueError(
                    f"Invalid method: {method}. Valid methods: id, description"
                )
        return sequence_to_object_map


class ADTSubsequence:
    """Wrapper class to make I/O easier for ADT data"""

    def __init__(self, path, device_num=-1, verbose=False) -> None:
        if device_num == -1:
            print("Using dummy subsequence data")
            raise NotImplementedError
        self._verbose = verbose
        self._device_num = device_num
        self.subsequence = AriaDigitalTwinDataProvider(path)
        self._rgb_stream_id = StreamId("214-1")
        self.rgb_timestamps = self.subsequence.get_aria_device_capture_timestamps_ns(
            self._rgb_stream_id
        )
        self._data_keys = [
            "rgb",
            # "slam-l",  # not supported right now
            # "slam-r",  # not supported right now
            "segmentation",
            "depth",
            "2dbbox",
            "3dbbox",
            "pose",
        ]
        self._raw_data = {}
        self._data_getters = []
        self._data_getters.append(self._get_rgb)
        self._data_getters.append(self._get_segmentation)
        self._data_getters.append(self._get_depth)
        self._data_getters.append(self._get_2dbbox)
        self._data_getters.append(self._get_3dbbox)
        self._data_getters.append(self._get_pose)
        self._load_object_info()
        self._T_device_rgb = (
            self.subsequence.raw_data_provider_ptr()
            .get_device_calibration()
            .get_transform_device_sensor("camera-rgb")
        )

    def _transform_pose(self, data) -> Tuple[Any, int]:
        """Transforms pose from left IMU frame to RGB camera frame"""
        transform_world_rgb = (
            data["pose"].transform_scene_device.matrix() @ self._T_device_rgb.matrix()
        )
        return (
            transform_world_rgb,
            data["dt_all"]["pose"],
        )  # this is probably a Sophus SE3 object

    def get_rectified_2d_bbox(self, bbox_2d) -> Optional[np.ndarray]:
        """Rectifies 2D bounding box"""
        sensor_name = self.subsequence.raw_data_provider_ptr().get_label_from_stream_id(
            self._rgb_stream_id
        )
        device_calib = self.subsequence.raw_data_provider_ptr().get_device_calibration()
        src_calib = device_calib.get_camera_calib(sensor_name)
        bbox2d_coords = bbox2d_to_image_coordinates(bbox_2d)
        # create output calibration: a pin-hole rectificied image size 512x512 and focal length 280
        dst_calib = calibration.get_linear_camera_calibration(
            512, 512, 280, sensor_name
        )
        src_calib = device_calib.get_camera_calib(sensor_name)
        rect_bbox2d_xs = []
        rect_bbox2d_ys = []
        rect_bbox2d = None
        for bbox2d_coord in bbox2d_coords:
            unprojected_bbox2d_ray = src_calib.unproject_no_checks(bbox2d_coord)
            rect_bbox2d_coord = dst_calib.project(unprojected_bbox2d_ray)
            if rect_bbox2d_coord is not None:
                rect_bbox2d_xs.append(rect_bbox2d_coord[0])
                rect_bbox2d_ys.append(rect_bbox2d_coord[1])
            else:
                return rect_bbox2d
        rect_bbox2d = [
            min(rect_bbox2d_xs),
            max(rect_bbox2d_xs),
            min(rect_bbox2d_ys),
            max(rect_bbox2d_ys),
        ]
        return rect_bbox2d

    def __process_image_frame(
        self, frame_data, timestamp, frame_name: str
    ) -> Tuple[np.ndarray, int]:
        """Helper function to process image frames and check validity"""
        assert frame_data.is_valid(), "{} not valid at timestamp: {}".format(
            frame_name, timestamp
        )
        if self._verbose:
            print(
                f"Delta b/w given {timestamp*1e-9=} and time of frame {frame_data.dt_ns()*1e-9=}"
            )
        self._raw_data[frame_name] = frame_data
        return frame_data.data().to_numpy_array(), frame_data.dt_ns()

    def __process_data_frame(
        self, frame_data, timestamp, frame_name: str
    ) -> Tuple[np.ndarray, int]:
        """Helper function to process data frames and check validity"""
        assert frame_data.is_valid(), "{} not valid at timestamp: {}".format(
            frame_name, timestamp
        )
        self._raw_data[frame_name] = frame_data
        if self._verbose:
            print(
                f"Delta b/w given {timestamp*1e-9=} and time of frame {frame_data.dt_ns()*1e-9=}"
            )
        return frame_data.data(), frame_data.dt_ns()

    def __iter__(self):
        return ADTSubsequenceIterator(self)

    def __reversed__(self):
        return ADTSubsequenceIterator(self, reverse=True)

    def _get_rgb(self, timestamp) -> Tuple[np.ndarray, int]:
        """Gets RGB image at timestamp"""
        image = self.subsequence.get_aria_image_by_timestamp_ns(
            timestamp, self._rgb_stream_id
        )
        return self.__process_image_frame(image, timestamp, "RGBImage")

    def _get_depth(self, timestamp) -> Tuple[np.ndarray, int]:
        """Gets depth image at timestamp"""
        depth = self.subsequence.get_depth_image_by_timestamp_ns(
            timestamp, self._rgb_stream_id
        )
        return self.__process_image_frame(depth, timestamp, "DepthImage")

    def _get_segmentation(self, timestamp) -> Tuple[np.ndarray, int]:
        """Gets segmentation annotation image at timestamp from RGB stream"""
        segmentation = self.subsequence.get_segmentation_image_by_timestamp_ns(
            timestamp, self._rgb_stream_id
        )
        return self.__process_image_frame(segmentation, timestamp, "SegmentationImage")

    def _get_2dbbox(self, timestamp) -> Tuple[np.ndarray, int]:
        """Gets GT 2D bounding boxes for all objects at timestamp from RGB stream"""
        bbox_2d = self.subsequence.get_object_2d_boundingboxes_by_timestamp_ns(
            timestamp, self._rgb_stream_id
        )
        return self.__process_data_frame(bbox_2d, timestamp, "2DBoundingBox")

    def _get_3dbbox(self, timestamp) -> Tuple[np.ndarray, int]:
        """Gets GT 3D bounding boxes for all objects at timestamp from RGB stream"""
        bbox_3d = self.subsequence.get_object_3d_boundingboxes_by_timestamp_ns(
            timestamp
        )
        return self.__process_data_frame(bbox_3d, timestamp, "3DBoundingBox")

    def __to_dict(self, an_object) -> Dict[str, Any]:
        """helper function to convert an object to dict"""
        return {k: getattr(an_object, k) for k in an_object.__slots__}

    def _get_intrinsics(self) -> Tuple[Any, int]:
        """Gets camera intrinsics at timestamp"""
        _intrinsics = self.subsequence.get_aria_camera_calibration(self._rgb_stream_id)
        # intrinsics = self.__to_dict(_intrinsics)
        return (_intrinsics, 0)

    def _load_object_info(self):
        """gets all objects in the sequence and creates an ID to Info mapping"""
        self.object_instance_ids = self.subsequence.get_instance_ids()
        self.objects = {}
        for instance_id in self.object_instance_ids:
            self.objects[instance_id] = self.subsequence.get_instance_info_by_id(
                instance_id
            )

    def _get_pose(self, timestamp) -> Tuple[np.ndarray, int]:
        """Gets agent 3D pose at timestamp, this is pose of left IMU"""
        agent_pose = self.subsequence.get_aria_3d_pose_by_timestamp_ns(timestamp)
        return self.__process_data_frame(agent_pose, timestamp, "AgentPose")

    def get_data_by_timestamp(self, timestamp, synced=True) -> Dict[str, Any]:
        """makes sure data is time-synced and returns the closest data to the
        input timestamp"""
        data = {}
        step = 0.25 * 1e9
        dt_threshold = 0.1 * 1e9
        old_timestamp = timestamp
        while (
            timestamp >= self.rgb_timestamps[0] and timestamp <= self.rgb_timestamps[-1]
        ):
            dt_max = 0
            dt_sign = 1
            for key, getter in zip(self._data_keys, self._data_getters):
                datum = getter(timestamp)
                data[key] = datum[0]
                if "dt_all" not in data:
                    data["dt_all"] = {}
                data["dt_all"][key] = datum[1]
                new_dt_max = max(dt_max, abs(datum[1]))
                dt_sign = np.sign(datum[1]) if new_dt_max != dt_max else dt_sign
                dt_max = new_dt_max
            data["dt_max"] = dt_max
            if not synced or dt_max < dt_threshold:
                data["timestamp"] = timestamp
                data["dt_all"]["timestamp"] = timestamp - old_timestamp
                data["transformed_pose"] = self._transform_pose(data)
                data["intrinsics"] = self._get_intrinsics()
                # TODO: update dt_ns to be wrt original timestamp and not the modified one
                for key in ["rgb", "segmentation", "depth"]:
                    input_img = data[key]
                    if key == "depth":
                        input_img = input_img.astype(np.float32)
                    data[f"rectified-{key}"] = self._rectify_rgb(input_img)
                    data["dt_all"][f"rectified-{key}"] = data["dt_all"][key]
                    logging.debug(
                        f"Min-Max value for {key=}: {np.min(input_img)}, {np.max(input_img)}"
                    )
                break
            else:
                logging.debug(f"{dt_sign=}, {dt_max=}")
                logging.debug(
                    f"{timestamp=} is not valid, trying {timestamp+dt_sign*step}"
                )
                logging.debug(
                    f"Limits of timestamps: {self.rgb_timestamps[0]}, {self.rgb_timestamps[-1]}"
                )
                timestamp += int(dt_sign * step)
                data = {}
        return data

    def _rectify_rgb(self, rgb_image: np.ndarray) -> np.ndarray:
        """Rectifies RGB image"""
        # get source calibration - Aria original camera model
        sensor_name = self.subsequence.raw_data_provider_ptr().get_label_from_stream_id(
            self._rgb_stream_id
        )
        device_calib = self.subsequence.raw_data_provider_ptr().get_device_calibration()
        src_calib = device_calib.get_camera_calib(sensor_name)

        # create output calibration: a pin-hole rectificied image size 512x512 and focal length 280
        dst_calib = calibration.get_linear_camera_calibration(
            512, 512, 280, sensor_name
        )

        # rectify image
        rectified_image = calibration.distort_by_calibration(
            rgb_image, dst_calib, src_calib
        )
        return rectified_image

    def visualize_image_with_3dbbox(self, image, bbox3d, aria3dpose) -> None:
        """Visualizes RGB image with 3D bounding boxes"""
        # get object poses and Aria poses of the selected frame
        print("AABB [xmin, xmax, ymin, ymax, zmin, zmax]: ", bbox3d.aabb)

        # now to project 3D bbox to Aria camera
        # get 6DoF object pose with respect to the target camera
        transform_cam_device = self.subsequence.get_aria_transform_device_camera(
            self._rgb_stream_id
        ).inverse()
        transform_cam_scene = (
            transform_cam_device.matrix()
            @ aria3dpose.transform_scene_device.inverse().matrix()
        )
        transform_cam_obj = transform_cam_scene @ bbox3d.transform_scene_object.matrix()

        # get projection function
        cam_calibration = self.subsequence.get_aria_camera_calibration(
            self._rgb_stream_id
        )
        assert cam_calibration is not None, "no camera calibration"

        # get projected bbox
        reprojected_bbox = adt_utils.project_3d_bbox_to_image(
            bbox3d.aabb, transform_cam_obj, cam_calibration
        )
        if reprojected_bbox:
            # plot
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.axis("off")
            ax.add_patch(
                plt.Polygon(
                    reprojected_bbox,
                    linewidth=1,
                    edgecolor="y",
                    facecolor="none",
                )
            )
            plt.show()
        else:
            print("\033[1m" + "\033[91m" + "Try another object!" + "\033[0m")

    def linear_search_for_object_in_sequence(
        self,
        object_id_list: list,
        num_instances: int = 1,
        stream_id: StreamId = StreamId("214-1"),  # RGB device code
    ) -> dict:
        """Linear search for object in sequence"""
        # TODO: This is a generally useful function, maybe move out to a utils file?
        found_frames = {}
        found = {oid: False for oid in object_id_list}
        iterator = ADTSubsequenceIterator(self, reverse=True)
        object_id_list = set(object_id_list)
        while True:
            try:
                data = next(iterator)
            except StopIteration:
                print("Reached end of sequence!")
                break
            segmentation_frame = data["segmentation"][0]
            instance_ids_in_frame = set(np.unique(segmentation_frame))
            seg_matches = list(instance_ids_in_frame.intersection(object_id_list))

            if seg_matches:
                for match in seg_matches:
                    if match not in found_frames:
                        found_frames[match] = [data]
                    else:
                        found_frames[match].append(data)
                    if len(found_frames[match]) == num_instances:
                        found[match] = True
                        object_id_list.remove(match)
            if len(object_id_list) == 0:
                print("Found enough instances! Breaking!")
                break
        return found_frames


class ADTSubsequenceIterator:
    """Iterator class for ADTSubSequence (only operates over RGB data right now)"""

    def __init__(
        self, container: ADTSubsequence, reverse=False, ns_delta=0.1 * 1e9
    ) -> None:
        self.container = container
        self._ns_delta = int(ns_delta)
        self._threshold_ns = int(0.1 * 1e9)
        if reverse:
            self.end_limit = self.container.rgb_timestamps[0]
            self.start_limit = self.container.rgb_timestamps[-1]
            self._ns_delta *= -1
        else:
            self.end_limit = self.container.rgb_timestamps[-1]
            self.start_limit = self.container.rgb_timestamps[0]
        self.curr_ts = self.start_limit
        self._reverse = reverse

    def __next__(self):
        while True:
            if self._reverse and self.curr_ts <= self.end_limit:
                raise StopIteration
            elif not self._reverse and self.curr_ts >= self.end_limit:
                raise StopIteration
            data = self.container.get_data_by_timestamp(self.curr_ts, synced=False)
            self.curr_ts = data["timestamp"]
            self.curr_ts += self._ns_delta
            if abs(data["dt_all"]["segmentation"]) < self._threshold_ns:
                break
        return data


@click.command()
@click.option("--adt-path", type=click.Path(exists=True))
@click.option("--is-root/--not-root", type=bool, default=True, is_flag=True)
@click.option("--verbose/--not-verbose", type=bool, default=False, is_flag=True)
@click.option("--visualize/--no-visualize", type=bool, default=True, is_flag=True)
@click.option("--fwd/--reverse", type=bool, default=True, is_flag=True)
def main(adt_path: str, is_root: bool, verbose: bool, visualize: bool, fwd: bool):
    adt_sequences = ADTSequences(adt_path, is_path=(not is_root), verbose=verbose)
    sequence_index = 10
    subsequence_index = 0
    adt_sequences.load_sequence(sequence_index, subsequence_index)
    if fwd:
        print("Trying fwd access")
    else:
        print("Trying reverse access")
    iterator = ADTSubsequenceIterator(adt_sequences.data, reverse=(not fwd))
    data = next(iterator)
    print(data)
    all_object_ids = np.unique(data["segmentation"])
    all_object_ids = all_object_ids[all_object_ids != 0]
    random_object_id = np.random.choice(all_object_ids)
    object_info, _ = adt_sequences.get_all_annotated_objects()
    print("Object info: ", object_info[random_object_id])
    print(f"Aria pose: {data['pose'].transform_scene_device.matrix()}")
    if visualize:
        plt.imshow(data["rgb"])
        plt.show()
        plt.imshow(data["segmentation"])
        plt.show()
        adt_sequences.data.visualize_image_with_3dbbox(
            data["rgb"],
            data["3dbbox"][random_object_id],
            data["pose"],
        )
        adt_sequences.data.visualize_image_with_3dbbox(
            data["segmentation"],
            data["3dbbox"][random_object_id],
            data["pose"],
        )


@click.command()
@click.option("--adt-path", type=click.Path(exists=True))
@click.option("--output-path", type=click.Path(exists=True))
@click.option("--is-root/--not-root", type=bool, default=True, is_flag=True)
@click.option("--verbose/--not-verbose", type=bool, default=False, is_flag=True)
@click.option("--visualize/--no-visualize", type=bool, default=True, is_flag=True)
def get_object_stats(
    adt_path: str, output_path: str, is_root: bool, verbose: bool, visualize: bool
):
    object_pair_list = []
    object_cat_to_id_count = {}
    adt_sequences = ADTSequences(adt_path, is_path=(not is_root), verbose=verbose)
    for sequence_idx in tqdm(range(len(adt_sequences.file_paths))):
        adt_sequences.load_sequence(sequence_idx)
        objects, _ = adt_sequences.get_all_annotated_objects()
        for oid, oinfo in objects.items():
            if oinfo.category not in object_cat_to_id_count:
                object_cat_to_id_count[oinfo.category] = 1
            else:
                object_cat_to_id_count[oinfo.category] += 1
            object_pair_list.append((oid, oinfo.name, oinfo.category))
    output_file = os.path.join(output_path, "adt_object_stats.yaml")
    with open(output_file, "w") as f:
        yaml.dump(object_cat_to_id_count, f, default_flow_style=False)
    print("Saved object stats to: ", output_file)
    output_file = os.path.join(output_path, "adt_object_pairs.yaml")
    with open(output_file, "w") as f:
        yaml.dump(object_pair_list, f, default_flow_style=False)
    print("Saved object pairs to: ", output_file)


if __name__ == "__main__":
    get_object_stats()
