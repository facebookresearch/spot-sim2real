import logging
import math
import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import click
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import openai
import torch
import yaml
from aria_data_utils.adt_data_loader import ADTSequences, ADTSubsequenceIterator
from aria_data_utils.memory_interface.memory_objects import Memory
from tqdm import tqdm

# TODO: we can reuse OwlVit implemented in spot-rl here instead of this barebone thing
from transformers import OwlViTForObjectDetection, OwlViTProcessor

logger = logging.getLogger("adt_memory_provider")
logging.basicConfig(level=logging.DEBUG)


class ADTMemoryProvider:
    def __init__(self, config) -> None:
        self.data_root = config.data_path
        self.data_type = "adt"
        self.data_loader = ADTSequences(
            self.data_root,
            is_path=(not config.is_root),
            verbose=config.verbose,
        )
        self._objects = None
        self._episode_id: int = None
        if hasattr(config, "sequence_id"):
            self.load(config.sequence_id)

        self.object_prompt = """
Find all matching ID of the TARGET-OBJECT in the list of FOUND-OBJECTS in the provided query. ANSWER should be a list of those IDs.

---
Example:

TARGET-OBJECT:
RedSmallVase

FOUND-OBJECTS:
RedBigVase which is category (Vase) has id (1),
RedSmallVase which is category (Vase) has id (2),
PinkSmallVase which is category (Vase) has id (3),

ANSWER: [2]
---
Example:

TARGET-OBJECT:
BlackDoorFrame

FOUND-OBJECTS:
BlackChair which is category (furniture) has id (1),
RedDoorFrame which is category (fixture) has id (2),
BlackPhotoFrame which is category (decor) has id (3),

ANSWER: []
---
Now answer following query:

TARGET-OBJECT:
{object_name}

FOUND-OBJECTS:
{found_object_name}

ANSWER:"""

    def load(self, sequence_id: int) -> None:
        """load a particular sequence from the data loader for access"""
        self.data_loader.load_sequence(sequence_id)
        self._episode_id = sequence_id
        self._objects = self.data_loader.data.objects

    def get_object_annotations(self) -> Tuple[Dict, List]:
        """get metadata for all objects in the sequence"""
        return self.data_loader.get_all_annotated_objects()

    def get_object_context(
        self,
        object_name: str,
        llm=None,
        debug=False,
        method: str = "llm",
    ) -> List[Memory]:
        """find location and context of seen objects in memory if object is
        similar to <object_name>

        - llm: Pass in a language model to use for generating the object list based
        on name similarity
        - vlm: Pass in a vision model to use for generating the object list based
        on visual similarity
        - method: A string which can be either "llm" or "vlm" to indicate which method
        to use for generating the object list
        """
        (
            object_info_dict,
            object_instance_ids,
        ) = self.data_loader.get_all_annotated_objects()
        if llm is None and method == "llm":
            logger.info("LLM not provided, assuming this is a test routine")
            object_id_list = [
                obj_id
                for obj_id, obj_info in object_info_dict.items()
                if obj_info.name == object_name
            ]
            # search for these objects in the sequence
            found_frames = self.data_loader.data.linear_search_for_object_in_sequence(
                object_id_list
            )
            ground_truth = True
        elif method == "llm":
            logger.info("Using LLM to generate object list")
            object_id_list = self._llm_based_object_search(
                object_name, object_info_dict, llm, debug
            )
            # search for these objects in the sequence
            found_frames = self.data_loader.data.linear_search_for_object_in_sequence(
                object_id_list
            )
            ground_truth = True
        elif method == "vlm":
            logger.info("Using VLM to generate object list")
            found_frames = self._vlm_based_object_search(
                object_name, object_info_dict, debug
            )
            ground_truth = False
            if not found_frames:
                return []
        # look up frame pose data for each retrieved ts
        return self._get_object_context_from_frames(
            found_frames, object_info_dict, debug, ground_truth=ground_truth
        )

    def __plot_boxes_on_image(
        self, boxes, image, save=False, ep_id=None, labels=None, name=None
    ):
        logger.info(f"Plotting {len(boxes)} boxes on image: {name}")
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        for i, box in enumerate(boxes):
            fig, ax = plt.subplots()
            ax.imshow(image)
            box = [int(i) for i in box]
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                fill=False,
                color="red",
            )
            ax.add_patch(rect)
            if not save:
                plt.show()
            else:
                if name is None:
                    plt.savefig(f"{ep_id}_{labels[i]}.png")
                else:
                    plt.savefig(name)
            plt.close()

    def __plot_adt_boxes_on_image(
        self, boxes, image, ep_id=None, save=False, label=None, name=None
    ):
        if ep_id is None:
            ep_id = self.data_loader.id
        logger.info(f"Plotting {len(boxes)} boxes on image {name}")
        image = np.rot90(image, k=3)

        fig, ax = plt.subplots()
        ax.imshow(image)
        for box in boxes:
            box = [int(i) for i in box]
            rect = patches.Rectangle(
                (box[0], box[2]),
                box[1] - box[0],
                box[3] - box[2],
                fill=False,
                color="red",
            )
            ax.add_patch(rect)
        if not save:
            plt.show()
        else:
            if name is None:
                plt.savefig(f"{ep_id}_{label}.png")
            else:
                plt.savefig(name)
        plt.close()

    def _vlm_based_object_search(
        self,
        object_names,
        object_ids,
        object_info_dict,
        debug: bool = False,
        num_instances=5,
        ep_id=None,
        ns_delta=0.25 * 1e9,
    ) -> Tuple[Dict[str, list], Dict[int, list]]:
        """use vision model to find objects similar to <object_names>
        multiple object-names may map to the same object instance, in that case
        object_ids should be provided to get a measure of both instance-level and
        description-level predictions
        """
        named_results: Dict[str, list] = {obj_name: [] for obj_name in object_names}
        if not object_ids or object_ids is None:
            raise ValueError(
                "object_ids must be provided for instance-level predictions"
            )
        id_results: Dict[int, list] = {oid: [] for oid in object_ids}
        processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        logger.info(f"Searching for {object_names} in sequence")
        reverse_iter = ADTSubsequenceIterator(
            self.data_loader.data, reverse=True, ns_delta=ns_delta
        )
        i = 0
        while True:
            try:
                frame = next(reverse_iter)
            except StopIteration:
                logger.info("Reached end of sequence")
                break
            logger.debug(f"Processing frame {i}")
            # normalize image and rotate clock-wise by 90 degs
            images = frame["rectified-rgb"]
            images = np.rot90(images, k=3)
            texts = [f"a photo of a {object_name}" for object_name in object_names]
            logger.debug(f"Searching for {object_names} in sequence")
            inputs = processor(text=texts, images=images, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.Tensor([images.shape[:2]])
            results = processor.post_process_object_detection(
                outputs=outputs, threshold=0.25, target_sizes=target_sizes
            )
            # TODO: following won't work with multiple images as input
            if len(results[0]["boxes"]) > 0:
                boxes = results[0]["boxes"]
                scores = results[0]["scores"]
                labels = results[0]["labels"]
                for box, score, label in zip(boxes, scores, labels):
                    # binary_score, iou_score = calculate_score(
                    #     frame,
                    #     object_ids[label],
                    #     box=box.detach().cpu().numpy(),
                    # )
                    result = {
                        "owl-vit-prompt": texts[label],
                        "object-name": object_names[label],
                        "object-id": object_ids[label],
                        "owl-vit-bbox": box.detach().cpu().numpy(),
                        "owl-vit-score": score.detach().cpu().numpy(),
                        "adt_data_timestamp": frame["timestamp"],
                        # "binary_score": binary_score,
                        # "iou_score": iou_score,
                    }
                    logger.debug(result)
                    named_results[object_names[label]].append(result)
                    if id_results is not None:
                        id_results[object_ids[label]].append(result)
                self.__plot_boxes_on_image(
                    boxes.detach().numpy(),
                    images,
                    save=True,
                    ep_id=f"{ep_id}_{i}",
                    labels=[object_names[label] for label in labels],
                )
                for oid, frames in id_results.items():
                    if oid in object_ids and len(frames) >= num_instances:
                        logger.info(
                            f"Found {num_instances} instances of {oid}, {[oname for i, oname in enumerate(object_names) if object_ids[i] == oid]}"
                        )
                        logger.debug(
                            f"List of objects before filtering: {object_ids=}, {object_names=}"
                        )
                        # drop this object from the list of objects to search for
                        # and also delete related names
                        index = [i for i, x in enumerate(object_ids) if x == oid]
                        object_ids = np.delete(object_ids, index).tolist()
                        object_names = np.delete(object_names, index).tolist()
                        logger.debug(
                            f"List of objects after filtering: {object_ids=}, {object_names=}"
                        )
                if len(object_names) == 0:
                    break
            i += 1
        return named_results, id_results

    def _llm_based_object_search(
        self, object_name, object_info_dict, llm, debug: bool = False
    ) -> List[int]:
        object_id_list: List[int] = []
        objects = ""
        object_num = 0
        tot_objects = len(object_info_dict)
        tot_obj_num = 0
        obj_threshold = 35
        for obj_id, obj_info in object_info_dict.items():
            objects += f"{obj_info.name} which is category ({obj_info.category}) has id ({obj_info.id}), \n"
            object_num += 1
            tot_obj_num += 1
            if object_num == obj_threshold or tot_obj_num == tot_objects:
                objects += "-------------------------------  \n"
                prompt = self.object_prompt.replace("{object_name}", object_name)
                prompt = prompt.replace("{found_object_name}", objects)
                response = None
                while response is None:
                    response = None
                    try:
                        response = llm.generate(prompt, stop="Done")
                    except openai.error.InvalidRequestError as e:
                        print(f"Exception: {e}")
                        print(f"Prompt: {prompt}, chars: {len(prompt)}")
                        breakpoint()
                # parse response and add each int to object_id_list
                response = response.strip("\n").strip(" ").lstrip("[").rstrip("]")
                for line in response.split(","):
                    if line.isdigit():
                        object_id_list.append(int(line))
                object_num = 0
                objects = ""
        return object_id_list

    def _get_object_context_from_frames(
        self, found_frames, object_info_dict, debug=False, ground_truth=False
    ):
        context = []
        if ground_truth:
            for obj_id, frames in found_frames.items():
                for frame_data in frames:
                    segmentation = frame_data["segmentation"]
                    seg_mask = np.zeros_like(segmentation)
                    seg_mask[segmentation == obj_id] = 1
                    object_mem = Memory(
                        name=object_info_dict[obj_id].name,
                        wearer_location=frame_data[
                            "pose"
                        ].transform_scene_device.matrix()[:3, 3],
                        instance_id=obj_id,
                        category_name=object_info_dict[obj_id].category,
                        associated_imgs=np.expand_dims(frame_data["rgb"], axis=0),
                        associated_3dbbox=np.expand_dims(
                            frame_data["3dbbox"][obj_id], axis=0
                        ),
                        seg_mask=np.expand_dims(
                            seg_mask,
                            axis=0,
                        ),
                    )
                    context.append(object_mem)
                if debug:
                    breakpoint()
                    # visualize each segmented image with object in it
                    object_mem.visualize_objects_with_seg([0])
        else:
            for obj_desc, frames in found_frames.items():
                for frame_data in frames:
                    # TODO: pass the bbox output from OWL-ViT here
                    object_mem = Memory(
                        name=obj_desc,
                        wearer_location=frame_data[
                            "pose"
                        ].transform_scene_device.matrix()[:3, 3],
                        instance_id=uuid.uuid4(),
                        category_name=obj_desc,
                        associated_imgs=np.expand_dims(frame_data["rgb"], axis=0),
                    )
                    context.append(object_mem)
        return context

    def _to_memory(self, object_info) -> Memory:
        return Memory(
            name=object_info.name,
            wearer_location=np.random.rand(3),
            instance_id=object_info.id,
            category_name=object_info.category,
        )

    def _choose_gt_objects(
        self, num_instances: int = 1
    ) -> Tuple[List[int], List[dict]]:
        (
            object_info_dict,
            object_instance_ids,
        ) = self.data_loader.get_all_annotated_objects()
        chosen_object_ids = np.random.choice(object_instance_ids, num_instances)
        chosen_objects = [object_info_dict[obj_id] for obj_id in chosen_object_ids]
        return chosen_objects, [object_info_dict[oid] for oid in chosen_object_ids]

    def create_object_retrieval_benchmark(
        self, num_sequences, num_objects, method="id", llm=None
    ):
        """Creates a retrieval benchmark for ADT sequences"""
        num_tot_sequences = len(self.data_loader.file_paths)
        sampled_sequences = random.sample(range(num_tot_sequences), num_sequences)
        sequence_to_object_map = {}
        for sequence_idx in sampled_sequences:
            self.load(sequence_idx)
            sampled_objects = random.sample(
                self.data_loader.data.object_instance_ids, num_objects
            )
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

    def _check_object_validity(
        self,
        object_id,
        object_data,
        frame_data,
        allow_list,
        area_threshold=5000,
        depth_threshold=1.0 * 1e3,
        object_depth_threshold=0.75,
    ) -> Tuple[bool, Any]:
        """checks if the object visuals are valid to use with a VLM"""
        valid = True
        # majority depth of the object should be less than a threshold
        valid &= bool(object_data.category in allow_list)
        if not valid:
            return valid, None
        segmentation = frame_data["segmentation"]
        depth = frame_data["depth"]
        segmentation[segmentation != object_id] = 0
        depth[segmentation == 0] = 0
        depth[depth > depth_threshold] = 0
        num_object_pixels = np.sum(segmentation == object_id)
        num_depth_filtered_pixels = np.sum(depth > 0)
        valid &= bool(
            (num_depth_filtered_pixels / num_object_pixels) > object_depth_threshold
        )
        if not valid:
            return valid, None
        # area to be greater than a threshold
        bbox = frame_data["2dbbox"][object_id].box_range
        valid &= bool(
            ((bbox[1] - bbox[0]) * (bbox[3] - bbox[2])) > area_threshold
        )  # in pixels
        # rectified bbox should not be None
        rect_bbox2d = self.data_loader.get_rectified_2d_bbox(
            frame_data["2dbbox"][object_id].box_range
        )
        valid &= bool(rect_bbox2d is not None)
        return valid, rect_bbox2d

    def create_owl_vit_retrieval_benchmark(
        self,
        num_seq=10,
        num_frames=10,
        depth_thresh=1.0 * 1e3,
        area_threshold=0.5,
        dt_threshold=0.25 * 1e9,
    ):
        """Creates a depth-based object retrieval benchmark for ADT sequences"""
        allow_list = [
            "wall clock",
            "tray",
            "toy",
            "spray bottle",
            "scissors",
            "pot",
            "plate",
            "pet bowl",
            "notebook",
            "laptop computer",
            "keys",
            "desk clock",
            "cup",
            "computer mouse",
            "computer keyboard",
            "candle",
            "can",
            "bottle",
            "bowl",
            "book",
            "basket",
            "food object",
            "food processor",
            "container",
            "clothes basket",
            "box",
            "basketball",
            "baseball bat",
            "baking pan",
            "air purifier",
            "dehumidifer",
            "cutting board",
            "decorative accessory",
            "frying pan",
            "musical instrument",
            "play set",
            "refrigerator",
            "spice rack",
            "tea kettle",
            "television",
            "toaster",
            "wallet",
        ]
        num_tot_sequences = len(self.data_loader.file_paths)
        sampled_sequences = random.sample(range(num_tot_sequences), num_seq)
        sequence_to_object_map = {}
        for sequence_idx in tqdm(sampled_sequences):
            self.load(sequence_idx)
            sampled_frames = random.sample(
                self.data_loader.data.rgb_timestamps, num_frames
            )
            sequence_to_object_map[sequence_idx] = []
            _objects = self.data_loader.data.objects
            sequence_categories = set()
            for frame_ts in sampled_frames:
                frame_data = self.data_loader.data.get_data_by_timestamp(frame_ts)
                # TODO: ignore this frame if max_dt is > some threshold
                if frame_data and frame_data["dt_max"] < dt_threshold:
                    # get depth data for this frame
                    depth_frame = frame_data["depth"]
                    logger.debug(f"Max depth value: {np.max(depth_frame)}")
                    depth_frame[depth_frame > depth_thresh] = 0
                    logger.debug(
                        f"Max depth value after thresholding: {np.max(depth_frame)}"
                    )
                    segmentation = frame_data["segmentation"]
                    object_is_valid = False
                    chosen_object = None
                    randomized_objects = np.random.permutation(np.unique(segmentation))
                    index = 0
                    while not object_is_valid and index < len(randomized_objects):
                        chosen_object = int(randomized_objects[index])
                        if chosen_object == 0:
                            index += 1
                            continue
                        object_data = _objects[chosen_object]
                        object_is_valid, rect_bbox2d = self._check_object_validity(
                            chosen_object, object_data, frame_data, allow_list
                        )
                        object_is_valid &= bool(
                            object_data.category not in sequence_categories
                        )
                        index += 1
                    if index == len(randomized_objects):
                        continue
                    sequence_categories.add(object_data.category)
                    object_info = _objects[chosen_object]
                    img_path = f"outputs/close_up_objects/{self.data_loader.id}_{chosen_object}.png"
                    sequence_to_object_map[sequence_idx].append(
                        {
                            "id": chosen_object,
                            "instance-name": object_info.name,
                            "category-name": object_info.category,
                            "image-path": img_path,
                            "depth-dt": frame_data["dt_all"]["depth"],
                            "segmentation-dt": frame_data["dt_all"]["segmentation"],
                            "object-description": "",
                        }
                    )
                    rgb_img = frame_data["rectified-rgb"]
                    self.__plot_adt_boxes_on_image(
                        [rect_bbox2d], rgb_img, save=True, name=img_path
                    )
        return sequence_to_object_map


@click.command()
@click.option("--data-path", type=str, default="data/ADT/ADT-1")
@click.option("--sequence-id", type=int, default=0)
@click.option("--is-root", type=bool, default=False, is_flag=True)
@click.option("--debug", type=bool, default=False, is_flag=True)
@click.option("--verbose", type=bool, default=False, is_flag=True)
def test_provider(
    data_path: str, sequence_id: int, is_root: bool, debug: bool, verbose: bool
):
    """test the provider by retrieving single and multiple objects"""

    @dataclass
    class test_config:
        """for replicating minimal version of overall config for testing"""

        data_path: str
        sequence_id: int
        is_root: bool
        debug: bool
        verbose: bool

    minimal_config = test_config(data_path, sequence_id, is_root, debug, verbose)
    provider = ADTMemoryProvider(minimal_config)
    print(f"Retrieving a single object from {data_path=}/{sequence_id=}")
    object_info, object_ids = provider.data_loader.get_all_annotated_objects()
    objects_not_found = 0
    for oid in object_ids:
        found_frames = {}
        found_frames = provider.data_loader.data.linear_search_for_object_in_sequence(
            [oid]
        )
        if len(found_frames) == 0:
            objects_not_found += 1
            print(f"Object {object_info[oid].name} not found in sequence {sequence_id}")
        else:
            print(f"Object {object_info[oid].name} found in sequence {sequence_id}")
    print(f"{objects_not_found} objects not found in sequence {sequence_id}")


def test_vlm():
    """test the vlm based object search"""

    @dataclass
    class test_config:
        """for replicating minimal version of overall config for testing"""

        data_path: str
        sequence_id: int
        is_root: bool
        debug: bool
        verbose: bool

    minimal_config = test_config("data/adt_data", 4, True, False, False)
    provider = ADTMemoryProvider(minimal_config)
    output = provider.create_owl_vit_retrieval_benchmark(num_seq=25, num_frames=200)
    with open("outputs/close_up_object_benchmark.yaml", "w") as out_file:
        yaml.dump(output, out_file, default_flow_style=False)


if __name__ == "__main__":
    test_vlm()
