# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# mypy: ignore-errors
import argparse
import random
import time
from typing import List

import cv2
import numpy as np
import rospy
import torch
from PIL import Image

try:
    from spot_rl.utils.sort import Sort
except Exception as e:
    print(e)
    Sort = False


def generate_fake_bboxes(detections, N, image_dims=(480, 640)):
    if len(detections) == 0:
        return detections  # don't add any fake detection if none is detected
    if detections[0]["scores"].shape[0] == 1:
        device = detections[0]["boxes"].device
        bboxes = detections[0]["boxes"].cpu()
        scores = detections[0]["scores"].cpu()
        labels = detections[0]["labels"].cpu()

        x1, y1, x2, y2 = bboxes[0].numpy()
        original_w = x2 - x1
        original_h = y2 - y1
        H, W = image_dims

        # List to store fake bounding boxes
        fake_bboxes = bboxes.numpy().tolist()
        fake_scores = scores.numpy().tolist()
        fake_labels = labels.numpy().tolist()
        fake_label = fake_labels[0]
        print("Adding fake detections to test tracker")
        for _ in range(N):
            # Generate random dimensions based on original bbox size
            w = np.random.randint(low=original_w // 2, high=min(W, original_w * 2))
            h = np.random.randint(low=original_h // 2, high=min(H, original_h * 2))

            # Ensure w > h by swapping if necessary
            if h > w:
                w, h = h, w

            # Generate random top left corner within the image boundaries
            x1_fake = np.random.randint(low=0, high=max(1, W - w))
            y1_fake = np.random.randint(low=0, high=max(1, H - h))

            # Calculate bottom right corner based on width and height
            x2_fake = x1_fake + w
            y2_fake = y1_fake + h

            # Add the fake bbox to the list
            fake_bboxes.append((x1_fake, y1_fake, x2_fake, y2_fake))
            fake_scores.append(0.05)
            fake_labels.append(fake_label)
        indices = list(range(len(fake_bboxes)))
        random.shuffle(indices)
        fake_detections = [
            {
                "boxes": torch.from_numpy(np.array(fake_bboxes)[indices])
                .reshape(-1, 4)
                .to(device),
                "scores": torch.from_numpy(np.array(fake_scores)[indices])
                .reshape(-1)
                .to(device),
                "labels": torch.from_numpy(np.array(fake_labels)[indices])
                .reshape(-1)
                .to(device),
            }
        ]
        return fake_detections
    else:
        return detections


def find_matching_bbox_index(bboxes, target_bbox):
    """
    Find the index of the bounding box in bboxes that matches the target_bbox,
    considering only the integer parts of their coordinates.

    Parameters:
    - bboxes (numpy.ndarray): An array of shape (n, 4) containing n bounding boxes.
    - target_bbox (tuple): A tuple of (x1, y1, x2, y2) representing the target bounding box.

    Returns:
    - numpy.ndarray: An array of indices of the matching bounding boxes. Can be empty if no matches are found.
    """
    # Convert to integers by truncating the decimal part
    bboxes_int = bboxes.astype(int)
    target_bbox_int = np.array(target_bbox).astype(int)

    # Use a vectorized comparison to check for equality with the target_bbox
    matches = np.all(bboxes_int == target_bbox_int, axis=1)

    # Find the indices where matches occur
    matching_indices = np.where(matches)[0]

    return matching_indices


class OwlVit:
    def __init__(
        self,
        labels,
        score_threshold,
        show_img,
        version: int = 1,
    ):
        if version < 1 or version > 2:
            raise ValueError(f"OWL-ViT version can only be 1 or 2. Received: {version}")
        self._version = version
        # self.device = torch.device('cpu')
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if version == 1:
            # Version protected import for OwlVIT
            from transformers import OwlViTForObjectDetection, OwlViTProcessor

            # Init model
            self.model = OwlViTForObjectDetection.from_pretrained(
                "google/owlvit-base-patch32",
                torch_dtype=torch.bfloat16,
            )

            # Init processor
            self.processor = OwlViTProcessor.from_pretrained(
                "google/owlvit-base-patch32"
            )
        else:
            # Version protected import for OwlV2
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            # Init model
            self.model = Owlv2ForObjectDetection.from_pretrained(
                "google/owlv2-base-patch16-ensemble",
                torch_dtype=torch.bfloat16,
            )

            # Init processor
            self.processor = Owlv2Processor.from_pretrained(
                "google/owlv2-base-patch16-ensemble"
            )
        self.model.eval()
        self.model.to(self.device)

        self.prefix = "an image of a"
        self.labels = [[f"{self.prefix} {label}" for label in labels[0]]]
        self.score_threshold = score_threshold
        self.show_img = show_img
        self.mot_tracker = None
        self.track_id = None

    def init_tracker(self):
        if not self.mot_tracker and not self.track_id:
            # Check if SORT was imported correctly
            if Sort:
                self.track_id = None
                self.mot_tracker = Sort(max_age=10, min_hits=10, iou_threshold=0.01)
            else:
                # if not imported correctly set is_tracking_enabled to be False
                rospy.set_param("is_tracking_enabled", False)

    def reset_tracker(self):
        self.track_id = None
        self.mot_tracker = None

    def reinit_tracker(self):
        # in case of id switch
        self.reset_tracker()
        self.init_tracker()

    def track(self, detections):
        is_tracking_enabled = rospy.get_param("is_tracking_enabled", False)
        if is_tracking_enabled:
            self.init_tracker()
        else:
            self.reset_tracker()
        if self.mot_tracker:
            if len(detections) == 0:
                detection = np.empty((0, 5))
            elif len(detections[0]["scores"]) == 0:
                detection = np.empty((0, 5))
            else:
                detection = detections[0]
                detection, score, _ = (
                    detection["boxes"],
                    detection["scores"],
                    detection["labels"],
                )
                device = detection.device
                (
                    detection,
                    score,
                ) = detection.clone().cpu().numpy(), score.clone().cpu().numpy().reshape(
                    -1, 1
                )
                detection = np.hstack([detection, score])
                # MAX
                # change np.argmax to be custom index based on LLM, currently its max conf index, can we use based on bbox y1
                det_to_track = detection[np.argmax(detection[:, -1])][:4]

            # Tracker will return NX5 array where last column is track_id,
            # Tracker will consume NX5 & may return MX5 where M != N
            tracks = self.mot_tracker.update(detection.copy())  # MX5 -> trackid
            print(
                f"Num of current detections {len(detection[:, :4])}, self.track_id {self.track_id}, tracker output {tracks}"
            )
            # track_id = 1 : t0  at t=n no track_id 1 from tracker output, rather it was 5
            if len(tracks) > 0:
                # once set per tracking episode, reset when tracking disabled
                try:
                    self.track_id = (
                        tracks[
                            find_matching_bbox_index(tracks[:, :4], det_to_track)
                        ].flatten()[-1]
                        if self.track_id is None
                        else self.track_id
                    )
                except:
                    breakpoint()
                track_id_found_at_mask = tracks[:, -1] == self.track_id
                if not np.any(track_id_found_at_mask):
                    # resetting tracker because of Id switch, this happens because of abrupt motion & we should reset tracker & redo tracking for this iteration
                    print(
                        f"id_switch event from {self.track_id} to {tracks[0][-1]} reintializing tracker"
                    )
                    self.reinit_tracker()
                    tracks = self.mot_tracker.update(detection.copy())
                    self.track_id = tracks[
                        find_matching_bbox_index(tracks[:, :4], det_to_track)
                    ].flatten()[-1]
                    track_id_found_at_mask = tracks[:, -1] == self.track_id
                    track_filtered_by_trackid = tracks[track_id_found_at_mask]
                track_filtered_by_trackid = tracks[
                    track_id_found_at_mask
                ]  # select only 1 track [x1, y1, x2, y2, track_id] where track_id == self.track_id

                if len(track_filtered_by_trackid) != 1:
                    # we didn't find any detection in current set of detections that associates with our current instance
                    return [
                        {
                            "scores": torch.Tensor([]),
                            "labels": torch.Tensor([]),
                            "boxes": torch.Tensor([]).reshape(0, 4),
                        }
                    ]
                # max_i = np.argmax(detection[:, -1])
                # just have one bounding box

                detections[0]["boxes"] = torch.cat(
                    (
                        detections[0]["boxes"],
                        torch.from_numpy(
                            track_filtered_by_trackid[0][:4].reshape(1, 4)
                        ).to(device),
                    )
                ).reshape(-1, 4)
                detections[0]["scores"] = torch.cat(
                    (detections[0]["scores"], torch.Tensor([1.0]).to(device))
                ).reshape(-1)
                detections[0]["labels"] = (
                    torch.cat((detections[0]["labels"], torch.Tensor([0]).to(device)))
                    .reshape(-1)
                    .to(torch.int64)
                )

                # Assign the one that is selected with highest score
                # detections[0]["scores"][max_i] = torch.tensor(100.0, device=self.device)
                print(
                    "Self.Track_id",
                    self.track_id,
                    "Det from mot_tracker where we found the self.track_id",
                    track_filtered_by_trackid[0][:4],
                )
            else:
                return [
                    {
                        "scores": torch.Tensor([]),
                        "labels": torch.Tensor([]),
                        "boxes": torch.Tensor([]).reshape(0, 4),
                    }
                ]

        return detections

    def run_inference(self, img):
        """
        img: an open cv image in (H, W, C) format
        """
        # Process inputs
        # img = img.to(self.device)
        inputs = self.processor(text=self.labels, images=img, return_tensors="pt")

        # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        # target_sizes = torch.Tensor([img.size[::-1]]) this is for PIL images
        target_sizes = torch.Tensor([img.shape[:2]]).to(self.device)
        inputs = inputs.to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs (bounding boxes and class logits) to COCO API
        if self._version == 1:
            results = self.processor.post_process(
                outputs=outputs, target_sizes=target_sizes
            )
        else:
            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes
            )
        # img = img.to('cpu')

        if self.show_img:
            self.show_img_with_overlaid_bounding_boxes(img, results)

        return self.get_most_confident_bounding_box_per_label(results)

    def run_inference_and_return_img(self, img, vis_img_required=True):
        """
        img: an open cv image in (H, W, C) format
        """
        # img = img.to(self.device)

        inputs = self.processor(text=self.labels, images=img, return_tensors="pt")
        target_sizes = (
            torch.Tensor([[max(img.shape[:2]), max(img.shape[:2])]]).to(self.device)
            if self._version > 1
            else torch.Tensor([img.shape[:2]]).to(self.device)
        )

        inputs = inputs.to(self.device)
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs (bounding boxes and class logits) to COCO API
        if self._version == 1:
            results = self.processor.post_process(
                outputs=outputs, target_sizes=target_sizes
            )
        else:
            results = self.processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes
            )
        # img = img.to('cpu')
        # if self.show_img:
        #    self.show_img_with_overlaid_bounding_boxes(img, results)
        # This will only return the bbox where we found our intial track

        # results = generate_fake_bboxes(results, 5)
        results = self.track(results)
        return (
            self.get_most_confident_bounding_box_per_label(results),
            self.create_img_with_bounding_box(img, results)
            if vis_img_required
            else None,
        )

    def show_img_with_overlaid_bounding_boxes(self, img, results):
        img = self.create_img_with_bounding_box(img, results)
        cv2.imshow("img", img)
        cv2.waitKey(1)

    def get_bounding_boxes(self, results):
        """
        Returns all bounding boxes with a score above the threshold
        """
        boxes, scores, labels = (
            results[0]["boxes"],
            results[0]["scores"],
            results[0]["labels"],
        )
        boxes = boxes.to("cpu")
        labels = labels.to("cpu")
        scores = scores.to("cpu")

        target_boxes = []
        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= self.score_threshold:
                target_boxes.append([self.labels[0][label.item()], score.item(), box])

        return target_boxes

    def get_most_confident_bounding_box(self, results):
        """
        Returns the most confident bounding box
        """
        boxes, scores, labels = (
            results[0]["boxes"],
            results[0]["scores"],
            results[0]["labels"],
        )
        boxes = boxes.to("cpu")
        labels = labels.to("cpu")
        scores = scores.to("cpu")

        target_box = []
        target_score = -float("inf")

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= self.score_threshold:
                if score > target_score:
                    target_score = score
                    target_box = box

        if target_score == -float("inf"):
            return None
        else:
            x1 = int(target_box[0])
            y1 = int(target_box[1])
            x2 = int(target_box[2])
            y2 = int(target_box[3])

            print("location:", x1, y1, x2, y2)
            return x1, y1, x2, y2

    def get_most_confident_bounding_box_per_label(self, results):
        """
        Returns the most confident bounding box for each label above the threshold
        """
        boxes, scores, labels = (
            results[0]["boxes"],
            results[0]["scores"],
            results[0]["labels"],
        )
        boxes = boxes.to("cpu")
        labels = labels.to("cpu")
        scores = scores.to("cpu")

        # Initialize dictionaries to store most confident bounding boxes and scores per label
        target_boxes = {}
        target_scores = {}

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= self.score_threshold:
                # If the current score is higher than the stored score for this label, update the target box and score
                if (
                    label.item() not in target_scores
                    or score > target_scores[label.item()]
                ):
                    target_scores[label.item()] = score.item()
                    target_boxes[label.item()] = box

        # Format the output
        result = []
        for label, box in target_boxes.items():
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])

            # Strip the prefix from the label
            label_without_prefix = self.labels[0][label][len(self.prefix) + 1 :]
            result.append(
                [label_without_prefix, target_scores[label], [x1, y1, x2, y2]]
            )

        return result

    def process_results(self, results):
        """
        Returns the all bounding box for each label above the threshold
        """
        boxes, scores, labels = (
            results[0]["boxes"],
            results[0]["scores"],
            results[0]["labels"],
        )
        boxes = boxes.to("cpu")
        labels = labels.to("cpu")
        scores = scores.to("cpu")

        # Initialize dictionaries to store most confident bounding boxes and scores per label
        target_boxes = {}
        target_scores = {}

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            if score >= self.score_threshold:
                # If the current score is higher than the stored score for this label, update the target box and score
                if label.item() not in target_scores:
                    target_scores[label.item()] = [score.item()]
                    target_boxes[label.item()] = [box]
                else:
                    target_scores[label.item()] += [score.item()]
                    target_boxes[label.item()] += [box]

        # Format the output
        result = []
        for label, boxes in target_boxes.items():
            for idx, box in enumerate(boxes):
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                # Strip the prefix from the label
                label_without_prefix = self.labels[0][label][len(self.prefix) + 1 :]
                result.append(
                    [label_without_prefix, target_scores[label][idx], [x1, y1, x2, y2]]
                )

        return result

    def create_img_with_bounding_box(self, img, results):
        """
        Returns an image with all bounding boxes above the threshold overlaid
        """
        results = self.process_results(results)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Get the score list
        scores = [score for _, score, _ in results]
        # Return the ranking from the most confidence to the least confidence
        sorted_scores = sorted(scores)
        rank_dict = {
            value: len(scores) - index for index, value in enumerate(sorted_scores)
        }
        ranks = [rank_dict[element] for element in scores]

        idx = 0
        for label, score, box in results:
            img = cv2.rectangle(img, box[:2], box[2:], (255, 0, 0), 5)
            if box[3] + 25 > 768:
                y = box[3] - 10
            else:
                y = box[3] + 25

            color = (255, 0, 0)
            if ranks[idx] == 1:
                color = (0, 255, 0)

            img = cv2.putText(
                img,
                f"{ranks[idx]}:{label}",
                (box[0], y),
                font,
                1,
                color,
                2,
                cv2.LINE_AA,
            )
            idx += 1

        return img

    def update_label(self, labels: List[List[str]]):
        """Update labels that need to be detected

        New labels should be in the format [[label1, label2, ...]]
        """
        labels_with_prefix = [[f"{self.prefix} {label}" for label in labels[0]]]
        self.labels = labels_with_prefix

    def process(self, img, return_image: bool = False):
        """Interface method for compatibility with Aria data-loaders"""
        if return_image:
            return self.run_inference_and_return_img(img)
        else:
            return self.run_inference(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="./input_image.jpg",
    )
    parser.add_argument("--score_threshold", type=float, default=0.1)
    parser.add_argument("--show_img", type=bool, default=True)
    parser.add_argument(
        "--labels",
        type=list,
        default=[
            [
                "lion plush",
                "penguin plush",
                "teddy bear",
                "bear plush",
                "caterpilar plush",
                "ball plush",
                "rubiks cube",
            ]
        ],
    )
    args = parser.parse_args()

    file = args.file
    img = cv2.imread(file)

    V = OwlVit(args.labels, args.score_threshold, args.show_img)
    results = V.run_inference(img)
    # Keep the window open for 10 seconds
    time.sleep(10)
