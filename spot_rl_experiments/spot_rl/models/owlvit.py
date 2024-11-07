# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# mypy: ignore-errors
import argparse
import time
from typing import List

import cv2
import torch
from PIL import Image
from torchvision.ops import box_area, box_iou

MEGRE_BBOX = False


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

    def run_inference(self, img):
        """
        img: an open cv image in (H, W, C) format
        """
        # Process inputs
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
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.score_threshold,
            )

        if self.show_img:
            self.show_img_with_overlaid_bounding_boxes(img, results)

        return self.get_most_confident_bounding_box_per_label(results)

    def run_inference_and_return_img(
        self, img, vis_img_required=True, multi_objects_per_label=False
    ):
        """
        img: an open cv image in (H, W, C) format
        multi_objects_per_label: if want to return multiple detections per label
        """
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
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.score_threshold,
            )

        return (
            self.get_confident_bounding_box_per_label(results)
            if multi_objects_per_label
            else self.get_most_confident_bounding_box_per_label(results),
            self.create_img_with_bounding_box_no_ranking(
                img, results, multi_objects_per_label
            )
            if vis_img_required
            else None,
        )

    def show_img_with_overlaid_bounding_boxes(self, img, results):
        img = self.create_img_with_bounding_box_no_ranking(img, results)
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

    def get_confident_bounding_box_per_label(self, results, merge_thresh=2.0):
        """
        Returns the confident bounding box for each label above the threshold.
        Each label could have multiple detections.
        """
        boxes, scores, labels = (
            results[0]["boxes"],
            results[0]["scores"],
            results[0]["labels"],
        )
        boxes = boxes.to("cpu")
        labels = labels.to("cpu")
        scores = scores.to("cpu")

        # Merging bounding boxes logic goes here
        if MEGRE_BBOX:
            areas = box_area(boxes)
            ious = box_iou(boxes, boxes).fill_diagonal_(0.0)
            iou_gtr_than_thresh = torch.argwhere(torch.triu(ious) > merge_thresh)
            print(iou_gtr_than_thresh)
            merge_record = {}
            for index_pair in iou_gtr_than_thresh:
                i, j = index_pair
                i, j = i.item(), j.item()
                if areas[i] > areas[j]:  # i has more area, thus suppress j
                    scores[j] = -1
                    merge_record[j] = i if i not in merge_record else merge_record[i]

                if areas[j] > areas[i]:  # j has more area, thus suppress i
                    scores[i] = -1
                    merge_record[i] = j if j not in merge_record else merge_record[j]

        # Initialize dictionaries to store most confident bounding boxes and scores per label
        target_boxes = {}
        target_scores = {}

        for box, score, label in zip(boxes, scores, labels):
            box = [round(i, 2) for i in box.tolist()]
            w, h = box[0] - box[2], box[1] - box[3]
            min_dim = min(w, h)
            if score >= self.score_threshold and min_dim <= 20.0:
                if label.item() not in target_scores:
                    target_scores[label.item()] = [score.item()]
                    target_boxes[label.item()] = [box]
                else:
                    target_scores[label.item()].append(score.item())
                    target_boxes[label.item()].append(box)

        # Format the output
        result = []
        for label, boxs in target_boxes.items():
            for i, box in enumerate(boxs):
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                # Strip the prefix from the label
                label_without_prefix = self.labels[0][label][len(self.prefix) + 1 :]

                # For computing area
                # w = x2 -x1 + 1
                # h = y2 - y1+ 1
                # area = w*h
                # print(f"Class {label_without_prefix}, score {target_scores[label][i]}, area : {area}, w {w}, h {h}")

                result.append(
                    [label_without_prefix, target_scores[label][i], [x1, y1, x2, y2]]
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
            img = cv2.putText(
                img,
                f"{ranks[idx]}:{label}",
                (box[0], y),
                font,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            idx += 1

        return img

    def create_img_with_bounding_box_no_ranking(
        self, img, results, multi_objects_per_label=False
    ):
        """
        Returns an image with all bounding boxes above the threshold overlaid.
        Each class has only one bounding box.
        """

        results = (
            self.get_most_confident_bounding_box_per_label(results)
            if not multi_objects_per_label
            else self.get_confident_bounding_box_per_label(results)
        )
        font = cv2.FONT_HERSHEY_SIMPLEX

        for label, score, box in results:
            img = cv2.rectangle(img, box[:2], box[2:], (255, 0, 0), 5)
            if box[3] + 25 > 768:
                y = box[3] - 10
            else:
                y = box[3] + 25
            img = cv2.putText(
                img, label, (box[0], y), font, 1, (255, 0, 0), 2, cv2.LINE_AA
            )

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

    V = OwlVit(args.labels, args.score_threshold, args.show_img, version=2)
    results = V.run_inference(img)
    # Keep the window open for 10 seconds
    time.sleep(10)
