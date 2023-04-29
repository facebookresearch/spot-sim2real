import argparse
import glob
import os
import os.path as osp
import time

import cv2
import detectron2.data.transforms as T
import tqdm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.catalog import Metadata
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer

R_101_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
R_50_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
THIS_DIR = osp.dirname(osp.abspath(__file__))
CLASSES_TXT = osp.join(THIS_DIR, "classes.txt")

IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
VIDEO_EXTENSIONS = ["mp4", "mov"]


class MaskRcnnInference:
    def __init__(
        self,
        weights_file,
        coco_json=None,
        min_size_test=800,
        score_thresh=0.7,
        config_path="101",
        device=None,
    ):
        self.weights_file = weights_file
        self.coco_json = coco_json
        self.score_thresh = score_thresh

        self.metadata = self.generate_metadata()
        if config_path == "101":
            config_path = R_101_CONFIG
        elif config_path == "50":
            config_path = R_50_CONFIG
        self.cfg = self.generate_cfg(config_path)
        if device is not None:
            self.cfg.MODEL.DEVICE = device
        print("Loading predictor...")
        self.predictor = DefaultPredictor(self.cfg)
        print("Predictor loaded.")

        # Get resize transform (stolen from detectron2 COCOEvaluator)
        self.resize_edge = T.ResizeShortestEdge(
            short_edge_length=min_size_test, max_size=1333
        )

    def generate_metadata(self):
        if not osp.isfile(CLASSES_TXT):
            assert self.coco_json is not None, (
                f"{CLASSES_TXT} does not exist; locate COCO format train dataset json "
                "and re-run with -j command line arg to generate it automatically."
            )
            print("Parsing JSON file to determine classes list...")
            dataset_name = "test"
            register_coco_instances(dataset_name, {}, self.coco_json, "")
            DatasetCatalog.get(dataset_name)
            metadata = MetadataCatalog.get(dataset_name)
            classes_list = "\n".join(metadata.thing_classes)
            with open(CLASSES_TXT, "w") as f:
                f.write(classes_list)
            print(f"Classes have been recorded in {CLASSES_TXT}.")

        metadata = Metadata()
        with open(CLASSES_TXT) as f:
            thing_classes = f.read().splitlines()
        print(f"Classes have been read from {CLASSES_TXT}.")
        metadata.set(thing_classes=thing_classes)

        return metadata

    def generate_cfg(self, config_path):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_path))
        cfg.MODEL.WEIGHTS = self.weights_file
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_thresh
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.metadata.thing_classes)
        return cfg

    def resize_img(self, img):
        resize_transform = self.resize_edge.get_transform(img)
        return resize_transform.apply_image(img)

    def inference(self, img, get_pred_time=False):
        if get_pred_time:
            start_time = time.time()
        outputs = self.predictor(img)
        if get_pred_time:
            return outputs, time.time() - start_time
        return outputs

    def visualize_inference(self, img, outputs=None, get_pred_time=False):
        if outputs is None:
            img = self.resize_img(img)
            outputs = self.inference(img, get_pred_time=get_pred_time)
            if get_pred_time:
                outputs, pred_time = outputs
        v = Visualizer(img[:, :, ::-1], metadata=self.metadata)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2BGR)

        if get_pred_time:
            return img, pred_time

        return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_path", help="path to trained weights")
    parser.add_argument("test_data", help="path to image, dir of images, or video")
    parser.add_argument("-j", "--train_coco_json", default=None)
    parser.add_argument(
        "-o",
        "--output_dir",
        help="where visualized inferences are saved (default: test_output)",
        default="test_output",
    )
    parser.add_argument("--shrink", action="store_true")
    parser.add_argument("--normal", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    weights_path = args.weights_path
    test_data = args.test_data
    train_coco_json = args.train_coco_json
    output_dir = args.output_dir

    assert osp.isdir(test_data) or osp.isfile(test_data), f"{test_data} does not exist!"
    os.makedirs(output_dir, exist_ok=True)

    config_path = R_50_CONFIG if args.fast else R_101_CONFIG
    if args.normal:
        min_size_test = 480
    elif args.shrink:
        min_size_test = 240
    else:
        min_size_test = None
    mask_rcnn = MaskRcnnInference(
        weights_path,
        train_coco_json,
        min_size_test=min_size_test,
        config_path=config_path,
    )

    # Determine if input path is an image, directory, or video
    if osp.isdir(test_data):
        all_files = glob.glob(osp.join(test_data, "*"))
        img_paths = [
            p for p in all_files if p.split(".")[-1].lower() in IMAGE_EXTENSIONS
        ]
    elif test_data.split(".")[-1].lower() in IMAGE_EXTENSIONS:
        img_paths = [test_data]
    elif test_data.split(".")[-1].lower() in VIDEO_EXTENSIONS:
        img_paths = None  # denotes that test_data is a video
    else:
        raise Exception(f"{test_data} is not a valid file for inference!")

    if img_paths is not None:
        print(f"Starting inference on {len(img_paths)} images...")
        for idx, img_path in enumerate(tqdm.tqdm(img_paths)):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vis_img = mask_rcnn.visualize_inference(img)
            output_path = osp.join(output_dir, f"{idx:04}.png")
            cv2.imwrite(output_path, vis_img)
    else:
        vid = cv2.VideoCapture(test_data)
        total_num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Starting inference on input video...")
        out_vid = None
        pbar = tqdm.trange(total_num_frames)
        for idx in pbar:
            _, frame = vid.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = frame
            vis_img, pred_time = mask_rcnn.visualize_inference(img, get_pred_time=True)

            # Add pred_time to progress bar
            pbar.set_postfix(pred_time=round(pred_time, 4))

            # Use first frame to set up output video
            if idx == 0:
                weights_dir = osp.dirname(osp.abspath(weights_path))
                out_vid_path = osp.join(weights_dir, osp.basename(test_data))
                four_cc = cv2.VideoWriter_fourcc(*"MP4V")
                fps = vid.get(cv2.CAP_PROP_FPS)
                height, width = vis_img.shape[:2]
                out_vid = cv2.VideoWriter(out_vid_path, four_cc, fps, (width, height))
            out_vid.write(vis_img)
        print(f"Output video was saved to {out_vid_path}")


if __name__ == "__main__":
    main()
