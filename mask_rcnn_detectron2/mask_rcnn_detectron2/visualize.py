import argparse
import cv2
import os
import os.path as osp
import random
import tqdm

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer

VISUALIZATIONS_DIR = "visualizations/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("coco_json_path", help="path to COCO format json file")
    parser.add_argument("images_dir", help="path to directory containing images")
    parser.add_argument(
        "-n",
        "--num_images",
        help="# of visualization images to generate (default: 20)",
        type=int,
        default=20,
    )
    args = parser.parse_args()

    coco_json_path = args.coco_json_path
    images_dir = args.images_dir
    num_vis_images = args.num_images

    # Register dataset
    print("Registering dataset...")
    register_coco_instances(
        "test",
        {},
        coco_json_path,
        images_dir,
    )
    dataset = DatasetCatalog.get("test")
    metadata = MetadataCatalog.get("test")
    print("Done registering.")

    # Create visualization dir if non-existent
    if not osp.isdir(VISUALIZATIONS_DIR):
        os.mkdir(VISUALIZATIONS_DIR)

    print("Generating visualization images...")
    for d in tqdm.tqdm(random.sample(dataset, num_vis_images)):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        img = vis.get_image()[:, :, ::-1]
        basename = osp.basename(d["file_name"])
        filename = osp.join(VISUALIZATIONS_DIR, basename)
        cv2.imwrite(filename, img)

if __name__ == "__main__":
    main()
