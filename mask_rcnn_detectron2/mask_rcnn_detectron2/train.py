import argparse
import os
import os.path as osp

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


class CocoTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder="coco_eval"):
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, ["bbox"], False, output_folder)


def main():
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_coco_json_path", help="path to train COCO format json file"
    )
    parser.add_argument("val_coco_json_path", help="path to val COCO format json file")
    parser.add_argument(
        "train_images_dir", help="path to train directory containing images"
    )
    parser.add_argument(
        "val_images_dir", help="path to val directory containing images"
    )
    parser.add_argument("-p", "--prefix")
    parser.add_argument("--shrink", action="store_true")
    parser.add_argument("--normal", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    train_images_dir = args.train_images_dir
    val_images_dir = args.val_images_dir
    train_coco_json_path = args.train_coco_json_path
    val_coco_json_path = args.val_coco_json_path

    # Register datasets
    train_dataset_name = osp.splitext(osp.basename(train_coco_json_path))[0] + "_train"
    val_dataset_name = osp.splitext(osp.basename(val_coco_json_path))[0] + "_val"
    register_coco_instances(
        train_dataset_name, {}, train_coco_json_path, train_images_dir
    )
    register_coco_instances(val_dataset_name, {}, val_coco_json_path, val_images_dir)
    train(
        train_dataset_name,
        val_dataset_name,
        args.fast,
        args.prefix,
        args.shrink,
        args.normal,
    )


def train(
    train_dataset_name,
    val_dataset_name,
    fast=False,
    prefix=None,
    shrink=False,
    normal=False,
):
    if fast:
        MODEL_TYPE = "new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ.py"
        CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    else:
        MODEL_TYPE = "new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py"
        CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG))
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_TYPE)  # pretrained weights

    if prefix is not None:
        out_dir = osp.abspath(cfg["OUTPUT_DIR"])
        dirname = osp.dirname(out_dir)
        basename = osp.basename(out_dir)
        new_basename = prefix + "_" + basename
        cfg["OUTPUT_DIR"] = osp.join(dirname, new_basename)

    if shrink:
        cfg.INPUT.MIN_SIZE_TRAIN = (240,)

    if normal:
        cfg.INPUT.MIN_SIZE_TRAIN = (480,)

    # Number of images per batch across all machines.
    cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.SOLVER.MAX_ITER = 1500  # No. of iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    # Set num classes for ROI heads
    DatasetCatalog.get(train_dataset_name)
    metadata = MetadataCatalog.get(train_dataset_name)
    num_classes = len(metadata.thing_classes)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # No. of iterations after which the Validation Set is evaluated.
    cfg.TEST.EVAL_PERIOD = 2500
    cfg.SOLVER.CHECKPOINT_PERIOD = 100
    print("Output dir:", cfg.OUTPUT_DIR)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print("Config:\n", cfg)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
