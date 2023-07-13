import os.path as osp

import cv2
from deblur_gan.predictor import DeblurGANv2
from mask_rcnn_detectron2.inference import MaskRcnnInference


def generate_mrcnn_detections(
    img, scale, mrcnn, grayscale=True, deblurgan=None, return_img=False, stopwatch=None
):
    if scale != 1.0:
        img = cv2.resize(
            img,
            (0, 0),
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    if deblurgan is not None:
        img = deblurgan(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if stopwatch is not None:
            stopwatch.record("deblur_secs")
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    detections = mrcnn.inference(img)
    if stopwatch is not None:
        stopwatch.record("mrcnn_secs")

    if return_img:
        return detections, img

    return detections


def pred2string(pred):
    detections = pred["instances"]
    if len(detections) == 0:
        return "None"

    detection_str = []
    for det_idx in range(len(detections)):
        class_id = detections.pred_classes[det_idx]
        score = detections.scores[det_idx]
        x1, y1, x2, y2 = detections.pred_boxes[det_idx].tensor.squeeze(0)
        det_attrs = [str(i.item()) for i in [class_id, score, x1, y1, x2, y2]]
        detection_str.append(",".join(det_attrs))
    detection_str = ";".join(detection_str)
    return detection_str


def get_mrcnn_model(config):
    mask_rcnn_weights = (
        config.WEIGHTS.MRCNN_50 if config.USE_FPN_R50 else config.WEIGHTS.MRCNN
    )
    mask_rcnn_device = config.DEVICE
    config_path = "50" if config.USE_FPN_R50 else "101"
    mrcnn = MaskRcnnInference(
        mask_rcnn_weights,
        score_thresh=0.7,
        device=mask_rcnn_device,
        config_path=config_path,
    )
    return mrcnn


def get_deblurgan_model(config):
    if config.USE_DEBLURGAN and config.USE_MRCNN:
        weights_path = config.WEIGHTS.DEBLURGAN
        model_name = osp.basename(weights_path).split(".")[0]
        print("Loading DeblurGANv2 with:", weights_path)
        model = DeblurGANv2(weights_path=weights_path, model_name=model_name)
        return model
    return None
