# Code contains minimal ports from the YOLOV8 official github repo & the example such that we don't break habitat lab environment,
# Flow explaination - > Export yolov8 model in torchscript with fixed imagesize, this is a wrapper that will preprocess image to that fixed image size then pass through the network,
# post process predictions (nms, scaling, etc)
# No new package is installed, these preprocessing & post processing are done using simple torchvision & pytorch libraries
import time
from typing import Any, List, Tuple

import cv2
import numpy as np
import torch
import torchvision
from spot_wrapper.spot import Spot
from spot_wrapper.spot import SpotCamIds as Cam
from spot_wrapper.spot import image_response_to_cv2, scale_depth_img


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert (
        x.shape[-1] == 4
    ), f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = (
        torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    )  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[
                x[:, 4].argsort(descending=True)[:max_nms]
            ]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        # # Experimental
        # merge = False  # use merge-NMS
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     from .metrics import box_iou
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     redundant = True  # require redundant detections
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


class YOLOV8Predictor:
    """This is a custom wrapper to preprocess images & then postprocess the results of the torchscript exported Yolov8 model in xyxy format
    The preprocess functions & postprocess are minimally exported from the Yolov8 repo such that it doesn't disturb the habitat lab environment
    """

    def __init__(
        self,
        path_to_to_exported_torchscript_model: str = "yolov8x.torchscript",
        imgsz: int = 640,
    ):
        # imgsz is the size with which the yolov8 model was exported
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.imgsz = imgsz
        self.model = torch.jit.load(
            path_to_to_exported_torchscript_model, map_location=self.device
        )
        self.classnames = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush",
        }

    def _preprocess(self, im: np.ndarray) -> Tuple[torch.Tensor, float]:
        not_tensor = not isinstance(im, torch.Tensor)
        assert not_tensor, "Please pass the numpy images only through yolov8 predictor"
        height, width = im.shape[:2]
        max_len = max(im.shape[:2])
        square_im = np.zeros((max_len, max_len, 3), dtype=np.uint8)
        square_im[:height, :width] = im
        square_im = cv2.resize(square_im, (self.imgsz, self.imgsz))
        # required to rescale detections
        scale = max_len / self.imgsz
        square_im = square_im.reshape(1, self.imgsz, self.imgsz, 3)
        # Please confirm whether colors are bgr or rgb
        square_im = square_im[..., ::-1].transpose(
            (0, 3, 1, 2)
        )  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        square_im = np.ascontiguousarray(square_im)  # contiguous

        square_im = torch.from_numpy(square_im).float()
        square_im = square_im.to(self.device)
        square_im /= 255.0  # 0 - 255 to 0.0 - 1.0
        return square_im, scale

    def _postprocess(self, preds: torch.Tensor, scale: float) -> List:
        """Post-processes predictions and returns a list of Results objects."""
        preds = non_max_suppression(preds)[
            0
        ]  # since we are processing image in stream we are processing only 1 image
        preds[:, :4] *= scale
        return preds.cpu().numpy().tolist()

    def __call__(
        self, img: np.ndarray, visualize: bool = False
    ) -> Tuple[List, np.ndarray]:
        img_preprocessed, scale = self._preprocess(img)
        preds = self.model(img_preprocessed)
        preds = self._postprocess(preds, scale)
        if visualize:
            plot = self._generate_plot(img, preds)
            return preds, plot
        return preds, None

    def _generate_plot(
        self, orig_image: np.ndarray, detections: List[List]
    ) -> np.ndarray:
        orig_image_tensor = torch.from_numpy(orig_image).permute(2, 0, 1).contiguous()
        detections = np.array(detections).reshape(-1, 6)
        detections, scores, class_ids = (
            torch.from_numpy(detections[:, :4]),  # type: ignore
            detections[:, 4].tolist(),  # type: ignore
            detections[:, -1].astype(int).tolist(),  # type: ignore
        )
        class_names = [
            "{}({:.2f})".format(self.classnames[class_id], scores[i])
            for i, class_id in enumerate(class_ids)
        ]
        plot_image = torchvision.utils.draw_bounding_boxes(
            orig_image_tensor, detections, colors="red", labels=class_names
        )
        return plot_image.permute(1, 2, 0).contiguous().numpy()


if __name__ == "__main__":
    yolov8predictor = YOLOV8Predictor("yolov8x.torchscript")
    MAX_HAND_DEPTH = 1.7
    spot = Spot("YoloV8Predictor")

    image_sources = [
        # Cam.FRONTRIGHT_DEPTH,
        # Cam.FRONTLEFT_DEPTH,
        Cam.HAND_DEPTH_IN_HAND_COLOR_FRAME,
        Cam.HAND_COLOR,
    ]
    n = 20 * 3
    cam_stream = cv2.VideoWriter(
        "spot_yolov8_3m.avi", cv2.VideoWriter_fourcc(*"MPEG"), 20, (640, 480)
    )
    for i in range(n):
        try:
            image_responses = spot.get_image_responses(image_sources, quality=100)
        except Exception as e:
            print(f"Can't get images from spot due to {e}")
            break
        imgs_list = [image_response_to_cv2(r) for r in image_responses]
        print(imgs_list[-1].shape)
        detections, image_with_detections = yolov8predictor(imgs_list[-1], True)
        cam_stream.write(image_with_detections)
        # cv2.imshow('Yolov8predictor',image_with_detections)

        # Press Q on keyboard to  exit
        # if cv2.waitKey(25) & 0xFF == ord('q'):break
    cam_stream.release()

    # image = cv2.imread("yolov8images/dog_bike_car.jpg")
    # detections, plot = yolov8predictor(image, visualize=True)
    # #print(detections)
    # save_filename = "yolov8images/yolov8_detections.png"
    # cv2.imwrite(save_filename, plot)
    # print(f"Image Written to {save_filename}")
    # cap = cv2.VideoCapture("hand_rgb_record.avi")
    # n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # writecap = cv2.VideoWriter("hand_rgb_record_with_detections.avi", cv2.VideoWriter_fourcc(*'MPEG'), 20, (640, 480))

    # for i in range(n_frames):
    #     ret, frame = cap.read()
    #     # if frame is read correctly ret is True
    #     if not ret: break
    #     #print(frame.shape, frame.dtype)
    #     detections_i, plot_i = yolov8predictor(frame, visualize=True)
    #     writecap.write(plot_i)

    # cap.release()
    # #cv2.destroyAllWindows()
    # writecap.release()
