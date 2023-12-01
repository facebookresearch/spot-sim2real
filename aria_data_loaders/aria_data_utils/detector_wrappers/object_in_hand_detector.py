import os

import model as hand_object_detector
import numpy as np
import torch
from aria_data_utils.detector_wrappers.generic_detector_interface import GenericDetector
from hand_object_detector.faster_rcnn.faster_rcnn import _fasterRCNN
from hand_object_detector.faster_rcnn.resnet import resnet
from hand_object_detector.faster_rcnn.vgg16 import vgg16


class ObjectInHandDetectorWrapper(GenericDetector):
    def __init__(self, net="res101", class_agnostic=False) -> None:
        super().__init__()
        self._model_module_dir = os.path.abspath(hand_object_detector)
        self._model_path = os.path.join(
            self._model_module_dir,
            "models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth",
        )
        self._net = net
        self._class_agnostic = class_agnostic
        self._POOLING_MODE = None

        self.fasterrcnn: _fasterRCNN = None
        if not os.path.exists(self._model_path):
            raise Exception("There is no model file at " + self._model_path)
        pascal_classes = np.asarray(["__background__", "targetobject", "hand"])
        set_cfgs = ["ANCHOR_SCALES", "[8, 16, 32, 64]", "ANCHOR_RATIOS", "[0.5, 1, 2]"]

        self.fasterRCNN = self._setup_network(pascal_classes, set_cfgs)

    def _setup_network(self, pascal_classes, set_cfgs):
        # initilize the network here.
        if self.net == "vgg16":
            fasterRCNN = vgg16(
                pascal_classes, pretrained=False, class_agnostic=self._class_agnostic
            )
        elif self.net == "res101":
            fasterRCNN = resnet(
                pascal_classes,
                101,
                pretrained=False,
                class_agnostic=self._class_agnostic,
            )
        elif self.net == "res50":
            fasterRCNN = resnet(
                pascal_classes,
                50,
                pretrained=False,
                class_agnostic=self._class_agnostic,
            )
        elif self.net == "res152":
            fasterRCNN = resnet(
                pascal_classes,
                152,
                pretrained=False,
                class_agnostic=self._class_agnostic,
            )
        else:
            print("network is not defined")
            breakpoint()

        fasterRCNN.create_architecture()
        checkpoint = torch.load(self._model_path)
        fasterRCNN.load_state_dict(checkpoint["model"])
        if "pooling_mode" in checkpoint.keys():
            self._POOLING_MODE = checkpoint["pooling_mode"]

        fasterRCNN.cuda()
        fasterRCNN.eval()
        return fasterRCNN

    def process_frame(self, frame: np.ndarray):
        """
        100DOH detector fires on the input image and returns:
        #TODO:
        """
        pass


if __name__ == "__main__":
    obj = ObjectInHandDetectorWrapper()
