from typing import Any, Tuple

import torch
from model.faster_rcnn.faster_rcnn import _fasterRCNN
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16


def setup_network(
    pascal_classes, cfgs, model_path, net: str = "vgg16", class_agnostic: bool = False
) -> Tuple[_fasterRCNN, Any]:
    fasterRCNN: _fasterRCNN = None  # type: ignore
    # initilize the network here.
    if net == "vgg16":
        fasterRCNN = vgg16(
            pascal_classes, pretrained=False, class_agnostic=class_agnostic
        )
    elif net == "res101":
        fasterRCNN = resnet(
            pascal_classes,
            101,
            pretrained=False,
            class_agnostic=class_agnostic,
        )
    elif net == "res50":
        fasterRCNN = resnet(
            pascal_classes,
            50,
            pretrained=False,
            class_agnostic=class_agnostic,
        )
    elif net == "res152":
        fasterRCNN = resnet(
            pascal_classes,
            152,
            pretrained=False,
            class_agnostic=class_agnostic,
        )
    else:
        print("network is not defined")
        breakpoint()

    fasterRCNN.create_architecture()
    checkpoint = torch.load(model_path)
    fasterRCNN.load_state_dict(checkpoint["model"])
    if "pooling_mode" in checkpoint.keys():
        POOLING_MODE = checkpoint["pooling_mode"]

    fasterRCNN.cuda()
    fasterRCNN.eval()

    return fasterRCNN, POOLING_MODE
