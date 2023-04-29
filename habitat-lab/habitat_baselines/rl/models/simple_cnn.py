import time
from typing import Dict

import cv2
import numpy as np
import torch
from torch import nn as nn

RGB_KEYS = ["rgb", "arm_rgb", "3rd_rgb"]
ARM_VISION_KEYS = ["arm_depth", "arm_rgb", "arm_depth_bbox"]
HEAD_VISION_KEYS = [
    "depth",
    "rgb",
    "spot_left_depth",
    "spot_right_depth",
    "spot_left_rgb",
    "spot_right_rgb",
]
DEPTH_KEYS = [i for i in ARM_VISION_KEYS + HEAD_VISION_KEYS if "depth" in i]

DEBUGGING = False


def reject_obs_key(key, head_only, arm_only):
    assert not head_only or not arm_only  # Can't be both head and arm only
    return (
        head_only
        and key in ARM_VISION_KEYS
        or arm_only
        and key in HEAD_VISION_KEYS
    )


class SimpleCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer

    Takes in observations and produces an embedding of the rgb and/or depth
    components

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(
        self,
        observation_space,
        output_size,
        force_blind,
        head_only=False,
        arm_only=False,
        init=True,
    ):
        super().__init__()

        self.output_size = output_size
        if force_blind:
            self.cnn = nn.Sequential()
            self._n_input_rgb = 0
            self._n_input_depth = 0
            return
        self.use_rgb_keys = []
        self.use_depth_keys = []

        rgb_shape = None
        self._n_input_rgb = 0
        # HACK: Never use RGB for policies.
        # for k in RGB_KEYS:
        #     if k in observation_space.spaces and not reject_obs_key(
        #         k, head_only, arm_only
        #     ):
        #         self._n_input_rgb += 3
        #         rgb_shape = observation_space.spaces[k].shape[:2]
        #         self.use_rgb_keys.append(k)

        # Ensure both the single camera AND two camera setup is NOT being used
        self.using_one_camera = "depth" in observation_space.spaces
        self.using_two_cameras = (
            "spot_left_depth" in observation_space.spaces
            or "spot_right_depth" in observation_space.spaces
        )
        assert not (self.using_one_camera and self.using_two_cameras)

        depth_shape = None
        self._n_input_depth = 0
        for k in DEPTH_KEYS:
            if k in observation_space.spaces and not reject_obs_key(
                k, head_only, arm_only
            ):
                self._n_input_depth += 1
                depth_shape = observation_space.spaces[k].shape[:2]
                self.use_depth_keys.append(k)

        if self.using_two_cameras:
            # Ensure both eyes are being used if at all
            eyes = ["spot_left_depth", "spot_right_depth"]
            assert all([i in observation_space.spaces for i in eyes])
            # Revert num_channels from 2 to 1
            self._n_input_depth = 1
            # Make depth_shape twice as wide
            height, width = depth_shape
            depth_shape = np.array([height, width * 2], dtype=np.float32)

        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]

        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        if self._n_input_rgb > 0:
            cnn_dims = np.array(rgb_shape, dtype=np.float32)
        elif self._n_input_depth > 0:
            cnn_dims = np.array(depth_shape, dtype=np.float32)

        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            for kernel_size, stride in zip(
                self._cnn_layers_kernel_size, self._cnn_layers_stride
            ):
                cnn_dims = self._conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32),
                )

            self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=self._n_input_rgb + self._n_input_depth,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ),
                nn.ReLU(True),
                nn.Conv2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ),
                #  nn.ReLU(True),
                nn.Flatten(),
                nn.Linear(32 * cnn_dims[0] * cnn_dims[1], output_size),
                nn.ReLU(True),
            )

        self.layer_init(init)
        self.count = 0
        self.debug_prefix = f"{time.time() * 1e7:.0f}"[-5:]

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def layer_init(self, init=True):
        if not init:
            return
        for layer in self.cnn:  # type: ignore
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations: Dict[str, torch.Tensor]):
        cnn_input = []
        for k in self.use_rgb_keys:
            if k in observations:
                rgb_observations = observations[k]
                # permute tensor to [BATCH x CHANNEL x HEIGHT X WIDTH]
                rgb_observations = rgb_observations.permute(0, 3, 1, 2)
                rgb_observations = (
                    rgb_observations.float() / 255.0
                )  # normalize RGB
                cnn_input.append(rgb_observations)
        using_arm_depth = False
        depth_observations = []
        if self.using_two_cameras:
            depth_observations.append(
                torch.cat(
                    [
                        # Spot is cross-eyed; right is on the left on the FOV
                        observations["spot_right_depth"],
                        observations["spot_left_depth"],
                    ],
                    dim=2,
                )
            )
        else:
            for k in self.use_depth_keys:
                if k in observations:
                    depth_observations.append(observations[k])
                    if k in ARM_VISION_KEYS:
                        using_arm_depth = True

        # Save images to disk for debugging
        if DEBUGGING:
            img = None
            for orig_img in cnn_input + depth_observations:
                h, w, c = orig_img.shape[1:]
                for c_idx in range(c):
                    new_img = orig_img[0][:, :, c_idx].cpu().numpy()
                    if img is None:
                        img = new_img
                    else:
                        img = np.hstack([img, new_img])
            img = (img * 255).astype(np.uint8)
            out_path = f"{self.debug_prefix}_{self.count:04}.png"
            cv2.imwrite(out_path, img)
            print("Saved visual observations to", out_path)
            self.count += 1
        # permute tensors to [BATCH x CHANNEL x HEIGHT X WIDTH]
        cnn_input.extend([d.permute(0, 3, 1, 2) for d in depth_observations])
        cnn_inputs = torch.cat(cnn_input, dim=1)

        # Just return all zeros without using CNN if gripper images are black
        if using_arm_depth and torch.count_nonzero(cnn_inputs).item() == 0:
            num_envs = cnn_inputs.shape[0]
            visual_features = torch.zeros(
                num_envs, self.output_size, device=cnn_inputs.device
            )
        else:
            # [BATCH x OUTPUT_SIZE (512)]
            visual_features = self.cnn(cnn_inputs)

            if using_arm_depth:
                # Mask out visual features where corresponding image was zeros
                non_zero_idxs = torch.nonzero(torch.sum(cnn_inputs, (1, 2, 3)))
                visual_features_mask = torch.zeros_like(visual_features)
                visual_features_mask[non_zero_idxs] = 1.0

                visual_features = visual_features * visual_features_mask

        if DEBUGGING:
            print(
                f"[simple_cnn.py]: Sum of vis feats ({self.count}):",
                visual_features.sum(),
            )

        return visual_features
