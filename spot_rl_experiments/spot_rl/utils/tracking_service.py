import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import zmq
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


sam2_checkpoint = (
    "/home/jmmy/research/segment-anything-2/checkpoints/sam2_hiera_large.pt"
)
model_cfg = "sam2_hiera_l.yaml"


class Track:
    def __init__(self):
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        self.cur_imgs = None
        self.video_segments = {}

    def init_state(self, imgs):
        self.cur_imgs = imgs
        self.inference_state = self.predictor.init_state_with_images(imgs)
        self.reset_state()

    def reset_state(self):
        self.predictor.reset_state(self.inference_state)

    def add_bbox(self, bbox):
        # a box at (x_min, y_min, x_max, y_max) to get started
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=0,
            obj_id=0,
            box=bbox,
        )

    def track(self):
        # run propagation throughout the video and collect the results in a dict
        self.video_segments = (
            {}
        )  # video_segments contains the per-frame segmentation results
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        final_masks = None
        for out_obj_id, out_mask in self.video_segments[out_frame_idx].items():
            print("size of mask", out_mask.shape)
            xmax, ymax = np.max(np.where(out_mask == 1), 1)
            xmin, ymin = np.min(np.where(out_mask == 1), 1)
            final_masks = [xmin, ymin, xmax, ymax]
        return final_masks

    def vis(
        self,
    ):
        # render the segmentation results every few frames
        frame_names = ["prev_frame", "cur_frame"]
        plt.close("all")
        for out_frame_idx in range(0, len(frame_names)):
            plt.figure(figsize=(6, 4))
            plt.title(f"{out_frame_idx}")
            plt.imshow(self.cur_imgs[out_frame_idx])
            for out_obj_id, out_mask in self.video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)


def connect_socket(port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")
    print(f"Socket Connected at {port}")
    return socket


def tracking_with_socket(rgb_image, bbox, port=21001):
    socket = connect_socket(port)
    socket.send_pyobj((rgb_image, bbox))
    return socket.recv_pyobj()


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


if __name__ == "__main__":
    device = "cuda"

    sam2 = Track()

    # Load the images
    from sam2.utils.misc import load_video_frames_light

    images, video_height, video_width = load_video_frames_light(
        video_path="/home/jmmy/research/segment-anything-2/notebooks/videos/handle",
        image_size=sam2.predictor.image_size,
        offload_video_to_cpu=False,
        async_loading_frames=False,
    )
    images = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
    images = images[80:, :, :, :]

    cur_bbox = [300, 0, 500, 400]  # (x_min, y_min, x_max, y_max)

    # Loop over images
    for i in range(len(images) - 1):
        # images with the shape of torch.Size([# of frames, 1024, 1024, 3])
        # Init the predictor
        start_time = time.time()
        sam2.init_state(images[i : i + 2])
        # Add the bbox for the first frame
        sam2.add_bbox(cur_bbox)
        # Track the object
        raw_cur_bbox = sam2.track()
        cur_bbox = [raw_cur_bbox[1], raw_cur_bbox[0], raw_cur_bbox[3], raw_cur_bbox[2]]
        print("time taken to track one frame:", time.time() - start_time, "sec")
        # Visualize the results
        sam2.vis()
        breakpoint()

    breakpoint()

    port = "21001"
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    print(f"Segmentation Server Listening on port {port}")

    while True:
        """A service for running segmentation service, send request using zmq socket"""
        img, bbox = socket.recv_pyobj()
        print("Recieved img for Segmentation")
        masks = sam2.track()
        mask = masks[0, 0].cpu().numpy()  # hxw, bool
        socket.send_pyobj(mask)

# If you want to use detection service, then use the following code to listen to socket
# def detect_with_socket(img, object_name, thresh=0.01, device="cuda"):
#     """Fetch the detection result"""
#     subprocess.Popen("spot_rl_detection_service", shell=True)
#     port = 21001
#     context = zmq.Context()
#     socket = context.socket(zmq.REQ)
#     socket.connect(f"tcp://localhost:{port}")
#     print(f"Socket Connected at {port}")
#     socket.send_pyobj((img, object_name, thresh, device))
#     return socket.recv_pyobj()
