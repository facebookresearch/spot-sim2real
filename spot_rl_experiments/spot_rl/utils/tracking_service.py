import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import zmq
from PIL import Image

try:
    from sam2.build_sam import build_sam2_video_predictor
except Exception:
    print("Do not import sam2 in the main loop. sam2 needs sam2 conda env")

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# SAM2_CKPT = "segment-anything-2/checkpoints/sam2_hiera_large.pt"
SAM2_CKPT = None
for root, dirs, files in os.walk("/home/"):
    if "sam2_hiera_large.pt" in files:
        SAM2_CKPT = os.path.join(root, "sam2_hiera_large.pt")
        break

if SAM2_CKPT is None:
    print("Cannot import sam2. Please provide sam2 checkpoint")
    raise Exception("Cannot import sam2. Please provide sam2 checkpoint")
MODEL_CFG = "sam2_hiera_l.yaml"


class Track:
    def __init__(self):
        self.predictor = build_sam2_video_predictor(MODEL_CFG, SAM2_CKPT)
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
            obj_id=1,
            box=np.array(bbox, dtype=np.float32),
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

        # Get the final mask
        _, out_mask = list(self.video_segments[out_frame_idx])[-1]
        if np.sum(out_mask) != 0:
            xmax, ymax = np.max(np.where(out_mask[0] == 1), 1)
            xmin, ymin = np.min(np.where(out_mask[0] == 1), 1)
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
            plt.show()


def connect_socket(port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")
    return socket


def tracking_with_socket(images, bbox, port=21002):
    # images with the size of [2, H, W, 3]
    # bbox with the format of [x_min, y_min, x_max, y_max]
    socket = connect_socket(port)
    socket.send_pyobj((images, bbox))
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
    port = "21002"
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    print(f"Tracking Server Listening on port {port}")

    sam2 = Track()
    while True:
        """A service for running tracking service, send request using zmq socket"""
        images, bbox = socket.recv_pyobj()
        start_time = time.time()
        sam2.init_state(images)
        cur_bbox = bbox
        sam2.add_bbox(cur_bbox)
        new_bbox = sam2.track()
        # sam2.vis() # debug
        socket.send_pyobj(new_bbox)

# Debug code
# sam2 = Track()

# # Load the images
# from sam2.utils.misc import load_video_frames_light

# images, video_height, video_width = load_video_frames_light(
#     video_path="/home/jmmy/research/segment-anything-2/notebooks/videos/handle",
#     image_size=sam2.predictor.image_size,
#     offload_video_to_cpu=False,
#     async_loading_frames=False,
# )
# images = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype("uint8")
# images = images[80:, :, :, :]

# cur_bbox = [200, 340, 300, 350]  # (x_min, y_min, x_max, y_max)
# # Loop over images
# for i in range(len(images) - 1):
#     # images with the shape of torch.Size([# of frames, 1024, 1024, 3])
#     # Init the predictor
#     start_time = time.time()
#     sam2.init_state(images[i : i + 2])
#     # Add the bbox for the first frame
#     sam2.add_bbox(cur_bbox)
#     # Track the object
#     try:
#         raw_cur_bbox = sam2.track()
#         print("time taken to track one frame:", time.time() - start_time, "sec")
#         sam2.vis()
#     except Exception as e:
#         # Visualize the results
#         sam2.vis()
#     # For img coordinate
#     cur_bbox = [raw_cur_bbox[1], raw_cur_bbox[0], raw_cur_bbox[3], raw_cur_bbox[2]]
