import cv2
import numpy as np
import torch
import zmq
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "/home/jmmy/research/segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

class Track():
    def __init__(self):
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    def init_state(self, imgs):
        self.inference_state = self.predictor.init_state_with_images(imgs)
        self.predictor.reset_state(self.inference_state)

    def add_bbox(self, bbox):
        # a box at (x_min, y_min, x_max, y_max) to get started
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=0,
            obj_id=4,
            box=bbox,
        )
    
    def track(self):
        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            pass
            # return out_mask

def connect_socket(port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")
    print(f"Socket Connected at {port}")
    return socket

def segment_with_socket(rgb_image, bbox, port=21001):
    socket = connect_socket(port)
    socket.send_pyobj((rgb_image, bbox))
    return socket.recv_pyobj()

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
    images = (images.permute(0,2,3,1).cpu().numpy()*255).astype('uint8')
    # images with the shape of torch.Size([160, 3, 1024, 1024])
    sam2.init_state(images)




    breakpoint()


    port =  "21001"
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    print(f"Segmentation Server Listening on port {port}")

    while True:
        """A service for running segmentation service, send request using zmq socket"""
        img, bbox = socket.recv_pyobj()
        print("Recieved img for Segmentation")
        masks = sam2.detect(img)
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
