import os
token = os.environ['HF_TOKEN']
import torch
from transformers import AutoTokenizer, AutoModel
from decord import VideoReader, cpu
from PIL import Image
import numpy as np
import decord
from decord import VideoReader, cpu
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import time
decord.bridge.set_bridge("torch")

class InternVideo2():
    def __init__(self):
        # Load the tokenizer and model
        self.tokenizer =  AutoTokenizer.from_pretrained('OpenGVLab/InternVideo2-Chat-8B', trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            'OpenGVLab/InternVideo2-Chat-8B',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True).cuda()

    def _get_index(self, num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def load_video(self, video_path, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        frame_indices = self._get_index(num_frames, num_segments)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean, std)
        ])

        frames = vr.get_batch(frame_indices)
        frames = frames.permute(0, 3, 1, 2)
        frames = transform(frames)

        T_, C, H, W = frames.shape
            
        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            # " " should be added in the start and end
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return frames, msg
        else:
            return frames


if __name__ == "__main__":
    IV2 = InternVideo2()
    num_segments = 8
    for video_path in ["hand_interaction_with_can.mp4", "hand_interaction_with_bottle.mp4", "hand_interaction_with_cup.mp4", "hand_interaction_with_toy_plush.mp4"]:
        
        print("==========current video=========")
        print(f"video_path: {video_path}")

        # sample uniformly 8 frames from the video
        video_tensor = IV2.load_video(video_path, num_segments=num_segments, return_msg=False)
        video_tensor = video_tensor.to(IV2.model.device)

        chat_history= []
        start_time = time.time()
        response, chat_history = IV2.model.chat(IV2.tokenizer, '', \
            'describe the action step by step.', media_type='video', media_tensor=video_tensor, \
            chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
        print("1. response:", response)

        response, chat_history = IV2.model.chat(IV2.tokenizer, '', \
            'what is object that human is holding?', media_type='video', media_tensor=video_tensor, \
            chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
        print("2. response:", response)

        response, chat_history = IV2.model.chat(IV2.tokenizer, '', \
            'where does that object that human is holding move from, and move to?', media_type='video', \
            media_tensor=video_tensor, chat_history= chat_history, return_history=True,generation_config={'do_sample':False})
        print("3. response:", response)

        end_time = time.time()
        print(f"Avg time to process videos: {(end_time-start_time)/3.0} s")