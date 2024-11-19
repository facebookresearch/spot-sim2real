import os
import re
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from spot_rl.models.robopoint.llava_llama import LlavaLlamaForCausalLM
from transformers import AutoTokenizer

IMAGE_TOKEN_INDEX = -200


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    new_images = []
    for image in images:
        image = expand2square(
            image, tuple(int(x * 255) for x in image_processor.image_mean)
        )
        image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][
            0
        ]
        new_images.append(image)
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def load_pretrained_model(
    model_path,
    device_map="auto",
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}
    kwargs["torch_dtype"] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, **kwargs
    )

    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    vision_tower.load_model(device_map=device_map)
    image_processor = vision_tower.image_processor

    context_len = 2048
    return tokenizer, model, image_processor, context_len


def load_robopoint_model(model_path):
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    model_path = os.path.expanduser(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path)
    return tokenizer, model, image_processor, context_len


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def robopoint_predict_waypoint(
    img_cv2, tokenizer, model, image_processor, temperature=1.0, top_p=0.7
):
    custom_prompt = "Find a few spots within the vacant area on the closest table top surface, away from the edge."
    # custom_prompt = "Find a few spots within the vacant area on the closest table top surface, away from the edge and to the left of the soda can."
    prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> {custom_prompt}. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image. ASSISTANT: "
    print("prompt: ", prompt)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    image = Image.fromarray(img_cv2)
    image_tensor = process_images([image], image_processor, model.config)[0]
    do_sample = True if temperature > 0 else False
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    outputs = parse_coordinates(outputs)
    filtered, average = analyze_coordinates(outputs)
    return average


def parse_coordinates(s):
    # Find all pairs of numbers
    pairs = re.findall(r"([0-9.]+),\s*([0-9.]+)", s)
    # Convert strings to floats
    return [(float(x), float(y)) for x, y in pairs]


def analyze_coordinates(
    coordinates: List[Tuple[float, float]], k: float = 1.5
) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """
    Analyze coordinates by removing outliers and calculating the average position.

    Args:
        coordinates: List of (x, y) coordinate tuples
        k: IQR multiplier for outlier detection (default = 1.5)

    Returns:
        Tuple containing:
        - List of coordinates with outliers removed
        - Tuple of (average_x, average_y) after outlier removal
    """
    if not coordinates:
        return [], (0.0, 0.0)

    # Convert to numpy arrays for easier manipulation
    x_coords = np.array([x for x, _ in coordinates])
    y_coords = np.array([y for _, y in coordinates])

    def get_outlier_mask(data: np.ndarray) -> np.ndarray:
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        return (data >= lower_bound) & (data <= upper_bound)

    # Get masks for both x and y coordinates
    x_mask = get_outlier_mask(x_coords)
    y_mask = get_outlier_mask(y_coords)

    # Combined mask (point is kept if both x and y are not outliers)
    combined_mask = x_mask & y_mask

    # Filter coordinates using the mask
    filtered_coordinates = [
        coord for coord, mask in zip(coordinates, combined_mask) if mask
    ]

    if not filtered_coordinates:
        return [], (0.0, 0.0)

    # Calculate averages
    avg_x = sum(x for x, _ in filtered_coordinates) / len(filtered_coordinates)
    avg_y = sum(y for _, y in filtered_coordinates) / len(filtered_coordinates)

    return filtered_coordinates, (avg_x, avg_y)
