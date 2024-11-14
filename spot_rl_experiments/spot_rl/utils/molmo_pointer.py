from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import re
import base64
from mimetypes import guess_type
from spot_rl.utils.robopoint_pointer import analyze_coordinates


def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def load_molmo_model(model_path, large=False):
    # model_name = "Molmo-72B-0924" if large else "Molmo-7B-D-0924"
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto",
    )
    return model, processor


def molmo_predict_waypoint(
    image,
    prompt,
    model,
    processor,
) -> Tuple[str, str, List[Tuple[float, float]]]:
    # image = Image.open(image_path)

    # process the image and text
    inputs = processor.process(images=[image], text=prompt)

    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(
            max_new_tokens=200, stop_strings="<|endoftext|>", do_sample=False
        ),
        tokenizer=processor.tokenizer,
    )

    # only get generated tokens; decode them to text
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )
    # generated_text = generated_text.replace("<|eot_id|>", "")
    # Parse out the points from the generated text using regex
    points = []
    for match in re.finditer(
        r'<point x="([0-9.]+)" y="([0-9.]+)" alt="([^"]+)">([^<]+)</point>',
        generated_text,
    ):
        x, y, alt, text = match.groups()
        points.append((float(x) / 100, float(y) / 100))

    filtered, average = analyze_coordinates(points)
    return average
