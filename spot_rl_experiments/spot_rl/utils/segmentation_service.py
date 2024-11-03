import cv2
import numpy as np
import torch
import zmq
from segment_anything import SamPredictor, build_sam
from spot_rl.utils.construct_configs import construct_config
from transformers import Owlv2ForObjectDetection, Owlv2Processor

socket = None
model, processor = None, None
sam = None
device = "cuda"
spot_rl_config = construct_config()


def load_model(model_name="owlvit", device="cpu"):
    global model
    global processor
    global sam
    global spot_rl_config
    if model_name == "owlvit":
        print("Loading OwlVit2")
        model = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble"
        ).to(device)
        processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        return model, processor
    if model_name == "sam":
        print("Loading SAM")
        # TODO: load sam weights from config
        checkpoint_path = spot_rl_config.WEIGHTS.SAM
        sam = SamPredictor(build_sam(checkpoint=checkpoint_path).to(device))


def detect(img, text_queries, score_threshold, device, model=None, processor=None):
    if model is None or processor is None:
        load_model("owlvit", device)

    text_queries = text_queries
    text_queries = text_queries.split(",")
    size = max(img.shape[:2])
    target_sizes = torch.Tensor([[size, size]])
    device = model.device
    inputs = processor(text=text_queries, images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    outputs.logits = outputs.logits.cpu()
    outputs.pred_boxes = outputs.pred_boxes.cpu()
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes
    )
    boxes, scores, labels = (
        results[0]["boxes"],
        results[0]["scores"],
        results[0]["labels"],
    )

    result_labels = []
    for box, score, label in zip(boxes, scores, labels):
        box = [int(i) for i in box.tolist()]
        print(box, score)
        if score < score_threshold:
            continue
        result_labels.append((box, text_queries[label.item()], score))
    result_labels.sort(key=lambda x: x[-1], reverse=True)
    return result_labels


def segment(image, boxes, size, device):
    global sam
    if sam is None:
        load_model("sam", device)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam.set_image(image)

    for i in range(boxes.shape[0]):
        boxes[i] = torch.Tensor(boxes[i])

    boxes = torch.tensor(boxes, device=sam.device)

    transformed_boxes = sam.transform.apply_boxes_torch(boxes, image.shape[:2])

    masks, _, _ = sam.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    return masks


socket = None


def connect_socket(port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://localhost:{port}")
    print(f"Socket Connected at {port}")
    return socket


def segment_with_socket(rgb_image, bbox, port=21001):
    global socket
    socket = connect_socket(port) if socket is None else socket
    socket.send_pyobj((rgb_image, bbox))
    return socket.recv_pyobj()


if __name__ == "__main__":
    device = "cuda"
    load_model("sam", device)

    port = spot_rl_config.SEG_PORT  # "21001"
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{port}")
    print(f"Segmentation Server Listening on port {port}")

    while True:
        """A service for running segmentation service, send request using zmq socket"""
        img, bbox = socket.recv_pyobj()
        print("Recieved img for Segmentation")
        masks = segment(img, np.array([bbox]), img.shape[:2], device)
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
