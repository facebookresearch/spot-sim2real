import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import zmq
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
from ultralytics import YOLOWorld


def load_yolo_model():
    # Load a pretrained YOLOv8s-worldv2 model
    model = YOLOWorld("yolov8x-worldv2.pt").cuda()
    return model


# Set up ZMQ server to receive images and data
def zmq_server(port=21001):
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP socket to listen
    socket.bind(f"tcp://*:{port}")
    time.sleep(2)
    print(f"Yolo world Server Listening on port {port}")
    return socket


def load_clip():
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32", device_map="cuda")
    processor = AutoProcessor.from_pretrained(
        "openai/clip-vit-base-patch32", device_map="cuda"
    )
    return model, processor


def crop_bboxes(image, detections):
    # Assuming detections contain bounding box coordinates
    cropped_images = []
    for det in detections:
        x1, y1, x2, y2 = det  # Adjust this based on actual FastSAM output format
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_image = image[y1:y2, x1:x2][..., ::-1]
        cropped_images.append(cropped_image)
    return cropped_images


def get_embeddings(model, processor, image_inputs, text_inputs):
    # Get embeddings for either images or texts
    inputs = processor(
        text=text_inputs, images=image_inputs, return_tensors="pt", padding=True
    )

    with torch.no_grad():
        inputs = inputs.to("cuda")
        with torch.autocast("cuda"):
            outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image  # .softmax(dim=1)
        score_per_label, label_per_detection = (
            torch.max(probs, dim=1).values.cpu().numpy().astype(float).tolist(),
            torch.max(probs, dim=1).indices.cpu().numpy().astype(int).tolist(),
        )

    return [text_inputs[l] for i, l in enumerate(label_per_detection)], score_per_label


def draw_bboxes(image, detections, matched_classes, scores):
    short_class_map = {
        "blue ball": "bb",
        "bulldozer toy car": "bull",
        "can": "can",
        "bottle": "bo",
        "avocado plush toy": "av",
        "green frog plush toy": "fr",
        "pink donut plush toy": "do",
        "yellow pineapple plush toy": "pi",
        "green caterpillar plush toy": "cat",
        "black penguin plush toy": "pe",
    }
    for i, (det, score, class_name) in enumerate(
        zip(detections, scores, matched_classes)
    ):
        x1, y1, x2, y2 = det  # Adjust based on detection format
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{short_class_map.get(class_name, class_name)} ({score:.2f})"
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Put the label
        cv2.putText(
            image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 1
        )
    # breakpoint()
    return image


if __name__ == "__main__":

    model = load_yolo_model()
    clipmodel, processor = load_clip()
    socket = zmq_server()
    while True:
        """A service for running segmentation service, send request using zmq socket"""
        img, query_classes, visualize = socket.recv_pyobj()
        print(f"Recieved img for Detection; query classes {query_classes}")
        model.set_classes(query_classes)
        if visualize:
            visualization_img = img.copy()
        with torch.no_grad():
            results = model.predict(img, device="cuda", stream=False, conf=0.1)
        bboxes, probs, labels = [], [], []
        if len(results) == 1:  # since we only send 1 image
            r = results[0]
            bboxes = r.boxes.xyxy.cpu().numpy().tolist()
            probs = r.boxes.conf.cpu().numpy().astype(float).tolist()
            clss = r.boxes.cls.cpu().numpy().astype(int).tolist()
            labels = [model.names[clsi] for clsi in clss]

            # Plot results image
            if visualize:
                visualization_img = r.plot(img=visualization_img)
                if len(bboxes) > 0:
                    cropped_images = crop_bboxes(img, bboxes)
                    new_labels, new_scores = get_embeddings(
                        clipmodel, processor, cropped_images, query_classes
                    )
                    labels, probs = new_labels, new_scores
                    visualization_img = draw_bboxes(
                        img.copy(), bboxes, new_labels, new_scores
                    )
                    # new_visualization = np.concatenate((visualization_img, visualization_img_refined), axis=1)
                    # plt.imshow(new_visualization)
                    # plt.show()

        if not visualize:
            visualization_img = None
        socket.send_pyobj((bboxes, probs, labels, visualization_img))
