from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch
from PIL import Image
import time
import cv2
import argparse

class OwlVit():
    def __init__(self, labels, score_threshold, show_img):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.model.eval()
        self.model.to(self.device)

        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

        self.labels = labels
        self.score_threshold = score_threshold
        self.show_img = show_img

    def run_inference(self, img):
        '''
        img: an open cv image in (H, W, C) format
        '''
        # Process inputs
        inputs = self.processor(text=self.labels, images=img, return_tensors="pt")

        #Target image sizes (height, width) to rescale box predictions [batch_size, 2]
        #target_sizes = torch.Tensor([img.size[::-1]]) this is for PIL images
        target_sizes = torch.Tensor([img.shape[:2]])

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs (bounding boxes and class logits) to COCO API
        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)

        if self.show_img:
            self.show_img_with_overlaid_bounding_boxes(img, results)

        return results


    def show_img_with_overlaid_bounding_boxes(self, img, results):

        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        font = cv2.FONT_HERSHEY_SIMPLEX

        for box, score, label in zip(boxes, scores, labels):
            box = [int(i) for i in box.tolist()]

            if score >= self.score_threshold:
                img = cv2.rectangle(img, box[:2], box[2:], (255,0,0), 5)
                if box[3] + 25 > 768:
                    y = box[3] - 10
                else:
                    y = box[3] + 25
                img = cv2.putText(
                    img, self.labels[0][label], (box[0], y), font, 1, (255,0,0), 2, cv2.LINE_AA
                )
        # Show
        cv2.imshow('img', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default='/Users/jimmytyyang/Downloads/i1.jpg')
    parser.add_argument("--score_threshold", type=float, default=0.1)
    parser.add_argument("--show_img", type=bool, default=True)
    parser.add_argument("--labels", type=list, default=[['lion plush', 'penguin plush', 'teddy bear', 'bear plush', 'caterpilar plush', 'ball plush', 'rubiks cube']])
    args = parser.parse_args()

    file = args.file
    img = cv2.imread(file)

    V = OwlVit(args.labels, args.score_threshold, args.show_img)
    results = V.run_inference(img) 
    import pdb; pdb.set_trace()
    # Keep the window open for 10 seconds
    time.sleep(10)