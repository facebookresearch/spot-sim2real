import argparse
import time

import cv2
import numpy as np

from deblur_gan.predictor import DeblurGANv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_path")
    parser.add_argument("img_path")
    args = parser.parse_args()
    img = cv2.imread(args.img_path, cv2.COLOR_BGR2RGB)
    predictor = DeblurGANv2(weights_path=args.weights_path)

    # First inference is always slow; run a random image
    predictor(np.zeros([256, 256, 3]))

    for _ in range(10):
        st = time.time()
        pred = predictor(img)
        print(time.time() - st)

    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    cv2.imwrite("output.png", pred)


if __name__ == "__main__":
    main()
