from typing import Optional
import argparse

import time
import cv2
import numpy as np
import torch

from deblur_gan.aug import get_normalize
from deblur_gan.models.networks import get_generator


class DeblurGANv2:
    def __init__(self, weights_path: str, model_name: str = 'fpn_mobilenet'):
        model = get_generator(
            {
                "g_name": model_name,
                "norm_layer": "instance"
            }
        )
        sd = torch.load(weights_path)['model']
        model.load_state_dict(
            {
                k[len("module."):]: v for k, v in sd.items()
            }
        )
        self.model = torch.jit.script(model.cuda())
        # self.model = model.cuda()
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        x = np.pad(x, mode='constant', constant_values=0, pad_width=((0, min_height - h), (0, min_width - w), (0, 0)))
        mask = np.pad(mask, mode='constant', constant_values=0, pad_width=((0, min_height - h), (0, min_width - w), (0, 0)))

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray]=None, ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        inputs = [img.cuda()]
        if not ignore_mask:
            inputs += [mask]
        with torch.no_grad():
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('weights_path')
    parser.add_argument('img_path')
    args = parser.parse_args()
    img = cv2.imread(args.img_path, cv2.COLOR_BGR2RGB)
    predictor = DeblurGANv2(weights_path=args.weights_path)

    for _ in range(100):
        st = time.time()
        pred = predictor(img)
        print(time.time() - st)

    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    cv2.imwrite('output.png', pred)


if __name__ == '__main__':
    main()
