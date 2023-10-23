import matplotlib.pyplot as plt
import numpy as np


class Memory(object):
    """Memory object to represent a unit of memory"""

    def __init__(
        self,
        name: str = None,
        wearer_location: np.ndarray = None,
        instance_id: int = None,
        category_name: str = None,
        associated_imgs: np.ndarray = None,
        associated_3dbbox: np.ndarray = None,
        seg_mask: np.ndarray = None,
    ) -> None:
        self.wearer_location = wearer_location
        self.name = name
        self.instance_id = instance_id
        self.category_name = category_name
        self.associated_imgs = associated_imgs
        self.associated_3dbbox = associated_3dbbox
        self.seg_mask = seg_mask

    def visualize_objects_with_seg(self, object_idxs):
        """Visualize objects in the images with segmentation masks overlaid"""
        for index in object_idxs:
            print("Visualizing object with index: ", index)
            image = self.associated_imgs[index]
            seg_mask = self.seg_mask[index]
            seg_img = np.zeros_like(image)
            seg_img[seg_mask == 1] = image[seg_mask == 1]
            plt.imshow(seg_img)
            plt.show()
